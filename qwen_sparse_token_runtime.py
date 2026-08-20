#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-Image 双流 Block 的真实 image-token 稀疏执行。

只对 active image token 计算 Q/K/V、attention 输出投影和 MLP。text token
始终完整计算；inactive image token 的旋转后 K/V 从上一 timestep 同层缓存
中复用，因此 active image query 仍然可以看到完整的 text+image 上下文。
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


SPARSE_RUNTIME_VERSION = "qwen_image_qkv_mlp_sparse_v2_1_decoupled_hard_budget"
ImageKVCache = Tuple[torch.Tensor, torch.Tensor]


def _require_attributes(module: Any, names: Sequence[str], label: str) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"当前Diffusers的{label}缺少属性{missing}，不能安全启用真实Token跳过。"
            "请使用包内inspect_qwen_runtime.py导出运行时信息后再适配。"
        )


def validate_sparse_block(block: torch.nn.Module) -> None:
    _require_attributes(
        block,
        (
            "img_mod",
            "txt_mod",
            "img_norm1",
            "txt_norm1",
            "img_norm2",
            "txt_norm2",
            "img_mlp",
            "txt_mlp",
            "attn",
            "_modulate",
        ),
        "QwenImageTransformerBlock",
    )
    _require_attributes(
        block.attn,
        (
            "to_q",
            "to_k",
            "to_v",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_out",
            "to_add_out",
            "heads",
        ),
        "Qwen双流Attention",
    )


def _resolve_rotary_function():
    module = importlib.import_module(
        "diffusers.models.transformers.transformer_qwenimage"
    )
    for name in ("apply_rotary_emb_qwen", "apply_rotary_emb"):
        function = getattr(module, name, None)
        if callable(function):
            return function
    raise RuntimeError("当前Diffusers中找不到Qwen rotary embedding函数。")


def _select_frequency_rows(value: Any, indices: torch.Tensor, full_length: int):
    if isinstance(value, torch.Tensor):
        candidate_dimensions = [
            dimension
            for dimension, size in enumerate(value.shape)
            if int(size) == int(full_length)
        ]
        if not candidate_dimensions:
            return value
        dimension = candidate_dimensions[0]
        return value.index_select(dimension, indices.to(value.device))
    if isinstance(value, tuple):
        return tuple(
            _select_frequency_rows(item, indices, full_length) for item in value
        )
    if isinstance(value, list):
        return [
            _select_frequency_rows(item, indices, full_length) for item in value
        ]
    return value


def _apply_rotary(function, tensor: torch.Tensor, frequencies: Any) -> torch.Tensor:
    attempts = ({"use_real": False}, {}, {"use_real": True})
    last_error: Optional[BaseException] = None
    for keyword_args in attempts:
        try:
            return function(tensor, frequencies, **keyword_args)
        except (TypeError, RuntimeError, ValueError) as error:
            last_error = error
    raise RuntimeError(
        "无法调用当前Diffusers的Qwen rotary embedding函数。"
    ) from last_error


def _project_image_key_value(
    attn: torch.nn.Module,
    image_states: torch.Tensor,
    image_rotary_emb: Optional[Tuple[Any, Any]],
    indices: Optional[torch.Tensor] = None,
) -> ImageKVCache:
    """计算完整或指定 image token 的旋转后 K/V。"""
    full_length = int(image_states.shape[1])
    selected_states = image_states
    selected_frequencies: Any = None
    if indices is not None:
        indices = indices.to(image_states.device, dtype=torch.long)
        selected_states = image_states.index_select(1, indices)

    image_key = attn.to_k(selected_states)
    image_value = attn.to_v(selected_states)
    heads = int(attn.heads)
    image_key = image_key.unflatten(-1, (heads, -1))
    image_value = image_value.unflatten(-1, (heads, -1))

    normalizer = getattr(attn, "norm_k", None)
    if normalizer is not None:
        image_key = normalizer(image_key)
    if image_rotary_emb is not None:
        image_frequencies, _ = image_rotary_emb
        selected_frequencies = (
            image_frequencies
            if indices is None
            else _select_frequency_rows(image_frequencies, indices, full_length)
        )
        image_key = _apply_rotary(
            _resolve_rotary_function(), image_key, selected_frequencies
        )
    return image_key, image_value


def _joint_attention_sparse_queries(
    block: torch.nn.Module,
    image_states: torch.Tensor,
    text_states: torch.Tensor,
    active_indices: torch.Tensor,
    image_rotary_emb: Optional[Tuple[Any, Any]],
    joint_attention_kwargs: Optional[Dict[str, Any]],
    cached_image_key: Optional[torch.Tensor],
    cached_image_value: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    attn = block.attn
    active_image_states = image_states.index_select(1, active_indices)

    image_query = attn.to_q(active_image_states)
    text_query = attn.add_q_proj(text_states)
    text_key = attn.add_k_proj(text_states)
    text_value = attn.add_v_proj(text_states)

    heads = int(attn.heads)
    image_query = image_query.unflatten(-1, (heads, -1))
    text_query = text_query.unflatten(-1, (heads, -1))
    text_key = text_key.unflatten(-1, (heads, -1))
    text_value = text_value.unflatten(-1, (heads, -1))

    for name, tensor_name in (
        ("norm_q", "image_query"),
        ("norm_added_q", "text_query"),
        ("norm_added_k", "text_key"),
    ):
        normalizer = getattr(attn, name, None)
        if normalizer is not None:
            if tensor_name == "image_query":
                image_query = normalizer(image_query)
            elif tensor_name == "text_query":
                text_query = normalizer(text_query)
            else:
                text_key = normalizer(text_key)

    if image_rotary_emb is not None:
        image_frequencies, text_frequencies = image_rotary_emb
        rotary = _resolve_rotary_function()
        active_frequencies = _select_frequency_rows(
            image_frequencies,
            active_indices,
            int(image_states.shape[1]),
        )
        image_query = _apply_rotary(rotary, image_query, active_frequencies)
        text_query = _apply_rotary(rotary, text_query, text_frequencies)
        text_key = _apply_rotary(rotary, text_key, text_frequencies)

    token_count = int(image_states.shape[1])
    kv_initialized = cached_image_key is None or cached_image_value is None
    if kv_initialized:
        # 第一轮稀疏执行没有上一轮Attention内部的旋转后K/V，使用当前已经
        # norm+modulate的完整image states建立一次缓存。之后只更新active位置。
        image_key, image_value = _project_image_key_value(
            attn,
            image_states,
            image_rotary_emb,
        )
        kv_initialization_token_count = token_count
    else:
        expected_prefix = (int(image_states.shape[0]), token_count, heads)
        if tuple(cached_image_key.shape[:3]) != expected_prefix:
            raise RuntimeError(
                "缓存image K形状与当前hidden不匹配："
                f"cache={tuple(cached_image_key.shape)}，expected_prefix={expected_prefix}。"
            )
        if tuple(cached_image_value.shape) != tuple(cached_image_key.shape):
            raise RuntimeError(
                "缓存image K/V形状不一致："
                f"K={tuple(cached_image_key.shape)}，V={tuple(cached_image_value.shape)}。"
            )
        # 推理阶段直接原位刷新active位置，避免每个Block复制完整K/V缓存。
        image_key = cached_image_key.to(image_states.device)
        image_value = cached_image_value.to(image_states.device)
        kv_initialization_token_count = 0

    if not kv_initialized:
        active_key, active_value = _project_image_key_value(
            attn,
            image_states,
            image_rotary_emb,
            indices=active_indices,
        )
        image_key.index_copy_(1, active_indices, active_key)
        image_value.index_copy_(1, active_indices, active_value)

    query = torch.cat((text_query, image_query), dim=1).transpose(1, 2)
    key = torch.cat((text_key, image_key), dim=1).transpose(1, 2)
    value = torch.cat((text_value, image_value), dim=1).transpose(1, 2)

    attention_kwargs = dict(joint_attention_kwargs or {})
    attention_mask = attention_kwargs.pop("attention_mask", None)
    attention_kwargs.pop("scale", None)
    unsupported = sorted(attention_kwargs)
    if unsupported:
        raise RuntimeError(
            "真实Token跳过暂不支持这些joint_attention_kwargs："
            + ", ".join(unsupported)
        )
    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)
    attended = attended.flatten(2, 3).to(query.dtype)

    text_length = int(text_states.shape[1])
    text_output = attended[:, :text_length]
    image_output = attended[:, text_length:]
    image_output = attn.to_out[0](image_output)
    if len(attn.to_out) > 1:
        image_output = attn.to_out[1](image_output)
    text_output = attn.to_add_out(text_output)
    kv_metadata = {
        "image_kv_cache_initialized": kv_initialized,
        "image_kv_initialization_token_count": kv_initialization_token_count,
        "computed_image_kv_token_count": (
            token_count if kv_initialized else int(active_indices.numel())
        ),
        "cached_image_kv_token_count": (
            0 if kv_initialized else token_count - int(active_indices.numel())
        ),
    }
    return image_output, text_output, image_key.detach(), image_value.detach(), kv_metadata


def sparse_qwen_block_forward(
    block: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: Optional[torch.Tensor],
    temb: torch.Tensor,
    active_indices: torch.Tensor,
    cached_image_residual: torch.Tensor,
    cached_image_key: Optional[torch.Tensor] = None,
    cached_image_value: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[Any, Any]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    modulate_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """返回(text_output, dense_image_output)，但只计算active image query/MLP。"""
    validate_sparse_block(block)
    if hidden_states.ndim != 3 or encoder_hidden_states.ndim != 3:
        raise RuntimeError("真实Token跳过要求text/image hidden形状均为[B,T,C]。")
    if int(hidden_states.shape[0]) != 1:
        raise RuntimeError("真实Token跳过v1目前只支持batch size=1。")
    if cached_image_residual.shape != hidden_states.shape:
        raise RuntimeError(
            "Token缓存残差形状与当前image hidden不一致："
            f"cache={tuple(cached_image_residual.shape)}，"
            f"hidden={tuple(hidden_states.shape)}。"
        )
    active_indices = active_indices.to(hidden_states.device, dtype=torch.long)
    if active_indices.ndim != 1 or active_indices.numel() == 0:
        raise RuntimeError("真实Token跳过至少需要计算一个image token。")

    image_mod1, image_mod2 = block.img_mod(temb).chunk(2, dim=-1)
    text_temb = temb
    if bool(getattr(block, "zero_cond_t", False)):
        if int(temb.shape[0]) != 2 * int(hidden_states.shape[0]):
            raise RuntimeError(
                "zero_cond_t要求temb batch等于image batch的2倍："
                f"temb={tuple(temb.shape)}，image={tuple(hidden_states.shape)}。"
            )
        text_temb = torch.chunk(temb, 2, dim=0)[0]
        if modulate_index is None:
            raise RuntimeError("zero_cond_t真实Token跳过缺少modulate_index。")
        if tuple(modulate_index.shape) != tuple(hidden_states.shape[:2]):
            raise RuntimeError(
                "modulate_index形状与image token不一致："
                f"index={tuple(modulate_index.shape)}，"
                f"image={tuple(hidden_states.shape[:2])}。"
            )
    text_mod1, text_mod2 = block.txt_mod(text_temb).chunk(2, dim=-1)

    image_modulate_kwargs: Dict[str, Any] = {}
    if modulate_index is not None:
        image_modulate_kwargs["index"] = modulate_index
    image_modulated, image_gate1 = block._modulate(
        block.img_norm1(hidden_states), image_mod1, **image_modulate_kwargs
    )
    text_modulated, text_gate1 = block._modulate(
        block.txt_norm1(encoder_hidden_states), text_mod1
    )
    (
        image_attention,
        text_attention,
        updated_image_key,
        updated_image_value,
        kv_metadata,
    ) = _joint_attention_sparse_queries(
        block,
        image_modulated,
        text_modulated,
        active_indices,
        image_rotary_emb,
        joint_attention_kwargs,
        cached_image_key,
        cached_image_value,
    )

    active_input = hidden_states.index_select(1, active_indices)
    active_gate1 = (
        image_gate1.index_select(1, active_indices)
        if int(image_gate1.shape[1]) == int(hidden_states.shape[1])
        else image_gate1
    )
    if int(active_gate1.shape[0]) != int(active_input.shape[0]):
        raise RuntimeError(
            "image attention gate batch未正确对齐："
            f"gate={tuple(active_gate1.shape)}，active={tuple(active_input.shape)}。"
        )
    active_hidden = active_input + active_gate1 * image_attention
    active_modulate_index = (
        None
        if modulate_index is None
        else modulate_index.index_select(1, active_indices.to(modulate_index.device))
    )
    active_modulate_kwargs: Dict[str, Any] = {}
    if active_modulate_index is not None:
        active_modulate_kwargs["index"] = active_modulate_index
    active_normed, image_gate2 = block._modulate(
        block.img_norm2(active_hidden), image_mod2, **active_modulate_kwargs
    )
    if int(image_gate2.shape[0]) != int(active_input.shape[0]):
        raise RuntimeError(
            "image MLP gate batch未正确对齐："
            f"gate={tuple(image_gate2.shape)}，active={tuple(active_input.shape)}。"
        )
    active_output = active_hidden + image_gate2 * block.img_mlp(active_normed)
    active_residual = active_output - active_input

    # 先用缓存残差生成dense输出，再仅覆盖active token，避免复制整份残差。
    image_output = hidden_states + cached_image_residual.to(hidden_states.device)
    image_output.index_copy_(1, active_indices, active_output)

    text_hidden = encoder_hidden_states + text_gate1 * text_attention
    text_normed, text_gate2 = block._modulate(
        block.txt_norm2(text_hidden), text_mod2
    )
    text_output = text_hidden + text_gate2 * block.txt_mlp(text_normed)
    if text_output.dtype == torch.float16:
        text_output = text_output.clip(-65504, 65504)
    if image_output.dtype == torch.float16:
        image_output = image_output.clip(-65504, 65504)
    return (
        text_output,
        image_output,
        updated_image_key,
        updated_image_value,
        kv_metadata,
    )
