#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit-2511：逐 timestep 搜索连续 Block 执行窗口。

严格执行规则
------------
假设模型共有 60 个 Block：

1. 第一个 timestep 强制完整执行 Block 1-60，并保存每个 Block 的双流层内
   残差（Block输出减去Block输入）。
2. 后续 timestep 的 Block 1 和 Block 60 必须执行，候选连续窗口只在
   Block 2-59 中滑动。
3. 如果当前候选窗口为 4-8（下文均用 1-based），则当前 timestep：

       执行：1, 4, 5, 6, 7, 8, 60
       缓存：2, 3, 9, 10, ..., 59

4. 被跳过的 Block 不做跨层插值，也不读取上一个 timestep 的绝对输出。
   它把“上一个 timestep 同编号 Block”的层内残差加到当前输入：

       text_delta[t-1, i]  = text_out[t-1, i]  - text_in[t-1, i]
       image_delta[t-1, i] = image_out[t-1, i] - image_in[t-1, i]

       text_out[t, i]  = text_in[t, i]  + text_delta[t-1, i]
       image_out[t, i] = image_in[t, i] + image_delta[t-1, i]

每个 CFG 分支维护独立残差缓存。某层如果连续多个 timestep 都未执行，其
缓存保持为该层最近一次实际执行时得到的层内残差。当前 hidden 始终逐层
向前传递，不会被上一 timestep 的绝对 hidden 覆盖。

搜索过程
--------
1. 第一个 timestep 只完整执行，不搜索候选窗口，并建立所有 Block 缓存。
2. 后续每个 timestep 先执行一次完整教师 Transformer。
3. 在相同 timestep、相同 Transformer 输入上，逐个测试固定长度的连续窗口；
   候选跳过层使用上一个教师 timestep 同编号 Block 的层内残差。
4. 候选与完整教师比较：
   - 最后一个 Block 的生成图 image token 相对 MSE；
   - 最后一个 Block 的 text token 相对 MSE；
   - Transformer 噪声预测相对 MSE。
5. 从第二个 timestep 开始选择综合误差最小的窗口。
6. 最后按所有 timestep 的最优窗口重新运行一次完整 pipeline，输出最终图片
   以及相对完整基线的图像误差。

主要输出
--------
- baseline_full.png：完整模型生成结果。
- candidate_scores.csv：每步每个候选的执行/跳过列表和聚合误差。
- candidate_layer_matrix.csv：每个候选 × 每个 Block 的 0/1 执行矩阵。
- candidate_branch_details.json：每个 CFG 分支的原始误差。
- best_schedule.json：逐 timestep 最优窗口、执行/跳过列表和误差。
- best_schedule_layer_matrix.csv：最优路径的 timestep × Block 执行矩阵。
- diagonal_bridge_best.png：按最优路径生成的最终图片。
- final_image_metrics.json：最终图片相对 baseline 的 MAE/MSE/PSNR 等。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qwen_sparse_token_runtime import (
    ImageKVCache,
    SPARSE_RUNTIME_VERSION,
    sparse_qwen_block_forward,
)

# 本文件来自旧蓝图运行时，只保留蓝图 Block cache 与低层 sparse token 执行。
# 新版本完全不使用旧 qwen_token_rl.py。下面两个占位符仅用于兼容原类定义；
# 只要新主程序不把 token_policy_mode 设成旧的 "rl"，它们就不会被调用。
class TokenPolicyRuntime:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("旧 Token RL 已禁用；请使用新主程序中的 BlockBudgetActorCritic。")

def build_token_features(*args, **kwargs):
    raise RuntimeError("旧 Token RL 特征已禁用；请使用新主程序中的 BlockBudgetActorCritic。")


try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError as import_error:
    raise ImportError(
        "当前环境没有 QwenImageEditPlusPipeline。请在已经配置好的 "
        "MMDITModelCompression Conda 环境中运行本脚本。"
    ) from import_error


TensorPair = Tuple[torch.Tensor, torch.Tensor]  # (text token, image token)
BlockCache = Dict[int, TensorPair]  # layer -> (text residual, image residual)
Window = Tuple[int, int]  # 0-based 闭区间
CACHE_STRATEGY_VERSION = (
    "dynamic_window_compute_penalty_previous_step_same_block_residual_cache_v3"
)
BLUE_LINE_CACHE_STRATEGY_VERSION = (
    "blue_line_profiled_previous_step_same_block_residual_cache_v4"
)
BLUE_LINE_TOKEN_CACHE_STRATEGY_VERSION = (
    "blue_line_block_cache_plus_fixed_ratio_image_token_cache_v1"
)
BLUE_LINE_SPARSE_TOKEN_CACHE_STRATEGY_VERSION = (
    "blue_line_global_budget_token_actor_critic_qkv_cache_v4_2"
)


class GlobalTokenBudgetState:
    """单个 CFG 分支内的全局 token 计算预算。

    total_budget 只约束 mixed token cache 单元；首步初始化、整体跳过 block、
    以及因形状异常导致的 full refresh 不计入全局 mixed 预算。
    """

    def __init__(
        self,
        *,
        total_cells: int,
        token_count: int,
        compute_ratio: float,
        min_compute_ratio: float,
    ) -> None:
        self.total_cells = max(1, int(total_cells))
        self.token_count = max(1, int(token_count))
        self.compute_ratio = float(compute_ratio)
        self.min_compute_ratio = float(min_compute_ratio)
        self.total_units = self.total_cells * self.token_count
        self.total_budget = int(math.ceil(self.total_units * self.compute_ratio))
        self.used_budget = 0
        self.processed_cells = 0
        self.priority_ema: Optional[float] = None

    def allocate(
        self,
        *,
        priority: float,
        min_compute: int,
    ) -> Tuple[int, Dict[str, Any]]:
        remaining_cells = self.total_cells - self.processed_cells
        if remaining_cells <= 0:
            raise RuntimeError("全局Token预算分配次数超过预期。")
        remaining_budget = self.total_budget - self.used_budget
        cells_after = remaining_cells - 1
        min_compute = max(1, min(int(min_compute), self.token_count))
        min_after = cells_after * min_compute
        max_after = cells_after * self.token_count
        lower = max(1, remaining_budget - max_after)
        upper = min(self.token_count, remaining_budget - min_after)
        if upper < lower:
            # 如果用户设置的 min ratio 与总预算冲突，优先保证总预算可完成。
            lower = max(0, min(self.token_count, remaining_budget - max_after))
            upper = min(self.token_count, remaining_budget)
        if cells_after == 0:
            budget = max(0, min(self.token_count, remaining_budget))
        else:
            base = remaining_budget / remaining_cells
            current_priority = max(1e-6, float(priority))
            if self.priority_ema is None:
                self.priority_ema = current_priority
            relative = max(0.35, min(2.25, current_priority / max(1e-6, self.priority_ema)))
            desired = int(round(base * relative))
            budget = max(lower, min(upper, desired))
            self.priority_ema = 0.90 * self.priority_ema + 0.10 * current_priority
        self.used_budget += int(budget)
        self.processed_cells += 1
        return int(budget), {
            "global_token_budget_total_cells": self.total_cells,
            "global_token_budget_cell_index_0based": self.processed_cells - 1,
            "global_token_budget_total_units": self.total_units,
            "global_token_budget_total_compute_units": self.total_budget,
            "global_token_budget_used_compute_units": self.used_budget,
            "global_token_budget_remaining_compute_units": (
                self.total_budget - self.used_budget
            ),
            "global_token_budget_priority": float(priority),
            "global_token_budget_priority_ema": (
                None if self.priority_ema is None else float(self.priority_ema)
            ),
            "global_token_budget_lower_bound": int(lower),
            "global_token_budget_upper_bound": int(upper),
        }


def compact_index_ranges(indices: Iterable[int], one_based: bool = True) -> str:
    """把离散索引压缩为1-3,7,9-12，避免Token明细CSV过大。"""
    values = sorted({int(value) for value in indices})
    if not values:
        return ""
    ranges: List[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        left = start + 1 if one_based else start
        right = previous + 1 if one_based else previous
        ranges.append(str(left) if left == right else f"{left}-{right}")
        start = previous = value
    left = start + 1 if one_based else start
    right = previous + 1 if one_based else previous
    ranges.append(str(left) if left == right else f"{left}-{right}")
    return ",".join(ranges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "第一个timestep完整计算；后续逐timestep穷举连续中间Block窗口；"
            "首尾Block必算，跳过层把上一timestep同编号Block层内残差"
            "加到当前输入。"
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data4/guowenwu/MMDITModelCompression/models/Qwen-Image-Edit-2511",
        help="本地 Qwen-Image-Edit-2511 Diffusers 模型目录。",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        nargs="+",
        required=True,
        help="一张或多张编辑参考图。",
    )
    parser.add_argument("--prompt", type=str, required=True, help="编辑指令。")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=" ",
        help="负面提示词；默认是一个空格。",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=4,
        help="去噪步数；当前压缩测试默认 4 步。",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help=(
            "每个 timestep 正常执行多少个连续中间 Block。"
            "例如设置5，可以测试4-8这种5层窗口；首尾 Block 不计入这5层。"
        ),
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="候选窗口滑动步长；1表示从Block 2开始逐段穷举。",
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=1.0,
        help="Qwen true CFG强度；压缩快速实验默认1.0。",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="传给Transformer guidance embedding的强度。",
    )
    parser.add_argument(
        "--forwards-per-step",
        type=int,
        choices=[1, 2],
        default=None,
        help=(
            "每个 timestep 调用 Transformer 的次数。不填写时根据 true CFG 推断；"
            "true_cfg_scale>1 且有 negative_prompt 时推断为2，否则为1。"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="固定随机种子。")
    parser.add_argument("--width", type=int, default=None, help="可选输出宽度。")
    parser.add_argument("--height", type=int, default=None, help="可选输出高度。")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="模型推理精度；H20推荐bf16。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="运行设备，例如cuda:0。",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="启用Diffusers model CPU offload；穷举搜索会明显变慢。",
    )
    parser.add_argument(
        "--noise-weight",
        type=float,
        default=1.0,
        help="Transformer噪声预测相对MSE的评分权重。",
    )
    parser.add_argument(
        "--image-token-weight",
        type=float,
        default=1.0,
        help="末层生成图image token相对MSE的评分权重。",
    )
    parser.add_argument(
        "--text-token-weight",
        type=float,
        default=0.25,
        help="末层text token相对MSE的评分权重。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./qwen_edit_diagonal_bridge_residual_cache_v3_outputs",
        help="结果保存目录。",
    )
    parser.add_argument(
        "--skip-final-run",
        action="store_true",
        help="只搜索每步最优窗口，不运行最终组合路径。",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="显示Diffusers去噪进度条。",
    )
    parser.add_argument(
        "--verbose-candidates",
        action="store_true",
        help="在终端打印每个候选窗口误差；默认只打印每步最优结果。",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help=(
            "搜索时每完成多少个候选打印一次简洁进度、耗时和ETA；"
            "默认25，设为0只保留step开始/完成日志。"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_path}")
    for image_path_text in args.input_image:
        image_path = Path(image_path_text)
        if not image_path.is_file():
            raise FileNotFoundError(f"输入图片不存在：{image_path}")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps必须大于0。")
    if args.window_size <= 0:
        raise ValueError("--window-size必须大于0。")
    if args.window_stride <= 0:
        raise ValueError("--window-stride必须大于0。")
    if args.progress_every < 0:
        raise ValueError("--progress-every不能小于0。")
    weights = (
        args.noise_weight,
        args.image_token_weight,
        args.text_token_weight,
    )
    if min(weights) < 0:
        raise ValueError("误差权重不能为负数。")
    if sum(weights) <= 0:
        raise ValueError("至少需要一个大于0的误差权重。")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Qwen-Image-Edit-2511测试必须使用CUDA设备。")
    if not torch.cuda.is_available():
        raise RuntimeError("当前Python环境没有检测到CUDA。")


def load_input_images(image_paths: Sequence[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as opened_image:
            images.append(opened_image.convert("RGB").copy())
    return images


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def infer_forwards_per_step(args: argparse.Namespace) -> int:
    if args.forwards_per_step is not None:
        return int(args.forwards_per_step)
    true_cfg_enabled = args.true_cfg_scale > 1.0 and args.negative_prompt is not None
    return 2 if true_cfg_enabled else 1


def build_candidate_windows(
    total_layers: int,
    window_size: int,
    stride: int,
) -> List[Window]:
    """
    Block 0和Block N-1是强制层，候选窗口只在1..N-2中滑动。
    """
    internal_layers = total_layers - 2
    if internal_layers <= 0:
        raise ValueError(f"模型只有{total_layers}层，无法进行中间层窗口测试。")
    if window_size > internal_layers:
        raise ValueError(
            f"--window-size={window_size}超过中间层总数{internal_layers}。"
        )
    first_start = 1
    last_start = total_layers - 1 - window_size
    starts = list(range(first_start, last_start + 1, stride))
    if not starts:
        raise RuntimeError("没有构造出任何候选窗口。")
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + window_size - 1) for start in starts]


def executed_layers_for_window(total_layers: int, window: Window) -> Set[int]:
    start_layer, end_layer = window
    return {0, total_layers - 1, *range(start_layer, end_layer + 1)}


def skipped_layers_for_window(total_layers: int, window: Window) -> List[int]:
    executed = executed_layers_for_window(total_layers, window)
    return [layer for layer in range(total_layers) if layer not in executed]


def one_based_layer_string(layers: Iterable[int]) -> str:
    return ",".join(str(layer + 1) for layer in layers)


def read_block_inputs(
    positional_args: Tuple[Any, ...],
    keyword_args: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回(image hidden, text hidden)。"""
    hidden_states = keyword_args.get("hidden_states")
    encoder_hidden_states = keyword_args.get("encoder_hidden_states")
    if hidden_states is None and len(positional_args) >= 1:
        hidden_states = positional_args[0]
    if encoder_hidden_states is None and len(positional_args) >= 2:
        encoder_hidden_states = positional_args[1]
    if not isinstance(hidden_states, torch.Tensor):
        raise RuntimeError("无法从Block.forward读取image hidden_states。")
    if not isinstance(encoder_hidden_states, torch.Tensor):
        raise RuntimeError("无法从Block.forward读取text encoder_hidden_states。")
    return hidden_states, encoder_hidden_states


def split_block_output(output: Any) -> TensorPair:
    """
    QwenImageTransformerBlock返回：
    (encoder_hidden_states, hidden_states)，即(text, image)。
    """
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise RuntimeError(
            "Qwen Block.forward返回值不是预期的双流tuple/list；"
            "请检查当前Diffusers版本。"
        )
    text_output, image_output = output[0], output[1]
    if not isinstance(text_output, torch.Tensor):
        raise RuntimeError("Block返回的text输出不是Tensor。")
    if not isinstance(image_output, torch.Tensor):
        raise RuntimeError("Block返回的image输出不是Tensor。")
    return text_output, image_output


def replace_block_output(
    output: Any,
    text_output: torch.Tensor,
    image_output: torch.Tensor,
) -> Any:
    """替换双流Block的前两个返回值，同时保留潜在的附加返回项。"""
    if isinstance(output, tuple):
        return (text_output, image_output, *output[2:])
    if isinstance(output, list):
        return [text_output, image_output, *output[2:]]
    raise RuntimeError("无法替换非tuple/list的Qwen Block输出。")


@contextmanager
def install_block_policy(
    transformer_blocks: Sequence[torch.nn.Module],
    original_block_forwards: Sequence[Callable[..., Any]],
    executed_layers: Set[int],
    previous_step_cache: Optional[BlockCache] = None,
    capture_all_layers: bool = False,
    image_token_policy: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]] = None,
):
    """
    executed_layers正常计算；其他层把上一timestep同编号Block的层内残差
    加到当前输入。这样当前timestep的hidden会连续向前传播，不会被旧的
    绝对输出覆盖。

    capture_all_layers=True时记录本次forward的每层双流层内残差，用于建立
    下一timestep缓存；否则只记录最后一个Block的真实输出，供候选误差计算。
    """
    total_layers = len(transformer_blocks)
    last_layer = total_layers - 1
    captured_last: Dict[str, Optional[TensorPair]] = {"tokens": None}
    captured_layers: BlockCache = {}
    captured_token_metadata: Dict[int, Dict[str, Any]] = {}

    if previous_step_cache is not None:
        missing_cached_layers = [
            layer_index
            for layer_index in range(total_layers)
            if (
                layer_index not in executed_layers
                and layer_index not in previous_step_cache
            )
        ]
        if missing_cached_layers:
            raise RuntimeError(
                "上一timestep缓存不完整，缺少Block："
                f"{one_based_layer_string(missing_cached_layers)}。"
                "第一个timestep必须完整执行。"
            )

    def make_policy_forward(layer_index: int):
        original_forward = original_block_forwards[layer_index]

        def policy_forward(_block_self, *positional_args, **keyword_args):
            image_input, text_input = read_block_inputs(
                positional_args,
                keyword_args,
            )
            if layer_index in executed_layers:
                sparse_executor = getattr(
                    image_token_policy, "execute_sparse", None
                )
                if callable(sparse_executor):
                    output, metadata = sparse_executor(
                        block=_block_self,
                        original_forward=original_forward,
                        positional_args=positional_args,
                        keyword_args=keyword_args,
                        layer_index=layer_index,
                    )
                    captured_token_metadata[layer_index] = metadata
                else:
                    output = original_forward(*positional_args, **keyword_args)
                text_output, image_output = split_block_output(output)
                text_residual = text_output - text_input
                image_residual = image_output - image_input
                if image_token_policy is not None and not callable(sparse_executor):
                    image_output, image_residual, metadata = image_token_policy(
                        layer_index=layer_index,
                        image_input=image_input,
                        text_input=text_input,
                        text_output=text_output,
                        image_output=image_output,
                        text_residual=text_residual,
                        image_residual=image_residual,
                    )
                    captured_token_metadata[layer_index] = metadata
                    output = replace_block_output(output, text_output, image_output)
            else:
                if previous_step_cache is None:
                    raise RuntimeError(
                        f"Block {layer_index + 1}需要上一timestep残差缓存，"
                        "但缓存尚未建立。"
                    )
                text_residual, image_residual = previous_step_cache[layer_index]
                text_output = text_input + text_residual
                image_output = image_input + image_residual
                output = (text_output, image_output)

            detached_residuals = (
                text_residual.detach(),
                image_residual.detach(),
            )
            if capture_all_layers:
                captured_layers[layer_index] = detached_residuals
            if layer_index == last_layer:
                captured_last["tokens"] = (
                    text_output.detach(),
                    image_output.detach(),
                )
            return output

        return policy_forward

    try:
        for layer_index, block in enumerate(transformer_blocks):
            block.forward = types.MethodType(make_policy_forward(layer_index), block)
        yield {
            "last_tokens": captured_last,
            "layer_cache": captured_layers,
            "token_metadata": captured_token_metadata,
        }
    finally:
        for block, original_forward in zip(
            transformer_blocks,
            original_block_forwards,
        ):
            block.forward = original_forward


def extract_transformer_sample(output: Any) -> torch.Tensor:
    if hasattr(output, "sample") and isinstance(output.sample, torch.Tensor):
        return output.sample
    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    if isinstance(output, torch.Tensor):
        return output
    raise RuntimeError("无法从Transformer返回值读取噪声预测sample。")


def relative_mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"误差Tensor形状不一致：reference={tuple(reference.shape)}，"
            f"candidate={tuple(candidate.shape)}。"
        )
    if reference.device != candidate.device:
        reference = reference.to(candidate.device)
    difference = candidate.float() - reference.float()
    numerator = difference.square().mean()
    denominator = reference.float().square().mean().clamp_min(1e-12)
    return float((numerator / denominator).item())


def score_candidate(
    teacher_output: Any,
    candidate_output: Any,
    teacher_last_tokens: TensorPair,
    candidate_last_tokens: TensorPair,
    args: argparse.Namespace,
) -> Dict[str, float]:
    teacher_text, teacher_image = teacher_last_tokens
    candidate_text, candidate_image = candidate_last_tokens
    teacher_sample = extract_transformer_sample(teacher_output)
    candidate_sample = extract_transformer_sample(candidate_output)

    noise_error = relative_mse(teacher_sample, candidate_sample)

    # 最后一层image stream一般包含“生成图token+参考图token”，而sample通常
    # 只保留生成图部分。形状允许时只比较生成图token，防止参考图token稀释误差。
    if (
        teacher_sample.ndim == 3
        and teacher_image.ndim == 3
        and candidate_image.ndim == 3
        and teacher_sample.shape[1] <= teacher_image.shape[1]
        and teacher_sample.shape[1] <= candidate_image.shape[1]
    ):
        generated_tokens = int(teacher_sample.shape[1])
        teacher_image_for_score = teacher_image[:, :generated_tokens]
        candidate_image_for_score = candidate_image[:, :generated_tokens]
    else:
        teacher_image_for_score = teacher_image
        candidate_image_for_score = candidate_image

    image_token_error = relative_mse(
        teacher_image_for_score,
        candidate_image_for_score,
    )
    text_token_error = relative_mse(teacher_text, candidate_text)
    total_score = (
        args.noise_weight * noise_error
        + args.image_token_weight * image_token_error
        + args.text_token_weight * text_token_error
    )
    return {
        "noise_relative_mse": noise_error,
        "image_token_relative_mse": image_token_error,
        "text_token_relative_mse": text_token_error,
        "score": total_score,
    }


def residual_change_metrics(
    previous: torch.Tensor,
    current: torch.Tensor,
) -> Dict[str, float]:
    """计算相邻timestep同层残差的变化，全部归约在当前设备上完成。"""
    if previous.shape != current.shape:
        raise RuntimeError(
            "相邻timestep残差形状不一致："
            f"previous={tuple(previous.shape)}，current={tuple(current.shape)}。"
        )
    previous_float = previous.float()
    current_float = current.float()
    difference = current_float - previous_float
    previous_energy = previous_float.square().mean().clamp_min(1e-12)
    current_energy = current_float.square().mean().clamp_min(1e-12)
    difference_energy = difference.square().mean()
    dot = (previous_float * current_float).mean()
    cosine = dot / torch.sqrt(previous_energy * current_energy).clamp_min(1e-12)
    return {
        "relative_l2": float(torch.sqrt(difference_energy / previous_energy).item()),
        "cosine_similarity": float(cosine.clamp(-1.0, 1.0).item()),
        "difference_rms": float(torch.sqrt(difference_energy).item()),
        "previous_residual_rms": float(torch.sqrt(previous_energy).item()),
        "current_residual_rms": float(torch.sqrt(current_energy).item()),
    }


class ResidualProfileController:
    """完整运行一次pipeline，记录每个step×block的双流残差变化。"""

    def __init__(
        self,
        transformer_blocks: Sequence[torch.nn.Module],
        original_transformer_forward: Callable[..., Any],
        args: argparse.Namespace,
        forwards_per_step: int,
    ) -> None:
        self.blocks = list(transformer_blocks)
        self.original_transformer_forward = original_transformer_forward
        self.original_block_forwards = [block.forward for block in self.blocks]
        self.args = args
        self.forwards_per_step = forwards_per_step
        self.total_layers = len(self.blocks)
        self.expected_calls = int(args.num_inference_steps) * forwards_per_step
        self.call_index = 0
        self.previous_caches: Dict[int, BlockCache] = {}
        self.rows: List[Dict[str, Any]] = []

    def __call__(self, *positional_args, **keyword_args):
        if self.call_index >= self.expected_calls:
            raise RuntimeError("残差统计Transformer forward次数超过预期。")
        step_index = self.call_index // self.forwards_per_step
        branch_index = self.call_index % self.forwards_per_step
        # print(
        #     f"[profile][{getattr(self.args, 'device', 'cuda')}]"
        #     f"[sample {int(getattr(self.args, 'sample_index', 0)):05d}] "
        #     f"step={step_index + 1}/{self.args.num_inference_steps}，"
        #     f"branch={branch_index + 1}/{self.forwards_per_step}："
        #     "完整计算并统计逐层残差变化",
        #     flush=True,
        # )
        executed = set(range(self.total_layers))
        with install_block_policy(
            transformer_blocks=self.blocks,
            original_block_forwards=self.original_block_forwards,
            executed_layers=executed,
            previous_step_cache=None,
            capture_all_layers=True,
        ) as captured:
            output = self.original_transformer_forward(
                *positional_args,
                **keyword_args,
            )
        current_cache: BlockCache = captured["layer_cache"]
        if len(current_cache) != self.total_layers:
            raise RuntimeError("完整轨迹没有采集到全部Block残差。")

        previous_cache = self.previous_caches.get(branch_index)
        for layer_index in range(self.total_layers):
            text_current, image_current = current_cache[layer_index]
            row: Dict[str, Any] = {
                "step_index_0based": step_index,
                "step_number_1based": step_index + 1,
                "branch_index_0based": branch_index,
                "branch_number_1based": branch_index + 1,
                "block_index_0based": layer_index,
                "block_number_1based": layer_index + 1,
            }
            if previous_cache is None:
                row.update(
                    {
                        "image_relative_l2": None,
                        "image_cosine_similarity": None,
                        "image_difference_rms": None,
                        "image_previous_residual_rms": None,
                        "image_current_residual_rms": float(
                            torch.sqrt(image_current.float().square().mean()).item()
                        ),
                        "text_relative_l2": None,
                        "text_cosine_similarity": None,
                        "text_difference_rms": None,
                        "text_previous_residual_rms": None,
                        "text_current_residual_rms": float(
                            torch.sqrt(text_current.float().square().mean()).item()
                        ),
                    }
                )
            else:
                text_previous, image_previous = previous_cache[layer_index]
                image_metrics = residual_change_metrics(image_previous, image_current)
                text_metrics = residual_change_metrics(text_previous, text_current)
                row.update(
                    {
                        f"image_{key}": value
                        for key, value in image_metrics.items()
                    }
                )
                row.update(
                    {
                        f"text_{key}": value
                        for key, value in text_metrics.items()
                    }
                )
            self.rows.append(row)

        self.previous_caches[branch_index] = current_cache
        self.call_index += 1
        return output

    def validate_complete(self) -> None:
        if self.call_index != self.expected_calls:
            raise RuntimeError(
                f"残差统计实际调用{self.call_index}次，预期{self.expected_calls}次。"
            )


class FullReferenceController:
    """完整运行并保留逐step教师输出，供随后缓存轨迹逐step对比。"""

    def __init__(
        self,
        transformer_blocks: Sequence[torch.nn.Module],
        original_transformer_forward: Callable[..., Any],
        args: argparse.Namespace,
        forwards_per_step: int,
    ) -> None:
        self.blocks = list(transformer_blocks)
        self.original_transformer_forward = original_transformer_forward
        self.original_block_forwards = [block.forward for block in self.blocks]
        self.args = args
        self.forwards_per_step = forwards_per_step
        self.total_layers = len(self.blocks)
        self.expected_calls = int(args.num_inference_steps) * forwards_per_step
        self.call_index = 0
        self.references: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def __call__(self, *positional_args, **keyword_args):
        if self.call_index >= self.expected_calls:
            raise RuntimeError("教师Transformer forward次数超过预期。")
        step_index = self.call_index // self.forwards_per_step
        branch_index = self.call_index % self.forwards_per_step
        print(
            f"[baseline][{getattr(self.args, 'device', 'cuda')}]"
            f"[sample {int(getattr(self.args, 'sample_index', 0)):05d}] "
            f"step={step_index + 1}/{self.args.num_inference_steps}，"
            f"branch={branch_index + 1}/{self.forwards_per_step}：完整计算",
            flush=True,
        )
        with install_block_policy(
            transformer_blocks=self.blocks,
            original_block_forwards=self.original_block_forwards,
            executed_layers=set(range(self.total_layers)),
            previous_step_cache=None,
            capture_all_layers=False,
        ) as captured:
            output = self.original_transformer_forward(
                *positional_args,
                **keyword_args,
            )
        last_tokens = captured["last_tokens"]["tokens"]
        if last_tokens is None:
            raise RuntimeError("教师轨迹没有记录到最后一层token。")
        # Teacher reference 只用于随后 fixed/RL 的误差比较，不参与后续模型 forward。
        # 旧版把 50 个 timestep 的 sample/last_tokens 全部 clone 在 GPU，训练第一张图时
        # 会和 token/block cache 叠加造成很高显存峰值。这里默认立即转到 CPU；
        # score_candidate/relative_mse 会在真正比较当前 step 时只把对应 teacher tensor
        # 临时搬回 candidate 所在 GPU，因此不会同时常驻 50 个 timestep 的 teacher tensor。
        teacher_sample = extract_transformer_sample(output).detach().cpu().clone()
        teacher_text, teacher_image = last_tokens
        self.references[(step_index, branch_index)] = {
            "sample": teacher_sample,
            "last_tokens": (
                teacher_text.detach().cpu().clone(),
                teacher_image.detach().cpu().clone(),
            ),
        }
        self.call_index += 1
        return output

    def validate_complete(self) -> None:
        if self.call_index != self.expected_calls:
            raise RuntimeError(
                f"教师轨迹实际调用{self.call_index}次，预期{self.expected_calls}次。"
            )


class BlueLineScheduledController:
    """执行离线生成的蓝线schedule，并详细记录逐step误差和逐Block缓存来源。"""

    def __init__(
        self,
        transformer_blocks: Sequence[torch.nn.Module],
        original_transformer_forward: Callable[..., Any],
        schedule: Dict[int, Dict[str, Any]],
        teacher_references: Dict[Tuple[int, int], Dict[str, Any]],
        args: argparse.Namespace,
        forwards_per_step: int,
    ) -> None:
        self.blocks = list(transformer_blocks)
        self.original_transformer_forward = original_transformer_forward
        self.original_block_forwards = [block.forward for block in self.blocks]
        self.schedule = schedule
        self.teacher_references = teacher_references
        self.args = args
        self.forwards_per_step = forwards_per_step
        self.total_layers = len(self.blocks)
        self.expected_calls = int(args.num_inference_steps) * forwards_per_step
        self.call_index = 0
        self.previous_caches: Dict[int, BlockCache] = {}
        self.last_refresh_steps: Dict[int, Dict[int, int]] = {}
        self.branch_step_rows: List[Dict[str, Any]] = []
        self.block_action_rows: List[Dict[str, Any]] = []

    def __call__(self, *positional_args, **keyword_args):
        if self.call_index >= self.expected_calls:
            raise RuntimeError("蓝线策略Transformer forward次数超过预期。")
        step_index = self.call_index // self.forwards_per_step
        branch_index = self.call_index % self.forwards_per_step
        item = self.schedule[step_index]
        executed = {
            int(layer) for layer in item["executed_blocks_0based"]
        }
        if step_index == 0:
            executed = set(range(self.total_layers))
            previous_cache = None
        else:
            previous_cache = self.previous_caches.get(branch_index)
            if previous_cache is None:
                raise RuntimeError(
                    f"蓝线step={step_index + 1}缺少上一step缓存。"
                )

        with install_block_policy(
            transformer_blocks=self.blocks,
            original_block_forwards=self.original_block_forwards,
            executed_layers=executed,
            previous_step_cache=previous_cache,
            capture_all_layers=True,
        ) as captured:
            output = self.original_transformer_forward(
                *positional_args,
                **keyword_args,
            )
        last_tokens = captured["last_tokens"]["tokens"]
        current_cache: BlockCache = captured["layer_cache"]
        if last_tokens is None or len(current_cache) != self.total_layers:
            raise RuntimeError("蓝线策略没有采集到完整缓存或最后层token。")

        teacher = self.teacher_references.get((step_index, branch_index))
        if teacher is None:
            raise RuntimeError(
                f"缺少step={step_index + 1}、branch={branch_index + 1}教师输出。"
            )
        metrics = score_candidate(
            teacher_output=teacher["sample"],
            candidate_output=output,
            teacher_last_tokens=teacher["last_tokens"],
            candidate_last_tokens=last_tokens,
            args=self.args,
        )
        skipped = sorted(set(range(self.total_layers)) - executed)
        print(
            f"[blue-line][{getattr(self.args, 'device', 'cuda')}]"
            f"[sample {int(getattr(self.args, 'sample_index', 0)):05d}] "
            f"step={step_index + 1}/{self.args.num_inference_steps}，"
            f"branch={branch_index + 1}/{self.forwards_per_step}："
            f"执行{len(executed)}层，跳过{len(skipped)}层；"
            f"执行=[{one_based_layer_string(sorted(executed))}]；"
            f"跳过=[{one_based_layer_string(skipped)}]",
            flush=True,
        )
        self.branch_step_rows.append(
            {
                "step_index_0based": step_index,
                "step_number_1based": step_index + 1,
                "branch_index_0based": branch_index,
                "branch_number_1based": branch_index + 1,
                "executed_block_count": len(executed),
                "skipped_block_count": len(skipped),
                "executed_blocks_1based": one_based_layer_string(sorted(executed)),
                "skipped_blocks_1based": one_based_layer_string(skipped),
                **metrics,
            }
        )

        refresh_steps = self.last_refresh_steps.setdefault(branch_index, {})
        for layer_index in range(self.total_layers):
            if layer_index in executed:
                source_step = step_index
                refresh_steps[layer_index] = step_index
                action = "execute"
                cache_age = 0
            else:
                source_step = refresh_steps.get(layer_index)
                if source_step is None:
                    raise RuntimeError(
                        f"Block {layer_index + 1}被跳过但没有真实缓存来源。"
                    )
                action = "cache"
                cache_age = step_index - source_step
            self.block_action_rows.append(
                {
                    "step_index_0based": step_index,
                    "step_number_1based": step_index + 1,
                    "branch_index_0based": branch_index,
                    "branch_number_1based": branch_index + 1,
                    "block_index_0based": layer_index,
                    "block_number_1based": layer_index + 1,
                    "action": action,
                    "cache_source_step_index_0based": source_step,
                    "cache_source_step_number_1based": source_step + 1,
                    "cache_age": cache_age,
                    "base_blue_line_action": (
                        "cache"
                        if layer_index in set(item.get("base_skipped_blocks_0based", []))
                        else "execute"
                    ),
                    "forced_refresh": bool(
                        layer_index in set(item.get("forced_refresh_blocks_0based", []))
                    ),
                }
            )

        self.previous_caches[branch_index] = current_cache
        self.call_index += 1
        return output

    def validate_complete(self) -> None:
        if self.call_index != self.expected_calls:
            raise RuntimeError(
                f"蓝线策略实际调用{self.call_index}次，预期{self.expected_calls}次。"
            )


def _read_forward_argument(
    positional_args: Tuple[Any, ...],
    keyword_args: Dict[str, Any],
    name: str,
    position: int,
    default: Any = None,
) -> Any:
    if name in keyword_args:
        return keyword_args[name]
    if len(positional_args) > position:
        return positional_args[position]
    return default


class _TokenExecutionPolicy:
    def __init__(
        self,
        controller: "BlueLineTokenScheduledController",
        step_index: int,
        branch_index: int,
        item: Dict[str, Any],
        previous_cache: Optional[BlockCache],
    ) -> None:
        self.controller = controller
        self.step_index = step_index
        self.branch_index = branch_index
        self.item = item
        self.previous_cache = previous_cache

    def __call__(self, **values):
        image_input: torch.Tensor = values["image_input"]
        image_output: torch.Tensor = values["image_output"]
        image_residual: torch.Tensor = values["image_residual"]
        decision = self.controller._prepare_token_decision(
            step_index=self.step_index,
            branch_index=self.branch_index,
            item=self.item,
            previous_cache=self.previous_cache,
            layer_index=int(values["layer_index"]),
            image_input=image_input,
        )
        metadata = decision["metadata"]
        cached_mask_cpu = decision["cached_mask_cpu"]
        if not bool(cached_mask_cpu.any()):
            return image_output, image_residual, metadata
        assert self.previous_cache is not None
        cached_residual = self.previous_cache[int(values["layer_index"])][1].to(
            image_residual.device
        )
        cached_mask = cached_mask_cpu.to(image_residual.device).view(1, -1, 1)
        mixed_residual = torch.where(cached_mask, cached_residual, image_residual)
        return image_input + mixed_residual, mixed_residual, metadata

    def execute_sparse(
        self,
        *,
        block: torch.nn.Module,
        original_forward: Callable[..., Any],
        positional_args: Tuple[Any, ...],
        keyword_args: Dict[str, Any],
        layer_index: int,
    ) -> Tuple[Any, Dict[str, Any]]:
        image_input, _ = read_block_inputs(positional_args, keyword_args)
        decision = self.controller._prepare_token_decision(
            step_index=self.step_index,
            branch_index=self.branch_index,
            item=self.item,
            previous_cache=self.previous_cache,
            layer_index=layer_index,
            image_input=image_input,
        )
        metadata = decision["metadata"]
        active_indices = decision["active_indices"]
        rl_transition = decision.get("rl_transition")
        encoder_hidden_states = _read_forward_argument(
            positional_args, keyword_args, "encoder_hidden_states", 1
        )
        encoder_hidden_states_mask = _read_forward_argument(
            positional_args, keyword_args, "encoder_hidden_states_mask", 2
        )
        temb = _read_forward_argument(positional_args, keyword_args, "temb", 3)
        image_rotary_emb = _read_forward_argument(
            positional_args, keyword_args, "image_rotary_emb", 4
        )
        joint_attention_kwargs = _read_forward_argument(
            positional_args, keyword_args, "joint_attention_kwargs", 5, {}
        )
        modulate_index = keyword_args.get("modulate_index")
        if not isinstance(encoder_hidden_states, torch.Tensor):
            raise RuntimeError("真实Token跳过无法读取text hidden。")
        if not isinstance(temb, torch.Tensor):
            raise RuntimeError("真实Token跳过无法读取temb。")
        kv_by_layer = self.controller.image_kv_caches.setdefault(
            self.branch_index, {}
        )
        cached_kv = kv_by_layer.get(layer_index)
        all_tokens_active = int(active_indices.numel()) == int(image_input.shape[1])
        runtime_cached_kv = None if all_tokens_active else cached_kv
        if self.previous_cache is None and not all_tokens_active:
            raise RuntimeError("真实Token跳过缺少上一step残差缓存。")
        cached_image_residual = (
            torch.zeros_like(image_input)
            if self.previous_cache is None
            else self.previous_cache[layer_index][1]
        )
        (
            text_output,
            image_output,
            updated_image_key,
            updated_image_value,
            kv_metadata,
        ) = sparse_qwen_block_forward(
            block=block,
            hidden_states=image_input,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            active_indices=active_indices,
            cached_image_residual=cached_image_residual,
            cached_image_key=(
                None if runtime_cached_kv is None else runtime_cached_kv[0]
            ),
            cached_image_value=(
                None if runtime_cached_kv is None else runtime_cached_kv[1]
            ),
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            modulate_index=modulate_index,
        )
        if (
            rl_transition is not None
            and self.controller.rl_policy is not None
            and bool(getattr(self.controller.args, "rl_record_teacher_labels", False))
        ):
            with torch.no_grad():
                full_output = original_forward(*positional_args, **keyword_args)
                _, full_image_output = split_block_output(full_output)
                difference = (image_output.detach().float() - full_image_output.detach().float())
                teacher_scores = difference.square().mean(dim=(0, 2)).sqrt()
                full_energy = (
                    full_image_output.detach().float().square().mean(dim=(0, 2)).sqrt()
                ).clamp_min(1e-6)
                teacher_scores = teacher_scores / full_energy
            self.controller.rl_policy.attach_teacher_labels(
                rl_transition, teacher_scores
            )
            metadata["teacher_label_recorded"] = True
            metadata["teacher_token_error_mean"] = float(teacher_scores.mean().item())
            metadata["teacher_token_error_max"] = float(teacher_scores.max().item())
        kv_by_layer[layer_index] = (updated_image_key, updated_image_value)
        metadata["token_level_real_compute_skipped"] = not all_tokens_active
        metadata["sparse_runtime_version"] = SPARSE_RUNTIME_VERSION
        metadata["image_kv_cache_action"] = (
            "refresh_from_full_compute"
            if all_tokens_active
            else (
                "initialize_then_sparse_update"
                if bool(kv_metadata["image_kv_cache_initialized"])
                else "sparse_update"
            )
        )
        metadata.update(kv_metadata)
        return (text_output, image_output), metadata


class BlueLineTokenScheduledController:
    """蓝线Block缓存 + 可切换Token仿真/真实稀疏 + 固定预算mask策略。"""

    def __init__(
        self,
        transformer_blocks: Sequence[torch.nn.Module],
        original_transformer_forward: Callable[..., Any],
        schedule: Dict[int, Dict[str, Any]],
        teacher_references: Dict[Tuple[int, int], Dict[str, Any]],
        args: argparse.Namespace,
        forwards_per_step: int,
    ) -> None:
        self.blocks = list(transformer_blocks)
        self.original_transformer_forward = original_transformer_forward
        self.original_block_forwards = [block.forward for block in self.blocks]
        self.schedule = schedule
        self.teacher_references = teacher_references
        self.args = args
        self.forwards_per_step = forwards_per_step
        self.total_layers = len(self.blocks)
        self.expected_calls = int(args.num_inference_steps) * forwards_per_step
        self.call_index = 0
        self.token_cache_ratio = float(
            getattr(args, "image_token_cache_ratio", 0.0)
        )
        self.max_token_cache_age = int(
            getattr(args, "max_token_cache_age", 3)
        )
        self.min_compute_ratio = float(
            getattr(args, "min_image_token_compute_ratio", 0.10)
        )
        self.token_budget_mode = str(getattr(args, "token_budget_mode", "per_block"))
        self.cache_edge_blocks = bool(
            getattr(args, "token_cache_edge_blocks", False)
        )
        self.token_execution_mode = str(
            getattr(args, "token_execution_mode", "simulation")
        )
        self.token_policy_mode = str(getattr(args, "token_policy_mode", "fixed"))
        if self.token_execution_mode not in {"simulation", "sparse"}:
            raise ValueError(
                f"未知token_execution_mode：{self.token_execution_mode}"
            )
        if self.token_policy_mode not in {"fixed", "rl"}:
            raise ValueError(f"未知token_policy_mode：{self.token_policy_mode}")
        if self.token_budget_mode not in {"per_block", "global"}:
            raise ValueError(f"未知token_budget_mode：{self.token_budget_mode}")
        self.rl_policy: Optional[TokenPolicyRuntime] = None
        if self.token_policy_mode == "rl":
            rl_device = str(getattr(args, "rl_policy_device", "auto"))
            if rl_device == "auto":
                rl_device = str(getattr(args, "device", "cpu"))
            self.rl_policy = TokenPolicyRuntime(
                policy_path=getattr(args, "rl_policy_path", None),
                device=rl_device,
                explore=bool(getattr(args, "rl_explore", False)),
                temperature=float(getattr(args, "rl_temperature", 1.0)),
                group_size=int(getattr(args, "rl_token_group_size", 16)),
                hidden_dim=int(getattr(args, "rl_hidden_dim", 64)),
                transition_block_stride=int(
                    getattr(args, "rl_transition_block_stride", 4)
                ),
            )
        self.previous_caches: Dict[int, BlockCache] = {}
        self.last_block_refresh_steps: Dict[int, Dict[int, int]] = {}
        self.token_source_inputs: Dict[int, Dict[int, torch.Tensor]] = {}
        self.token_last_refresh_steps: Dict[int, Dict[int, torch.Tensor]] = {}
        self.image_kv_caches: Dict[int, Dict[int, ImageKVCache]] = {}
        self.global_budget_states: Dict[int, GlobalTokenBudgetState] = {}
        self.branch_step_rows: List[Dict[str, Any]] = []
        self.block_action_rows: List[Dict[str, Any]] = []
        self.token_action_rows: List[Dict[str, Any]] = []

    @staticmethod
    def _score_values(
        current_input: torch.Tensor,
        source_input: torch.Tensor,
    ) -> torch.Tensor:
        if current_input.ndim != 3:
            raise RuntimeError(
                "Token缓存第一版要求image hidden形状为[batch,tokens,channels]，"
                f"实际为{tuple(current_input.shape)}。"
            )
        current = current_input.float()
        source = source_input.to(current_input.device).float()
        difference_energy = (current - source).square().mean(dim=(0, 2))
        source_energy = source.square().mean(dim=(0, 2)).clamp_min(1e-12)
        return torch.sqrt(difference_energy / source_energy)

    def _full_refresh_metadata(
        self,
        branch_index: int,
        layer_index: int,
        step_index: int,
        image_input: torch.Tensor,
        image_residual: torch.Tensor,
        reason: str,
    ) -> Dict[str, Any]:
        token_count = int(image_input.shape[1])
        is_forced_refresh = reason in {
            "block_cache_age_forced_full_refresh",
            "token_shape_changed_full_refresh",
            "token_count_changed_full_refresh",
            "previous_residual_shape_changed_full_refresh",
        }
        source_by_layer = self.token_source_inputs.setdefault(branch_index, {})
        refresh_by_layer = self.token_last_refresh_steps.setdefault(branch_index, {})
        source_by_layer[layer_index] = image_input.detach().clone()
        refresh_by_layer[layer_index] = torch.full(
            (token_count,), step_index, dtype=torch.long, device="cpu"
        )
        return {
            "token_action": "full_token_compute",
            "reason": reason,
            "image_token_count": token_count,
            "target_cached_image_token_count": 0,
            "cached_image_token_count": 0,
            "computed_image_token_count": token_count,
            "actual_image_token_cache_ratio": 0.0,
            "forced_refresh_image_token_count": (
                token_count if is_forced_refresh else 0
            ),
            "cached_token_ranges_1based": "",
            "forced_refresh_token_ranges_1based": (
                (f"1-{token_count}" if token_count > 1 else "1")
                if is_forced_refresh
                else ""
            ),
            "cached_token_age_min": 0,
            "cached_token_age_mean": 0.0,
            "cached_token_age_max": 0,
            "selection_score_all_mean": None,
            "selection_score_cached_mean": None,
            "selection_score_computed_mean": None,
            "token_level_real_compute_skipped": False,
            "image_kv_cache_action": "not_applicable_full_compute",
            "image_kv_cache_initialized": False,
            "image_kv_initialization_token_count": 0,
            "computed_image_kv_token_count": token_count,
            "cached_image_kv_token_count": 0,
        }

    @staticmethod
    def _schedule_risk(item: Dict[str, Any], layer_index: int) -> float:
        for key in (
            "smoothed_risk_by_block",
            "risk_by_block",
            "profile_risk_by_block",
        ):
            values = item.get(key)
            if isinstance(values, list) and layer_index < len(values):
                return float(values[layer_index])
            if isinstance(values, dict):
                value = values.get(str(layer_index), values.get(str(layer_index + 1)))
                if value is not None:
                    return float(value)
        return 0.0

    def _is_mixed_token_candidate(
        self,
        *,
        step_index: int,
        layer_index: int,
        item: Dict[str, Any],
    ) -> bool:
        if step_index == 0 or self.token_cache_ratio <= 0:
            return False
        if layer_index not in {int(value) for value in item["executed_blocks_0based"]}:
            return False
        if layer_index in {0, self.total_layers - 1} and not self.cache_edge_blocks:
            return False
        return True

    def _planned_mixed_cell_count(self) -> int:
        count = 0
        for step_index, item in self.schedule.items():
            for layer_index in item["executed_blocks_0based"]:
                if self._is_mixed_token_candidate(
                    step_index=int(step_index),
                    layer_index=int(layer_index),
                    item=item,
                ):
                    count += 1
        return max(1, count)

    def _global_budget_state(
        self,
        *,
        branch_index: int,
        token_count: int,
    ) -> GlobalTokenBudgetState:
        state = self.global_budget_states.get(branch_index)
        if state is None:
            state = GlobalTokenBudgetState(
                total_cells=self._planned_mixed_cell_count(),
                token_count=token_count,
                compute_ratio=max(0.0, 1.0 - self.token_cache_ratio),
                min_compute_ratio=self.min_compute_ratio,
            )
            self.global_budget_states[branch_index] = state
        elif int(state.token_count) != int(token_count):
            raise RuntimeError(
                "全局Token预算要求同一分支内token数量稳定："
                f"state={state.token_count}，current={token_count}。"
            )
        return state

    def _prepare_token_decision(
        self,
        *,
        step_index: int,
        branch_index: int,
        item: Dict[str, Any],
        previous_cache: Optional[BlockCache],
        layer_index: int,
        image_input: torch.Tensor,
    ) -> Dict[str, Any]:
        block_forced_refresh = layer_index in {
            int(value) for value in item.get("forced_refresh_blocks_0based", [])
        }
        if image_input.ndim != 3:
            raise RuntimeError(
                "Token缓存要求image hidden形状为[batch,tokens,channels]，"
                f"实际为{tuple(image_input.shape)}。"
            )
        token_count = int(image_input.shape[1])
        edge_block = layer_index in {0, self.total_layers - 1}
        full_reason: Optional[str] = None
        if step_index == 0:
            full_reason = "first_step_initializes_token_cache"
        elif self.token_cache_ratio <= 0 and self.token_policy_mode == "fixed":
            full_reason = "token_cache_disabled"
        elif edge_block and not self.cache_edge_blocks:
            full_reason = "edge_block_token_cache_disabled"
        elif previous_cache is None or layer_index not in previous_cache:
            full_reason = "missing_previous_residual_cache"

        source_by_layer = self.token_source_inputs.setdefault(branch_index, {})
        refresh_by_layer = self.token_last_refresh_steps.setdefault(branch_index, {})
        source_input = source_by_layer.get(layer_index)
        refresh_steps = refresh_by_layer.get(layer_index)
        if source_input is None or refresh_steps is None:
            full_reason = full_reason or "missing_token_source_cache"
        elif source_input.shape != image_input.shape:
            full_reason = "token_shape_changed_full_refresh"
        elif int(refresh_steps.numel()) != token_count:
            full_reason = "token_count_changed_full_refresh"
        if previous_cache is not None and layer_index in previous_cache:
            if previous_cache[layer_index][1].shape != image_input.shape:
                full_reason = "previous_residual_shape_changed_full_refresh"

        if full_reason is not None:
            metadata = self._full_refresh_metadata(
                branch_index,
                layer_index,
                step_index,
                image_input,
                image_input,
                full_reason,
            )
            metadata["requested_image_token_cache_ratio"] = 0.0
            metadata["token_execution_mode"] = self.token_execution_mode
            metadata["block_schedule_mode"] = item.get("mode")
            metadata["block_forced_refresh_with_token_budget"] = (
                block_forced_refresh
            )
            return {
                "cached_mask_cpu": torch.zeros(token_count, dtype=torch.bool),
                "active_indices": torch.arange(
                    token_count, device=image_input.device, dtype=torch.long
                ),
                "metadata": metadata,
            }

        assert source_input is not None and refresh_steps is not None
        scores = self._score_values(image_input, source_input)
        ages = step_index - refresh_steps
        expired_cpu = torch.zeros(token_count, dtype=torch.bool)
        if self.max_token_cache_age > 0:
            expired_cpu = ages > self.max_token_cache_age

        # per_block：每个mixed block固定同一比例；
        # global：整张图/分支只约束总mixed token计算比例，本cell可多可少。
        requested_ratio = self.token_cache_ratio
        minimum_compute = max(1, int(math.ceil(token_count * self.min_compute_ratio)))
        requested_compute_ratio = max(0.0, 1.0 - requested_ratio)
        rl_transition: Optional[Dict[str, Any]] = None
        actor_metadata: Dict[str, Any] = {
            "actor_observed_all_tokens": False,
            "actor_group_count": 0,
            "actor_group_size": 0,
            "actor_compute_count": 0,
            "mandatory_compute_count": 0,
            "actor_value_estimate": None,
        }
        previous_residual = (
            None
            if previous_cache is None or layer_index not in previous_cache
            else previous_cache[layer_index][1]
        )
        kv_cache = self.image_kv_caches.get(branch_index, {}).get(layer_index)
        token_features: Optional[torch.Tensor] = None
        if self.rl_policy is not None:
            token_features = build_token_features(
                current_input=image_input,
                source_input=source_input,
                change_scores=scores,
                ages=ages,
                cached_residual=previous_residual,
                cached_key=None if kv_cache is None else kv_cache[0],
                cached_value=None if kv_cache is None else kv_cache[1],
                step_index=step_index,
                num_steps=int(self.args.num_inference_steps),
                layer_index=layer_index,
                total_layers=self.total_layers,
                blue_line_risk=self._schedule_risk(item, layer_index),
                max_token_cache_age=self.max_token_cache_age,
            )

        global_budget_metadata: Dict[str, Any] = {}
        if self.token_budget_mode == "global":
            temporal_priority = (
                float(scores.detach().float().mean().item())
                + 0.10 * float(scores.detach().float().amax().item())
                + 0.25 * float(expired_cpu.float().mean().item())
                + 0.05 * math.log1p(max(0.0, self._schedule_risk(item, layer_index)))
            )
            priority = temporal_priority
            if self.rl_policy is not None and token_features is not None:
                budget_signal = self.rl_policy.estimate_budget_priority(
                    token_features=token_features,
                    fallback_scores=scores,
                )
                priority = (
                    0.50 * temporal_priority
                    + 0.35 * float(budget_signal["rl_budget_priority"])
                    + 0.15 * float(budget_signal["rl_budget_priority_top25"])
                )
                actor_metadata.update(budget_signal)
            compute_budget, global_budget_metadata = self._global_budget_state(
                branch_index=branch_index,
                token_count=token_count,
            ).allocate(priority=priority, min_compute=minimum_compute)
        else:
            compute_budget = max(
                minimum_compute,
                int(math.ceil(token_count * requested_compute_ratio)),
            )
            compute_budget = min(token_count, compute_budget)
            global_budget_metadata = {
                "global_token_budget_priority": None,
                "global_token_budget_total_compute_units": None,
                "global_token_budget_used_compute_units": None,
            }
        target_cached = token_count - compute_budget

        if self.rl_policy is not None:
            assert token_features is not None
            mandatory_target = min(
                compute_budget,
                int(
                    math.ceil(
                        token_count
                        * float(getattr(self.args, "rl_mandatory_compute_ratio", 0.05))
                    )
                ),
            )
            # 强制集合优先保护过期、高变化、高残差和中心语义区域，但不突破预算。
            mandatory_priority = (
                token_features[:, 2] * 1000.0
                + token_features[:, 0]
                + 0.50 * token_features[:, 6]
                + 0.25 * token_features[:, 11]
                + ages.to(token_features.device).float()
                / max(1, int(self.args.num_inference_steps))
            )
            mandatory_mask_device = torch.zeros(
                token_count, dtype=torch.bool, device=token_features.device
            )
            if mandatory_target > 0:
                mandatory_indices = torch.topk(
                    mandatory_priority,
                    k=mandatory_target,
                    largest=True,
                    sorted=False,
                ).indices
                mandatory_mask_device[mandatory_indices] = True
            computed_mask_device, rl_transition, actor_metadata = (
                self.rl_policy.select_tokens(
                    token_features=token_features,
                    compute_budget=compute_budget,
                    mandatory_mask=mandatory_mask_device,
                    fallback_scores=scores,
                    context={
                        "sample_index": int(getattr(self.args, "sample_index", 0)),
                        "step_index_0based": step_index,
                        "branch_index_0based": branch_index,
                        "block_index_0based": layer_index,
                    },
                )
            )
            computed_indices = torch.nonzero(
                computed_mask_device.to(image_input.device),
                as_tuple=False,
            ).flatten()
            computed_mask_cpu = computed_mask_device.detach().to(
                device="cpu", dtype=torch.bool
            )
        else:
            # 固定基线：过期Token优先，其余选择输入变化最大的Token。
            computed_mask_cpu = torch.zeros(token_count, dtype=torch.bool)
            expired_indices = torch.nonzero(expired_cpu, as_tuple=False).flatten()
            expired_refresh_count = min(compute_budget, int(expired_indices.numel()))
            if expired_refresh_count > 0:
                if expired_refresh_count == int(expired_indices.numel()):
                    selected_expired = expired_indices
                else:
                    selected_local = torch.topk(
                        ages.index_select(0, expired_indices),
                        k=expired_refresh_count,
                        largest=True,
                        sorted=False,
                    ).indices
                    selected_expired = expired_indices.index_select(0, selected_local)
                computed_mask_cpu[selected_expired] = True
            remaining_budget = compute_budget - expired_refresh_count
            if remaining_budget > 0:
                remaining_indices = torch.nonzero(
                    ~computed_mask_cpu, as_tuple=False
                ).flatten()
                remaining_device = remaining_indices.to(scores.device)
                selected_local = torch.topk(
                    scores.index_select(0, remaining_device),
                    k=remaining_budget,
                    largest=True,
                    sorted=False,
                ).indices
                selected_changed = remaining_device.index_select(
                    0, selected_local
                ).cpu()
                computed_mask_cpu[selected_changed] = True
            computed_indices = torch.nonzero(
                computed_mask_cpu, as_tuple=False
            ).flatten().to(image_input.device)

        cached_mask_cpu = ~computed_mask_cpu
        actual_cached = int(cached_mask_cpu.sum())
        if actual_cached != target_cached:
            raise RuntimeError(
                "Token严格预算失效："
                f"target_cached={target_cached}，actual_cached={actual_cached}。"
            )
        computed_mask = computed_mask_cpu.to(image_input.device).view(1, -1, 1)
        source_by_layer[layer_index] = torch.where(
            computed_mask,
            image_input.detach(),
            source_input.to(image_input.device),
        ).detach()
        updated_refresh_steps = refresh_steps.clone()
        updated_refresh_steps[computed_mask_cpu] = step_index
        refresh_by_layer[layer_index] = updated_refresh_steps

        cached_indices = torch.nonzero(cached_mask_cpu, as_tuple=False).flatten()
        expired_indices = torch.nonzero(expired_cpu, as_tuple=False).flatten()
        refreshed_expired_indices = torch.nonzero(
            expired_cpu & computed_mask_cpu, as_tuple=False
        ).flatten()
        deferred_expired_indices = torch.nonzero(
            expired_cpu & cached_mask_cpu, as_tuple=False
        ).flatten()
        cached_ages = ages[cached_mask_cpu]
        cached_scores = scores[cached_mask_cpu.to(scores.device)]
        computed_scores = scores[computed_mask_cpu.to(scores.device)]
        mean_cached_age = (
            float(cached_ages.float().mean()) if cached_ages.numel() else 0.0
        )
        actual_ratio = actual_cached / token_count
        if self.rl_policy is not None:
            self.rl_policy.set_actual_action(
                rl_transition, actual_ratio, mean_cached_age
            )
        metadata = {
            "token_action": "mixed_token_cache",
            "reason": (
                "adjustable_fixed_budget_all_token_actor_critic_topk"
                if self.rl_policy is not None
                else "fixed_ratio_hard_budget_expired_first_then_highest_input_change"
            ),
            "image_token_count": token_count,
            "requested_image_token_cache_ratio": requested_ratio,
            "requested_image_token_compute_ratio": requested_compute_ratio,
            "target_cached_image_token_count": target_cached,
            "cached_image_token_count": actual_cached,
            "computed_image_token_count": token_count - actual_cached,
            "actual_image_token_cache_ratio": actual_ratio,
            "expired_image_token_count": int(expired_cpu.sum()),
            "forced_refresh_image_token_count": int(refreshed_expired_indices.numel()),
            "deferred_expired_image_token_count": int(deferred_expired_indices.numel()),
            "cached_token_ranges_1based": compact_index_ranges(cached_indices.tolist()),
            "forced_refresh_token_ranges_1based": compact_index_ranges(
                refreshed_expired_indices.tolist()
            ),
            "deferred_expired_token_ranges_1based": compact_index_ranges(
                deferred_expired_indices.tolist()
            ),
            "cached_token_age_min": int(cached_ages.min()) if cached_ages.numel() else 0,
            "cached_token_age_mean": mean_cached_age,
            "cached_token_age_max": int(cached_ages.max()) if cached_ages.numel() else 0,
            "selection_score_all_mean": float(scores.mean().item()),
            "selection_score_cached_mean": (
                float(cached_scores.mean().item()) if cached_scores.numel() else None
            ),
            "selection_score_computed_mean": (
                float(computed_scores.mean().item()) if computed_scores.numel() else None
            ),
            "token_level_real_compute_skipped": False,
            "token_execution_mode": self.token_execution_mode,
            "strict_token_compute_budget": True,
            "block_schedule_mode": item.get("mode"),
            "block_forced_refresh_with_token_budget": block_forced_refresh,
            "token_budget_mode": self.token_budget_mode,
            "global_token_budget_enabled": self.token_budget_mode == "global",
            "image_kv_cache_action": "pending_sparse_execution",
            "image_kv_cache_initialized": False,
            "image_kv_initialization_token_count": 0,
            "computed_image_kv_token_count": token_count - actual_cached,
            "cached_image_kv_token_count": actual_cached,
            **actor_metadata,
            **global_budget_metadata,
        }
        return {
            "cached_mask_cpu": cached_mask_cpu,
            "active_indices": computed_indices,
            "rl_transition": rl_transition,
            "metadata": metadata,
        }

    def _make_token_policy(
        self,
        step_index: int,
        branch_index: int,
        item: Dict[str, Any],
        previous_cache: Optional[BlockCache],
    ) -> _TokenExecutionPolicy:
        policy = _TokenExecutionPolicy(
            controller=self,
            step_index=step_index,
            branch_index=branch_index,
            item=item,
            previous_cache=previous_cache,
        )
        if self.token_execution_mode == "simulation":
            policy.execute_sparse = None  # type: ignore[assignment]
        return policy

    def _whole_block_cache_token_metadata(
        self,
        step_index: int,
        branch_index: int,
        layer_index: int,
        previous_cache: BlockCache,
    ) -> Dict[str, Any]:
        token_count = int(previous_cache[layer_index][1].shape[1])
        refresh_steps = self.token_last_refresh_steps.get(branch_index, {}).get(
            layer_index
        )
        if refresh_steps is None:
            raise RuntimeError(
                f"Block {layer_index + 1}整体跳过，但Token缓存来源没有初始化。"
            )
        ages = step_index - refresh_steps
        return {
            "token_action": "whole_block_cache",
            "reason": "blue_line_skipped_entire_block",
            "image_token_count": token_count,
            "target_cached_image_token_count": token_count,
            "cached_image_token_count": token_count,
            "computed_image_token_count": 0,
            "actual_image_token_cache_ratio": 1.0,
            "forced_refresh_image_token_count": 0,
            "expired_image_token_count": 0,
            "deferred_expired_image_token_count": 0,
            "cached_token_ranges_1based": (
                f"1-{token_count}" if token_count > 1 else "1"
            ),
            "forced_refresh_token_ranges_1based": "",
            "deferred_expired_token_ranges_1based": "",
            "cached_token_age_min": int(ages.min()),
            "cached_token_age_mean": float(ages.float().mean()),
            "cached_token_age_max": int(ages.max()),
            "selection_score_all_mean": None,
            "selection_score_cached_mean": None,
            "selection_score_computed_mean": None,
            "token_level_real_compute_skipped": True,
            "strict_token_compute_budget": True,
            "image_kv_cache_action": "whole_block_cache_no_attention",
            "image_kv_cache_initialized": False,
            "image_kv_initialization_token_count": 0,
            "computed_image_kv_token_count": 0,
            "cached_image_kv_token_count": token_count,
        }

    def __call__(self, *positional_args, **keyword_args):
        if self.call_index >= self.expected_calls:
            raise RuntimeError("Block+Token策略Transformer forward次数超过预期。")
        step_index = self.call_index // self.forwards_per_step
        branch_index = self.call_index % self.forwards_per_step
        item = self.schedule[step_index]
        executed = {int(layer) for layer in item["executed_blocks_0based"]}
        if step_index == 0:
            executed = set(range(self.total_layers))
            previous_cache = None
        else:
            previous_cache = self.previous_caches.get(branch_index)
            if previous_cache is None:
                raise RuntimeError(
                    f"Block+Token step={step_index + 1}缺少上一step缓存。"
                )

        token_policy = self._make_token_policy(
            step_index,
            branch_index,
            item,
            previous_cache,
        )
        transformer_started = time.perf_counter()
        with install_block_policy(
            transformer_blocks=self.blocks,
            original_block_forwards=self.original_block_forwards,
            executed_layers=executed,
            previous_step_cache=previous_cache,
            capture_all_layers=True,
            image_token_policy=token_policy,
        ) as captured:
            output = self.original_transformer_forward(
                *positional_args,
                **keyword_args,
            )
        transformer_elapsed = time.perf_counter() - transformer_started
        last_tokens = captured["last_tokens"]["tokens"]
        current_cache: BlockCache = captured["layer_cache"]
        token_metadata: Dict[int, Dict[str, Any]] = captured["token_metadata"]
        if last_tokens is None or len(current_cache) != self.total_layers:
            raise RuntimeError("Block+Token策略没有采集到完整缓存或末层Token。")

        teacher = self.teacher_references.get((step_index, branch_index))
        if teacher is None:
            raise RuntimeError(
                f"缺少step={step_index + 1}、branch={branch_index + 1}教师输出。"
            )
        metrics = score_candidate(
            teacher_output=teacher["sample"],
            candidate_output=output,
            teacher_last_tokens=teacher["last_tokens"],
            candidate_last_tokens=last_tokens,
            args=self.args,
        )
        if self.rl_policy is not None:
            self.rl_policy.finish_step(
                step_index=step_index,
                branch_index=branch_index,
                teacher_score=float(metrics["score"]),
                elapsed_seconds=transformer_elapsed,
                quality_weight=float(getattr(self.args, "rl_quality_weight", 1.0)),
                age_penalty_weight=float(
                    getattr(self.args, "rl_age_penalty_weight", 0.01)
                ),
            )
        skipped = sorted(set(range(self.total_layers)) - executed)
        if previous_cache is not None:
            for layer_index in skipped:
                token_metadata[layer_index] = self._whole_block_cache_token_metadata(
                    step_index,
                    branch_index,
                    layer_index,
                    previous_cache,
                )
        if len(token_metadata) != self.total_layers:
            missing = sorted(set(range(self.total_layers)) - set(token_metadata))
            raise RuntimeError(
                "Block+Token策略缺少Token动作记录："
                f"{one_based_layer_string(missing)}"
            )

        total_image_tokens = sum(
            int(metadata["image_token_count"])
            for metadata in token_metadata.values()
        )
        computed_image_tokens = sum(
            int(metadata["computed_image_token_count"])
            for metadata in token_metadata.values()
        )
        cached_image_tokens = total_image_tokens - computed_image_tokens
        # print(
        #     f"[block+token][{getattr(self.args, 'device', 'cuda')}]"
        #     f"[sample {int(getattr(self.args, 'sample_index', 0)):05d}] "
        #     f"step={step_index + 1}/{self.args.num_inference_steps}，"
        #     f"branch={branch_index + 1}/{self.forwards_per_step}："
        #     f"执行Block={len(executed)}，跳过Block={len(skipped)}；"
        #     f"image token代理计算={computed_image_tokens}/{total_image_tokens}，"
        #     f"缓存={cached_image_tokens}",
        #     flush=True,
        # )
        self.branch_step_rows.append(
            {
                "step_index_0based": step_index,
                "step_number_1based": step_index + 1,
                "branch_index_0based": branch_index,
                "branch_number_1based": branch_index + 1,
                "executed_block_count": len(executed),
                "skipped_block_count": len(skipped),
                "executed_blocks_1based": one_based_layer_string(sorted(executed)),
                "skipped_blocks_1based": one_based_layer_string(skipped),
                "total_image_token_block_units": total_image_tokens,
                "computed_image_token_block_units_proxy": computed_image_tokens,
                "cached_image_token_block_units_proxy": cached_image_tokens,
                "image_token_compute_fraction_proxy": (
                    computed_image_tokens / total_image_tokens
                    if total_image_tokens
                    else 1.0
                ),
                "transformer_elapsed_seconds_host": transformer_elapsed,
                **metrics,
            }
        )

        block_refresh_steps = self.last_block_refresh_steps.setdefault(
            branch_index, {}
        )
        for layer_index in range(self.total_layers):
            metadata = token_metadata[layer_index]
            if layer_index in executed:
                source_step = step_index
                block_refresh_steps[layer_index] = step_index
                block_action = "execute"
                block_cache_age = 0
            else:
                source_step = block_refresh_steps.get(layer_index)
                if source_step is None:
                    raise RuntimeError(
                        f"Block {layer_index + 1}被跳过但没有真实缓存来源。"
                    )
                block_action = "cache"
                block_cache_age = step_index - source_step
            common = {
                "step_index_0based": step_index,
                "step_number_1based": step_index + 1,
                "branch_index_0based": branch_index,
                "branch_number_1based": branch_index + 1,
                "block_index_0based": layer_index,
                "block_number_1based": layer_index + 1,
            }
            self.block_action_rows.append(
                {
                    **common,
                    "action": block_action,
                    "cache_source_step_index_0based": source_step,
                    "cache_source_step_number_1based": source_step + 1,
                    "cache_age": block_cache_age,
                    "base_blue_line_action": (
                        "cache"
                        if layer_index
                        in set(item.get("base_skipped_blocks_0based", []))
                        else "execute"
                    ),
                    "forced_refresh": bool(
                        layer_index
                        in set(item.get("forced_refresh_blocks_0based", []))
                    ),
                }
            )
            self.token_action_rows.append(
                {
                    **common,
                    "block_action": block_action,
                    **metadata,
                }
            )

        self.previous_caches[branch_index] = current_cache
        self.call_index += 1
        return output

    def image_token_compute_summary(self) -> Dict[str, Any]:
        total = sum(
            int(row["image_token_count"]) for row in self.token_action_rows
        )
        computed = sum(
            int(row["computed_image_token_count"])
            for row in self.token_action_rows
        )
        cached = total - computed
        executed_rows = [
            row for row in self.token_action_rows if row["block_action"] == "execute"
        ]
        executed_total = sum(
            int(row["image_token_count"]) for row in executed_rows
        )
        executed_computed = sum(
            int(row["computed_image_token_count"]) for row in executed_rows
        )
        mixed_rows = [
            row for row in executed_rows if row["token_action"] == "mixed_token_cache"
        ]
        mixed_total = sum(int(row["image_token_count"]) for row in mixed_rows)
        mixed_computed = sum(
            int(row["computed_image_token_count"]) for row in mixed_rows
        )
        kv_computed = sum(
            int(row.get("computed_image_kv_token_count", 0))
            for row in self.token_action_rows
        )
        kv_initialization = sum(
            int(row.get("image_kv_initialization_token_count", 0))
            for row in self.token_action_rows
        )
        full_rows = [
            row for row in executed_rows if row["token_action"] == "full_token_compute"
        ]
        full_units = sum(int(row["image_token_count"]) for row in full_rows)
        initialization_full_units = sum(
            int(row["image_token_count"])
            for row in full_rows
            if row.get("reason") == "first_step_initializes_token_cache"
        )
        return {
            "total_image_token_block_units": total,
            "computed_image_token_block_units_proxy": computed,
            "cached_image_token_block_units_proxy": cached,
            "image_token_compute_fraction_proxy": computed / total if total else 1.0,
            "theoretical_image_token_block_speedup_proxy": (
                total / computed if computed else None
            ),
            "executed_block_image_token_units": executed_total,
            "executed_block_computed_image_token_units": executed_computed,
            "executed_block_image_token_compute_fraction": (
                executed_computed / executed_total if executed_total else 1.0
            ),
            "mixed_sparse_image_token_units": mixed_total,
            "mixed_sparse_computed_image_token_units": mixed_computed,
            "mixed_sparse_image_token_compute_fraction": (
                mixed_computed / mixed_total if mixed_total else None
            ),
            "computed_image_kv_token_units": kv_computed,
            "image_kv_initialization_token_units": kv_initialization,
            "image_kv_projection_token_units": kv_computed,
            "image_kv_cache_enabled": self.token_execution_mode == "sparse",
            "strict_token_compute_budget": True,
            "full_compute_image_token_units": full_units,
            "initialization_full_compute_image_token_units": initialization_full_units,
            "noninitialization_full_compute_image_token_units": (
                full_units - initialization_full_units
            ),
            "token_execution_mode": self.token_execution_mode,
            "token_policy_mode": self.token_policy_mode,
            "token_budget_mode": self.token_budget_mode,
            "global_token_budget_enabled": self.token_budget_mode == "global",
            "global_token_budget_total_units": sum(
                state.total_units for state in self.global_budget_states.values()
            ) or None,
            "global_token_budget_target_compute_units": sum(
                state.total_budget for state in self.global_budget_states.values()
            ) or None,
            "global_token_budget_used_compute_units": sum(
                state.used_budget for state in self.global_budget_states.values()
            ) or None,
            "global_token_budget_compute_fraction": (
                (
                    sum(state.used_budget for state in self.global_budget_states.values())
                    / sum(state.total_units for state in self.global_budget_states.values())
                )
                if self.global_budget_states
                and sum(state.total_units for state in self.global_budget_states.values())
                else None
            ),
            "token_timing_is_simulation_only": self.token_execution_mode == "simulation",
            "token_level_real_compute_skipped": self.token_execution_mode == "sparse",
            "sparse_runtime_version": (
                SPARSE_RUNTIME_VERSION if self.token_execution_mode == "sparse" else None
            ),
        }

    @property
    def rl_transitions(self) -> List[Dict[str, Any]]:
        return [] if self.rl_policy is None else self.rl_policy.transitions

    def finalize_rl_episode(self, final_metrics: Dict[str, Any]) -> None:
        if self.rl_policy is None:
            return
        self.rl_policy.finalize_episode(
            psnr=float(final_metrics["psnr"]),
            ssim=float(final_metrics["ssim"]),
            lpips=(
                None
                if final_metrics.get("lpips") is None
                else float(final_metrics["lpips"])
            ),
            psnr_weight=float(getattr(self.args, "rl_final_psnr_weight", 1.0)),
            ssim_weight=float(getattr(self.args, "rl_final_ssim_weight", 1.0)),
            lpips_weight=float(getattr(self.args, "rl_final_lpips_weight", 1.0)),
        )

    def validate_complete(self) -> None:
        if self.call_index != self.expected_calls:
            raise RuntimeError(
                f"Block+Token策略实际调用{self.call_index}次，"
                f"预期{self.expected_calls}次。"
            )


class SearchController:
    """
    在完整教师pipeline内部逐step穷举窗口，但返回教师输出维持完整基线轨迹。

    第一个timestep完整计算并按CFG分支建立全部Block层内残差缓存；后续候选
    的跳过层把上一教师timestep同分支、同编号Block的残差加到当前输入。
    """

    def __init__(
        self,
        transformer_blocks: Sequence[torch.nn.Module],
        original_transformer_forward: Callable[..., Any],
        candidates: Sequence[Window],
        args: argparse.Namespace,
        forwards_per_step: int,
    ) -> None:
        self.blocks = list(transformer_blocks)
        self.original_transformer_forward = original_transformer_forward
        self.original_block_forwards = [block.forward for block in self.blocks]
        self.candidates = list(candidates)
        self.args = args
        self.forwards_per_step = forwards_per_step
        self.total_layers = len(self.blocks)
        self.expected_calls = args.num_inference_steps * forwards_per_step
        self.call_index = 0
        self.branch_rows: List[Dict[str, Any]] = []
        self.aggregate_rows: List[Dict[str, Any]] = []
        self.schedule: Dict[int, Dict[str, Any]] = {}
        self.previous_teacher_caches: Dict[int, BlockCache] = {}
        self.progress_every = max(
            0,
            int(getattr(args, "progress_every", 25)),
        )
        self.search_started_at = time.perf_counter()
        self.total_candidate_evaluations = (
            max(0, args.num_inference_steps - 1)
            * forwards_per_step
            * len(self.candidates)
        )

    def _progress_prefix(self) -> str:
        parts = ["search"]
        device = getattr(self.args, "device", None)
        if device:
            parts.append(str(device))
        sample_index = getattr(self.args, "sample_index", None)
        if sample_index is not None:
            parts.append(f"sample {int(sample_index):05d}")
        return "".join(f"[{part}]" for part in parts)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _print_candidate_progress(
        self,
        step_index: int,
        branch_index: int,
        candidate_index: int,
    ) -> None:
        if self.progress_every <= 0:
            return
        candidate_count = len(self.candidates)
        if (
            candidate_index != 1
            and candidate_index != candidate_count
            and candidate_index % self.progress_every != 0
        ):
            return

        completed = (
            (
                (step_index - 1) * self.forwards_per_step
                + branch_index
            )
            * candidate_count
            + candidate_index
        )
        total = self.total_candidate_evaluations
        elapsed = time.perf_counter() - self.search_started_at
        percent = 100.0 * completed / total if total else 100.0
        if completed > 0 and total > completed:
            eta = elapsed * (total - completed) / completed
            eta_text = self._format_duration(eta)
        else:
            eta_text = "00:00"

        print(
            f"{self._progress_prefix()} "
            f"step={step_index + 1}/{self.args.num_inference_steps}，"
            f"branch={branch_index + 1}/{self.forwards_per_step}，"
            f"candidate={candidate_index}/{candidate_count}；"
            f"候选总体={completed}/{total} ({percent:.2f}%)；"
            f"已用={self._format_duration(elapsed)}，ETA={eta_text}",
            flush=True,
        )

    def _run_with_policy(
        self,
        positional_args: Tuple[Any, ...],
        keyword_args: Dict[str, Any],
        executed_layers: Set[int],
        previous_step_cache: Optional[BlockCache],
        capture_all_layers: bool,
    ) -> Tuple[Any, TensorPair, BlockCache]:
        with install_block_policy(
            transformer_blocks=self.blocks,
            original_block_forwards=self.original_block_forwards,
            executed_layers=executed_layers,
            previous_step_cache=previous_step_cache,
            capture_all_layers=capture_all_layers,
        ) as captured:
            output = self.original_transformer_forward(
                *positional_args,
                **keyword_args,
            )
        last_tokens = captured["last_tokens"]["tokens"]
        if last_tokens is None:
            raise RuntimeError("没有记录到最后一个Block的token。")
        layer_cache = captured["layer_cache"]
        if capture_all_layers and len(layer_cache) != self.total_layers:
            missing = [
                layer
                for layer in range(self.total_layers)
                if layer not in layer_cache
            ]
            raise RuntimeError(
                "没有记录完整的Block缓存，缺少："
                f"{one_based_layer_string(missing)}"
            )
        return output, last_tokens, layer_cache

    def _full_step_item(self) -> Dict[str, Any]:
        executed = list(range(self.total_layers))
        return {
            "step_index_0based": 0,
            "step_number_1based": 1,
            "window_start_0based": None,
            "window_end_0based": None,
            "window_start_1based": None,
            "window_end_1based": None,
            "window_size": self.total_layers,
            "executed_block_count": self.total_layers,
            "skipped_block_count": 0,
            "executed_blocks_1based": one_based_layer_string(executed),
            "skipped_blocks_1based": "",
            "noise_relative_mse": 0.0,
            "image_token_relative_mse": 0.0,
            "text_token_relative_mse": 0.0,
            "score": 0.0,
            "selected": True,
            "mode": "full_compute",
            "cache_source": "none_first_timestep",
        }

    def _finish_step(self, step_index: int) -> None:
        step_rows = [
            row
            for row in self.branch_rows
            if int(row["step_index_0based"]) == step_index
        ]
        if not step_rows:
            raise RuntimeError(f"step={step_index}没有候选结果。")

        rows_by_window: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for row in step_rows:
            key = (
                int(row["window_start_0based"]),
                int(row["window_end_0based"]),
            )
            rows_by_window.setdefault(key, []).append(row)

        aggregates: List[Dict[str, Any]] = []
        for start_layer, end_layer in self.candidates:
            branch_rows = rows_by_window.get((start_layer, end_layer), [])
            if len(branch_rows) != self.forwards_per_step:
                raise RuntimeError(
                    f"step={step_index}、窗口={start_layer}-{end_layer}只有"
                    f"{len(branch_rows)}个分支结果，预期{self.forwards_per_step}个。"
                )

            def mean(field: str) -> float:
                return float(
                    sum(float(row[field]) for row in branch_rows) / len(branch_rows)
                )

            window = (start_layer, end_layer)
            executed = sorted(executed_layers_for_window(self.total_layers, window))
            skipped = skipped_layers_for_window(self.total_layers, window)
            aggregate = {
                "step_index_0based": step_index,
                "step_number_1based": step_index + 1,
                "window_start_0based": start_layer,
                "window_end_0based": end_layer,
                "window_start_1based": start_layer + 1,
                "window_end_1based": end_layer + 1,
                "window_size": end_layer - start_layer + 1,
                "executed_block_count": len(executed),
                "skipped_block_count": len(skipped),
                "executed_blocks_1based": one_based_layer_string(executed),
                "skipped_blocks_1based": one_based_layer_string(skipped),
                "noise_relative_mse": mean("noise_relative_mse"),
                "image_token_relative_mse": mean(
                    "image_token_relative_mse"
                ),
                "text_token_relative_mse": mean(
                    "text_token_relative_mse"
                ),
                "score": mean("score"),
                "selected": False,
                "mode": "previous_timestep_same_block_residual_cache",
                "search_cache_source": (
                    "previous_teacher_timestep_same_block_residual"
                ),
                "cache_source": (
                    "previous_teacher_timestep_same_block_residual"
                ),
            }
            aggregates.append(aggregate)

        best = min(
            aggregates,
            key=lambda row: (
                float(row["score"]),
                -int(row["window_size"]),
                int(row["window_start_0based"]),
            ),
        )
        best["selected"] = True
        self.aggregate_rows.extend(aggregates)
        self.schedule[step_index] = {
            **best,
            "mode": "previous_timestep_same_block_residual_cache",
            "search_cache_source": (
                "previous_teacher_timestep_same_block_residual"
            ),
            "cache_source": (
                "previous_scheduled_timestep_same_block_residual"
            ),
        }
        print(
            f"{self._progress_prefix()} "
            f"step={step_index + 1}/{self.args.num_inference_steps}完成；"
            f"最低raw score窗口=Block "
            f"{best['window_start_1based']}-{best['window_end_1based']}；"
            f"窗口长度={best['window_size']}；"
            f"执行={best['executed_block_count']}层，"
            f"跳过={best['skipped_block_count']}层；"
            f"score={float(best['score']):.8e}",
            flush=True,
        )

    def __call__(self, *positional_args, **keyword_args):
        if self.call_index >= self.expected_calls:
            raise RuntimeError(
                "Transformer forward次数超过预期。"
                "请显式设置正确的--forwards-per-step。"
            )
        step_index = self.call_index // self.forwards_per_step
        branch_index = self.call_index % self.forwards_per_step
        print(
            f"{self._progress_prefix()} "
            f"开始step={step_index + 1}/{self.args.num_inference_steps}，"
            f"branch={branch_index + 1}/{self.forwards_per_step}；"
            + (
                "完整执行全部Block并建立缓存"
                if step_index == 0
                else f"准备测试{len(self.candidates)}个候选窗口"
            ),
            flush=True,
        )

        teacher_output, teacher_last_tokens, current_teacher_cache = (
            self._run_with_policy(
                positional_args,
                keyword_args,
                executed_layers=set(range(self.total_layers)),
                previous_step_cache=None,
                capture_all_layers=True,
            )
        )

        if step_index == 0:
            self.previous_teacher_caches[branch_index] = current_teacher_cache
            full_item = self._full_step_item()
            self.branch_rows.append(
                {
                    **full_item,
                    "branch_index_0based": branch_index,
                    "branch_number_1based": branch_index + 1,
                }
            )
            self.call_index += 1
            if branch_index == self.forwards_per_step - 1:
                self.schedule[0] = full_item
                print(
                    f"{self._progress_prefix()} "
                    "step=1完整执行结束，逐层缓存已经建立。",
                    flush=True,
                )
            return teacher_output

        previous_step_cache = self.previous_teacher_caches.get(branch_index)
        if previous_step_cache is None:
            raise RuntimeError(
                f"step={step_index + 1}、branch={branch_index + 1}"
                "没有找到上一timestep逐层缓存。"
            )

        for candidate_index, window in enumerate(self.candidates, start=1):
            start_layer, end_layer = window
            executed = executed_layers_for_window(self.total_layers, window)
            skipped = skipped_layers_for_window(self.total_layers, window)
            candidate_output, candidate_last_tokens, _ = self._run_with_policy(
                positional_args,
                keyword_args,
                executed_layers=executed,
                previous_step_cache=previous_step_cache,
                capture_all_layers=False,
            )
            metrics = score_candidate(
                teacher_output=teacher_output,
                candidate_output=candidate_output,
                teacher_last_tokens=teacher_last_tokens,
                candidate_last_tokens=candidate_last_tokens,
                args=self.args,
            )
            row = {
                "step_index_0based": step_index,
                "step_number_1based": step_index + 1,
                "branch_index_0based": branch_index,
                "branch_number_1based": branch_index + 1,
                "window_start_0based": start_layer,
                "window_end_0based": end_layer,
                "window_start_1based": start_layer + 1,
                "window_end_1based": end_layer + 1,
                "window_size": end_layer - start_layer + 1,
                "executed_block_count": len(executed),
                "skipped_block_count": len(skipped),
                "executed_blocks_1based": one_based_layer_string(sorted(executed)),
                "skipped_blocks_1based": one_based_layer_string(skipped),
                "mode": "previous_timestep_same_block_residual_cache",
                "search_cache_source": (
                    "previous_teacher_timestep_same_block_residual"
                ),
                "cache_source": (
                    "previous_teacher_timestep_same_block_residual"
                ),
                **metrics,
            }
            self.branch_rows.append(row)
            if self.args.verbose_candidates:
                print(
                    f"  candidate {candidate_index:03d}/{len(self.candidates):03d}: "
                    f"Block {start_layer + 1}-{end_layer + 1}，"
                    f"执行{len(executed)}层，缓存{len(skipped)}层，"
                    f"noise={metrics['noise_relative_mse']:.6e}，"
                    f"image_token={metrics['image_token_relative_mse']:.6e}，"
                    f"text_token={metrics['text_token_relative_mse']:.6e}，"
                    f"score={metrics['score']:.6e}",
                    flush=True,
                )
            else:
                self._print_candidate_progress(
                    step_index=step_index,
                    branch_index=branch_index,
                    candidate_index=candidate_index,
                )
            del candidate_output, candidate_last_tokens

        # 搜索阶段始终沿完整教师轨迹前进。下一timestep候选读取当前教师
        # timestep的逐层缓存，而不是读取任一候选的缓存。
        self.previous_teacher_caches[branch_index] = current_teacher_cache
        self.call_index += 1
        if branch_index == self.forwards_per_step - 1:
            self._finish_step(step_index)

        return teacher_output

    def validate_complete(self) -> None:
        if self.call_index != self.expected_calls:
            raise RuntimeError(
                f"pipeline实际调用Transformer {self.call_index}次，"
                f"预期{self.expected_calls}次。请检查--forwards-per-step。"
            )
        missing_steps = [
            step_index
            for step_index in range(self.args.num_inference_steps)
            if step_index not in self.schedule
        ]
        if missing_steps:
            raise RuntimeError(f"这些timestep没有选出执行策略：{missing_steps}")


class ScheduledController:
    """
    按每个timestep选出的最优窗口执行最终组合路径。

    与搜索不同，最终运行维护的是组合路径自身的逐层残差缓存：执行层用真实
    前向更新层内残差；跳过层把上一timestep同编号Block残差加到当前输入。
    """

    def __init__(
        self,
        transformer_blocks: Sequence[torch.nn.Module],
        original_transformer_forward: Callable[..., Any],
        schedule: Dict[int, Dict[str, Any]],
        args: argparse.Namespace,
        forwards_per_step: int,
    ) -> None:
        self.blocks = list(transformer_blocks)
        self.original_transformer_forward = original_transformer_forward
        self.original_block_forwards = [block.forward for block in self.blocks]
        self.schedule = schedule
        self.args = args
        self.forwards_per_step = forwards_per_step
        self.total_layers = len(self.blocks)
        self.expected_calls = args.num_inference_steps * forwards_per_step
        self.call_index = 0
        self.previous_step_caches: Dict[int, BlockCache] = {}

    def __call__(self, *positional_args, **keyword_args):
        if self.call_index >= self.expected_calls:
            raise RuntimeError("最终路径Transformer forward次数超过预期。")
        step_index = self.call_index // self.forwards_per_step
        branch_index = self.call_index % self.forwards_per_step
        schedule_item = self.schedule[step_index]
        mode = str(schedule_item.get("mode", ""))

        if step_index == 0 or mode == "full_compute":
            executed = set(range(self.total_layers))
            previous_step_cache = None
        else:
            window = (
                int(schedule_item["window_start_0based"]),
                int(schedule_item["window_end_0based"]),
            )
            executed = executed_layers_for_window(self.total_layers, window)
            previous_step_cache = self.previous_step_caches.get(branch_index)
            if previous_step_cache is None:
                raise RuntimeError(
                    f"最终路径step={step_index + 1}、branch={branch_index + 1}"
                    "没有上一timestep缓存。"
                )

        with install_block_policy(
            transformer_blocks=self.blocks,
            original_block_forwards=self.original_block_forwards,
            executed_layers=executed,
            previous_step_cache=previous_step_cache,
            capture_all_layers=True,
        ) as captured:
            output = self.original_transformer_forward(
                *positional_args,
                **keyword_args,
            )

        current_cache = captured["layer_cache"]
        if len(current_cache) != self.total_layers:
            missing = [
                layer
                for layer in range(self.total_layers)
                if layer not in current_cache
            ]
            raise RuntimeError(
                "最终路径没有记录完整缓存，缺少Block："
                f"{one_based_layer_string(missing)}"
            )
        self.previous_step_caches[branch_index] = current_cache
        self.call_index += 1
        return output

    def validate_complete(self) -> None:
        if self.call_index != self.expected_calls:
            raise RuntimeError(
                f"最终路径实际调用Transformer {self.call_index}次，"
                f"预期{self.expected_calls}次。"
            )


@contextmanager
def replace_transformer_forward(
    transformer: torch.nn.Module,
    controller: Callable[..., Any],
):
    original_forward = transformer.forward

    def controlled_forward(_transformer_self, *positional_args, **keyword_args):
        return controller(*positional_args, **keyword_args)

    transformer.forward = types.MethodType(controlled_forward, transformer)
    try:
        yield
    finally:
        transformer.forward = original_forward


def build_pipeline_inputs(
    input_images: Sequence[Image.Image],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    values: Dict[str, Any] = {
        "image": list(input_images) if len(input_images) > 1 else input_images[0],
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
        "generator": make_generator(args.seed),
        "max_sequence_length": 512,
    }
    if args.width is not None:
        values["width"] = args.width
    if args.height is not None:
        values["height"] = args.height
    return values


def generate_image(
    pipe: QwenImageEditPlusPipeline,
    input_images: Sequence[Image.Image],
    args: argparse.Namespace,
) -> Image.Image:
    with torch.inference_mode():
        output = pipe(**build_pipeline_inputs(input_images, args))
    return output.images[0].convert("RGB")


def load_pipeline(args: argparse.Namespace) -> QwenImageEditPlusPipeline:
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"正在加载模型：{args.model_path}", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if args.cpu_offload:
        device = torch.device(args.device)
        gpu_id = 0 if device.index is None else int(device.index)
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=not args.show_progress)
    return pipe


_LPIPS_MODEL: Optional[torch.nn.Module] = None
_LPIPS_UNAVAILABLE = False


def image_metrics(
    reference: Image.Image,
    candidate: Image.Image,
    compute_lpips: bool = False,
) -> Dict[str, Any]:
    reference_array = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    candidate_rgb = candidate.convert("RGB")
    if candidate_rgb.size != reference.size:
        candidate_rgb = candidate_rgb.resize(reference.size, Image.Resampling.LANCZOS)
    candidate_array = np.asarray(candidate_rgb, dtype=np.float32) / 255.0
    difference = candidate_array - reference_array
    absolute = np.abs(difference)
    mse = float(np.mean(np.square(difference)))
    mae = float(np.mean(absolute))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    changed_ratio = float(np.mean(np.max(absolute, axis=2) > (1.0 / 255.0)))
    reference_tensor = torch.from_numpy(reference_array).permute(2, 0, 1).unsqueeze(0)
    candidate_tensor = torch.from_numpy(candidate_array).permute(2, 0, 1).unsqueeze(0)
    kernel_size = 11
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(reference_tensor, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(candidate_tensor, kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(
        reference_tensor.square(), kernel_size, stride=1, padding=padding
    ) - mu_x.square()
    sigma_y = F.avg_pool2d(
        candidate_tensor.square(), kernel_size, stride=1, padding=padding
    ) - mu_y.square()
    sigma_xy = F.avg_pool2d(
        reference_tensor * candidate_tensor,
        kernel_size,
        stride=1,
        padding=padding,
    ) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    ssim = float(ssim_map.mean().item())

    lpips_value: Optional[float] = None
    global _LPIPS_MODEL, _LPIPS_UNAVAILABLE
    if compute_lpips and not _LPIPS_UNAVAILABLE:
        try:
            if _LPIPS_MODEL is None:
                import lpips

                _LPIPS_MODEL = lpips.LPIPS(net="alex").eval()
            with torch.inference_mode():
                lpips_value = float(
                    _LPIPS_MODEL(
                        reference_tensor * 2.0 - 1.0,
                        candidate_tensor * 2.0 - 1.0,
                    ).mean().item()
                )
        except Exception as error:
            _LPIPS_UNAVAILABLE = True
            print(
                "[WARN] LPIPS初始化或计算失败，本轮及后续LPIPS记录为null；"
                f"PSNR/SSIM奖励仍正常使用。原因：{type(error).__name__}: {error}",
                flush=True,
            )
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_value,
        "changed_ratio": changed_ratio,
    }


def save_json(value: Any, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(value, json_file, ensure_ascii=False, indent=2)


def write_candidate_scores(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    fieldnames = [
        "mode",
        "search_cache_source",
        "cache_source",
        "step_index_0based",
        "step_number_1based",
        "window_start_0based",
        "window_end_0based",
        "window_start_1based",
        "window_end_1based",
        "window_size",
        "executed_block_count",
        "skipped_block_count",
        "executed_blocks_1based",
        "skipped_blocks_1based",
        "noise_relative_mse",
        "image_token_relative_mse",
        "text_token_relative_mse",
        "score",
        "selected",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_candidate_layer_matrix(
    rows: Sequence[Dict[str, Any]],
    total_layers: int,
    output_path: Path,
) -> None:
    """
    每个step的每个候选一行；Block_001..Block_060中1=执行、0=跳过。
    """
    block_fields = [f"block_{layer + 1:03d}" for layer in range(total_layers)]
    fieldnames = [
        "mode",
        "search_cache_source",
        "cache_source",
        "step_number_1based",
        "window_start_1based",
        "window_end_1based",
        "score",
        "selected",
        *block_fields,
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            window = (
                int(row["window_start_0based"]),
                int(row["window_end_0based"]),
            )
            executed = executed_layers_for_window(total_layers, window)
            output_row: Dict[str, Any] = {
                "mode": row.get("mode"),
                "search_cache_source": row.get("cache_source"),
                "cache_source": row.get("cache_source"),
                "step_number_1based": row["step_number_1based"],
                "window_start_1based": row["window_start_1based"],
                "window_end_1based": row["window_end_1based"],
                "score": row["score"],
                "selected": row["selected"],
            }
            for layer in range(total_layers):
                output_row[f"block_{layer + 1:03d}"] = 1 if layer in executed else 0
            writer.writerow(output_row)


def write_best_schedule_matrix(
    schedule: Sequence[Dict[str, Any]],
    total_layers: int,
    output_path: Path,
) -> None:
    block_fields = [f"block_{layer + 1:03d}" for layer in range(total_layers)]
    fieldnames = [
        "mode",
        "search_cache_source",
        "cache_source",
        "step_number_1based",
        "window_start_1based",
        "window_end_1based",
        "noise_relative_mse",
        "image_token_relative_mse",
        "text_token_relative_mse",
        "score",
        *block_fields,
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in schedule:
            if item.get("mode") == "full_compute":
                executed = set(range(total_layers))
            else:
                window = (
                    int(item["window_start_0based"]),
                    int(item["window_end_0based"]),
                )
                executed = executed_layers_for_window(total_layers, window)
            row: Dict[str, Any] = {
                "mode": item.get("mode"),
                "search_cache_source": item.get("search_cache_source"),
                "cache_source": item.get("cache_source"),
                "step_number_1based": item["step_number_1based"],
                "window_start_1based": item["window_start_1based"],
                "window_end_1based": item["window_end_1based"],
                "noise_relative_mse": item["noise_relative_mse"],
                "image_token_relative_mse": item[
                    "image_token_relative_mse"
                ],
                "text_token_relative_mse": item[
                    "text_token_relative_mse"
                ],
                "score": item["score"],
            }
            for layer in range(total_layers):
                row[f"block_{layer + 1:03d}"] = 1 if layer in executed else 0
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            **vars(args),
            "strategy_version": CACHE_STRATEGY_VERSION,
            "first_timestep": "full_compute_all_blocks",
            "skipped_block_policy": (
                "current_input_plus_previous_timestep_same_block_residual"
            ),
        },
        output_dir / "run_config.json",
    )

    input_images = load_input_images(args.input_image)
    pipe = load_pipeline(args)
    transformer = pipe.transformer
    transformer_blocks = list(transformer.transformer_blocks)
    total_layers = len(transformer_blocks)
    candidates = build_candidate_windows(
        total_layers=total_layers,
        window_size=args.window_size,
        stride=args.window_stride,
    )
    forwards_per_step = infer_forwards_per_step(args)
    normal_executed_count = args.window_size + 2

    print(
        f"模型共{total_layers}个Block；中间候选窗口{len(candidates)}个；"
        "第一个timestep完整执行全部Block；"
        f"每个候选连续执行{args.window_size}个中间Block；"
        f"从第二个timestep起加上首尾后每步共执行{normal_executed_count}层、"
        f"跳过{total_layers - normal_executed_count}层；"
        "跳过层使用当前输入加上一timestep同编号Block层内残差；"
        f"forwards_per_step={forwards_per_step}。",
        flush=True,
    )

    original_transformer_forward = transformer.forward
    search_controller = SearchController(
        transformer_blocks=transformer_blocks,
        original_transformer_forward=original_transformer_forward,
        candidates=candidates,
        args=args,
        forwards_per_step=forwards_per_step,
    )

    print(
        "开始完整教师轨迹：step 1建立全层缓存，step 2起搜索候选……",
        flush=True,
    )
    torch.cuda.synchronize(torch.device(args.device))
    search_start = time.perf_counter()
    with replace_transformer_forward(transformer, search_controller):
        baseline_image = generate_image(pipe, input_images, args)
    torch.cuda.synchronize(torch.device(args.device))
    search_elapsed = time.perf_counter() - search_start
    search_controller.validate_complete()

    baseline_path = output_dir / "baseline_full.png"
    baseline_image.save(baseline_path)

    candidate_scores_path = output_dir / "candidate_scores.csv"
    candidate_matrix_path = output_dir / "candidate_layer_matrix.csv"
    branch_details_path = output_dir / "candidate_branch_details.json"
    write_candidate_scores(
        search_controller.aggregate_rows,
        candidate_scores_path,
    )
    write_candidate_layer_matrix(
        search_controller.aggregate_rows,
        total_layers,
        candidate_matrix_path,
    )
    save_json(search_controller.branch_rows, branch_details_path)

    ordered_schedule = [
        search_controller.schedule[step_index]
        for step_index in range(args.num_inference_steps)
    ]
    best_schedule_path = output_dir / "best_schedule.json"
    best_matrix_path = output_dir / "best_schedule_layer_matrix.csv"
    schedule_payload = {
        "method": (
            "第一个timestep完整执行全部Block；后续Block 1和最后一个Block"
            "始终执行，并执行固定长度的连续中间窗口；其他Block把上一"
            "timestep同编号Block的text/image层内残差加到当前输入。"
        ),
        "strategy_version": CACHE_STRATEGY_VERSION,
        "first_timestep": "full_compute_all_blocks",
        "skipped_block_policy": (
            "current_input_plus_previous_timestep_same_block_residual"
        ),
        "score_formula": (
            f"{args.noise_weight} * noise_relative_mse + "
            f"{args.image_token_weight} * image_token_relative_mse + "
            f"{args.text_token_weight} * text_token_relative_mse"
        ),
        "total_layers": total_layers,
        "num_inference_steps": args.num_inference_steps,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "candidate_count_first_timestep": 0,
        "candidate_count_later_timestep": len(candidates),
        "executed_block_count_first_timestep": total_layers,
        "executed_block_count_later_timestep": normal_executed_count,
        "skipped_block_count_first_timestep": 0,
        "skipped_block_count_later_timestep": (
            total_layers - normal_executed_count
        ),
        "forwards_per_step": forwards_per_step,
        "search_elapsed_seconds": search_elapsed,
        "steps": ordered_schedule,
    }
    save_json(schedule_payload, best_schedule_path)
    write_best_schedule_matrix(
        ordered_schedule,
        total_layers,
        best_matrix_path,
    )

    final_path: Optional[Path] = None
    final_metrics: Optional[Dict[str, float]] = None
    final_elapsed: Optional[float] = None
    if not args.skip_final_run:
        print("开始按逐step最优窗口执行最终组合路径……", flush=True)
        scheduled_controller = ScheduledController(
            transformer_blocks=transformer_blocks,
            original_transformer_forward=original_transformer_forward,
            schedule=search_controller.schedule,
            args=args,
            forwards_per_step=forwards_per_step,
        )
        torch.cuda.synchronize(torch.device(args.device))
        final_start = time.perf_counter()
        with replace_transformer_forward(transformer, scheduled_controller):
            final_image = generate_image(pipe, input_images, args)
        torch.cuda.synchronize(torch.device(args.device))
        final_elapsed = time.perf_counter() - final_start
        scheduled_controller.validate_complete()
        final_path = output_dir / "diagonal_bridge_best.png"
        final_image.save(final_path)
        final_metrics = image_metrics(baseline_image, final_image)
        save_json(
            {
                "baseline_image": str(baseline_path.resolve()),
                "candidate_image": str(final_path.resolve()),
                "elapsed_seconds": final_elapsed,
                **final_metrics,
            },
            output_dir / "final_image_metrics.json",
        )

    summary = {
        "baseline_image": str(baseline_path.resolve()),
        "candidate_scores": str(candidate_scores_path.resolve()),
        "candidate_layer_matrix": str(candidate_matrix_path.resolve()),
        "candidate_branch_details": str(branch_details_path.resolve()),
        "best_schedule": str(best_schedule_path.resolve()),
        "best_schedule_layer_matrix": str(best_matrix_path.resolve()),
        "search_elapsed_seconds": search_elapsed,
        "final_image": None if final_path is None else str(final_path.resolve()),
        "final_elapsed_seconds": final_elapsed,
        "final_image_metrics": final_metrics,
    }
    save_json(summary, output_dir / "summary.json")

    print("逐timestep最优结果：", flush=True)
    for item in ordered_schedule:
        if item.get("mode") == "full_compute":
            print(
                f"  step {item['step_number_1based']}: 完整执行全部"
                f"{item['executed_block_count']}个Block并建立缓存。",
                flush=True,
            )
            continue
        print(
            f"  step {item['step_number_1based']}: "
            f"窗口Block {item['window_start_1based']}-"
            f"{item['window_end_1based']}；"
            f"执行={item['executed_blocks_1based']}；"
            f"跳过={item['skipped_blocks_1based']}；"
            f"noise={item['noise_relative_mse']:.6e}；"
            f"image_token={item['image_token_relative_mse']:.6e}；"
            f"text_token={item['text_token_relative_mse']:.6e}；"
            f"score={item['score']:.6e}",
            flush=True,
        )
    if final_metrics is not None:
        print(
            f"最终组合图相对baseline：MAE={final_metrics['mae']:.8f}，"
            f"MSE={final_metrics['mse']:.8f}，"
            f"PSNR={final_metrics['psnr']:.4f} dB。",
            flush=True,
        )
    print(f"全部完成，结果目录：{output_dir.resolve()}", flush=True)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
