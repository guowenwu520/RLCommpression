#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit-2511：Fresh Blueprint + Low-Dimensional Continuous TD3 v7.5 六组对比版
===============================================================

这版只复用三类旧逻辑：
1) Qwen-Image-Edit 模型加载 / pipeline 调用；
2) 数据集图片扫描 + portrait_prompts.md 解析；
3) 已经验证过的 Blue-Line / Blueprint Block residual cache 运行时。

注意：本版不读取任何旧 Blueprint schedule。会先用当前模型、当前 step 配置和独立 calibration 样本
重新统计 step×block 残差变化，然后在本次输出目录内生成全新的 Blueprint schedule。

旧的 Token-RL、teacher collect、预训练 transition 文件、两轮 rollout 等流程全部取消。
新的 RL 使用低维连续 TD3：每个 timestep 只输出固定维 latent action，再根据当前实际 eligible Block 数量
动态插值并投影成该 timestep 的 Block token 预算。Blueprint 每个 step 可保留不同数量的 Block。Block 内具体计算哪些 token，仍然沿用稳定的规则：
过期 token 优先，其余选择 hidden 变化最大的 token。

最终一次评估固定输出六组：
    1. full_dense          : 完整模型，什么都不做
    2. blueprint_only      : 只使用 Blueprint Block cache
    3. blueprint_fixed25   : Blueprint + 每个可稀疏 Block 固定 25%
    4. blueprint_rl25      : Blueprint + RL 在 Block 间分配 25% 总预算
    5. full_fixed25        : 完整 Block 路径 + 每个可稀疏 Block 固定 25%
    6. full_rl25           : 完整 Block 路径 + RL 在 Block 间分配 25% 总预算

训练时 full 和 blueprint 分别训练一套 policy，避免两个底座的 hidden/cache 分布混在一起。
reward 只比较“同底座 fixed25”与“同底座 RL25”的逐 step teacher error：

    reward_t = clip((error_fixed25 - error_rl25) / (error_fixed25 + eps), -2, 2)

预算已经由代码硬锁死，所以 reward 中不再放 FLOPs/speed 惩罚。
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from qwen_blueprint_runtime import (
    BlueLineScheduledController,
    BlueLineTokenScheduledController,
    FullReferenceController,
    ResidualProfileController,
    generate_image,
    image_metrics,
    infer_forwards_per_step,
    load_pipeline,
    replace_transformer_forward,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_PROMPT_GROUPS = {
    "FFHQ 主训练",
    "FFHQ 额外评估",
    "早期 10 prompt 评估",
    "生活场景评估",
    "姿态背景评估",
}

# RL v7.5：低维连续 TD3。
# 一个 timestep 只有一个低维 latent action；latent action 会根据该 timestep 当前实际
# eligible Block 数量动态插值为任意长度（29/30/58...）的预算曲线，再严格投影到与 Fixed25
# 完全相同的整数总 token 预算。Actor 最后一层零初始化，因此 action==0 时严格回到 Fixed25。
RL_ALGORITHM_VERSION = "lowdim_td3_budget_v1"
TRAIN_STATIC_CACHE_VERSION = "full_reference_fixed25_cache_v1"  # 保持不变，继续复用 v7.4 静态缓存
TD3_STATE_GLOBAL_DIM = 6
TD3_BLOCK_FEATURE_DIM = 6


def td3_state_dim(latent_dim: int) -> int:
    return TD3_STATE_GLOBAL_DIM + TD3_BLOCK_FEATURE_DIM * int(latent_dim)


# -----------------------------------------------------------------------------
# 一、命令行参数
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blueprint + Block-Budget RL 六组统一训练/评估。"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "blueprint", "train", "eval"],
        default="all",
        help="all=Fresh Blueprint→训练两套policy→六组评估；blueprint=只重建蓝图；train=Fresh Blueprint→只训练；eval=只读取当前output-dir内本次生成的蓝图和policy评估。",
    )
    parser.add_argument(
        "--model-path",
        default="/data4/guowenwu/MMDITModelCompression/models/Qwen-Image-Edit-2511",
    )
    parser.add_argument(
        "--dataset-root",
        default="/data4/guowenwu/MMDITModelCompression/dataset/images1024x1024",
    )
    parser.add_argument(
        "--prompt-file",
        default="/data4/guowenwu/MMDITModelCompression/portrait_prompts.md",
    )
    parser.add_argument(
        "--output-dir",
        default="/data4/guowenwu/MMDITModelCompression/outputs/rl_block_budget_sixway_fresh_v2",
    )

    # Fresh Blueprint 校准样本与 RL/评估样本完全分开。
    # 默认 5000 是之前完整蓝图实验使用的规模；冒烟时可改成 4/8/20。
    parser.add_argument("--blueprint-calibration-count", type=int, default=5000)
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--eval-count", type=int, default=20)
    parser.add_argument("--profile-quantile", type=float, default=0.90)
    parser.add_argument("--target-cache-ratio", type=float, default=0.70)
    parser.add_argument("--profile-smoothing-radius", type=int, default=1)
    parser.add_argument(
        "--profile-aggregate-block-chunk",
        type=int,
        default=8,
        help=(
            "Fresh Blueprint 汇总 P90 时一次只处理多少个 Block。"
            "旧版会把全部 calibration 样本一次性堆进内存；本版先顺序写磁盘 memmap，"
            "再按 Block 分块做精确 nanquantile。默认8，数值越小峰值内存越低，但汇总稍慢。"
        ),
    )
    parser.add_argument("--blueprint-max-cache-age", type=int, default=5)
    parser.add_argument("--force-full-first-steps", type=int, default=1)
    parser.add_argument("--force-full-last-steps", type=int, default=0)
    parser.add_argument(
        "--rebuild-blueprint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "默认续跑/复用当前输出目录中的 Blueprint。传 --rebuild-blueprint 时，"
            "只允许复用本次 output-dir 内已经生成的 schedule，用于中断续跑；"
            "不会读取任何外部/历史 Blueprint。"
        ),
    )
    parser.add_argument("--sampling-seed", type=int, default=20260814)
    parser.add_argument("--generation-seed", type=int, default=900000)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--forwards-per-step", type=int, choices=[1, 2], default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--prompt-language", choices=["english", "chinese"], default="english")
    parser.add_argument("--include-viton-prompts", action="store_true")

    # Token sparse 运行参数。为了与之前的真实加速实验保持一致，默认 sparse。
    parser.add_argument("--compute-ratio", type=float, default=0.25)
    parser.add_argument("--min-compute-ratio", type=float, default=0.05)
    parser.add_argument("--max-token-cache-age", type=int, default=5)
    parser.add_argument(
        "--token-execution-mode",
        choices=["simulation", "sparse"],
        default="sparse",
    )
    parser.add_argument(
        "--token-cache-edge-blocks",
        action="store_true",
        help="默认首尾 Block 的 image token 全算；开启后也允许首尾 Block 做 token sparse。",
    )

    # 低维连续 TD3。真正昂贵的是 Qwen rollout，因此使用 replay buffer 反复利用历史 transition。
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="TD3 Actor learning rate。")
    parser.add_argument("--td3-critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--td3-latent-dim", type=int, default=6)
    parser.add_argument("--td3-replay-capacity", type=int, default=50000)
    parser.add_argument("--td3-batch-size", type=int, default=256)
    parser.add_argument("--td3-gradient-steps", type=int, default=8)
    parser.add_argument("--td3-warmup-transitions", type=int, default=512)
    parser.add_argument("--td3-tau", type=float, default=0.005)
    parser.add_argument("--td3-policy-delay", type=int, default=2)
    parser.add_argument("--td3-target-noise", type=float, default=0.10)
    parser.add_argument("--td3-target-noise-clip", type=float, default=0.25)
    parser.add_argument("--td3-exploration-noise", type=float, default=0.15)
    parser.add_argument("--td3-exploration-noise-min", type=float, default=0.02)
    parser.add_argument("--td3-exploration-noise-decay", type=float, default=0.98)
    parser.add_argument("--td3-allocation-logit-scale", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--update-every-samples", type=int, default=4)
    # 兼容旧 shell 参数：接受但 TD3 不使用。
    parser.add_argument("--ppo-epochs", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument(
        "--cache-train-static",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "默认开启：第一次见到训练样本时把逐 timestep Full Reference tensor 和同底座 "
            "Fixed25 score 落盘。后续 epoch 只重新跑 RL rollout，不重复完整计算/Fixed25。"
        ),
    )

    # 自动收敛判断。train-count 只定义训练池大小，不再定义训练终点。
    # 训练池会循环 rollout，直到 reward 进入平台期且 deterministic Actor 的 policy_delta 足够小。
    parser.add_argument("--convergence-min-epochs", type=int, default=2)
    parser.add_argument("--convergence-window-updates", type=int, default=8)
    parser.add_argument("--convergence-patience", type=int, default=3)
    parser.add_argument("--convergence-reward-abs-tol", type=float, default=0.003)
    parser.add_argument("--convergence-reward-rel-tol", type=float, default=0.02)
    parser.add_argument(
        "--convergence-policy-delta-threshold", "--convergence-kl-threshold",
        dest="convergence_policy_delta_threshold", type=float, default=0.0015,
        help="TD3 deterministic Actor 在收敛窗口内的平均动作变化阈值；旧 --convergence-kl-threshold 作为兼容别名。",
    )
    parser.add_argument(
        "--convergence-max-epochs",
        type=int,
        default=0,
        help="0=不设人工训练轮数上限；>0 时只作为防止无限训练的安全上限，不是正常停止条件。",
    )

    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--distributed-timeout-seconds", type=int, default=7200)

    # teacher error 使用旧 runtime 的三项组合，保持与之前实验可比。
    parser.add_argument("--noise-weight", type=float, default=1.0)
    parser.add_argument("--image-token-weight", type=float, default=1.0)
    parser.add_argument("--text-token-weight", type=float, default=0.25)
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 零、分布式/续跑辅助函数
# -----------------------------------------------------------------------------
def dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if dist_is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist_is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def dist_barrier() -> None:
    """控制面同步只走 CPU/Gloo，不让 NCCL 参与。

    本项目不是 DDP：每个 rank 独立加载/运行一份 Qwen，跨 rank 只需要 barrier
    和同步很小的 RL policy。因此 GPU collective 没有收益，反而会把实验绑定到
    NCCL/P2P/拓扑状态。默认 process group 统一使用 Gloo 后，这里只做普通 CPU barrier。
    """
    if dist_is_initialized():
        dist.barrier()


def _broadcast_tensor_via_cpu(tensor: torch.Tensor, src: int = 0) -> None:
    """用 Gloo/CPU 同步一个 tensor，再拷回原 device。

    policy 很小，这个拷贝代价可以忽略；换来的好处是多卡控制通信完全不依赖 NCCL。
    """
    if not dist_is_initialized():
        return
    rank = get_rank()
    if rank == src:
        cpu_tensor = tensor.detach().cpu().contiguous()
    else:
        cpu_tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
    dist.broadcast(cpu_tensor, src=src)
    tensor.copy_(cpu_tensor.to(device=tensor.device, dtype=tensor.dtype))


def broadcast_model_from_rank0(model: nn.Module) -> None:
    """多卡 rollout 共享同一 policy；参数通过 CPU/Gloo 广播。"""
    if not dist_is_initialized():
        return
    with torch.no_grad():
        for parameter in model.parameters():
            _broadcast_tensor_via_cpu(parameter.data, src=0)
        for buffer in model.buffers():
            _broadcast_tensor_via_cpu(buffer.data, src=0)


def resolve_device(spec: str, local_rank: int) -> str:
    if spec in {"auto", "cuda", "cuda:auto"}:
        return f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if get_world_size() > 1 and spec.startswith("cuda"):
        return f"cuda:{local_rank}"
    return spec


def setup_distributed(args: argparse.Namespace) -> argparse.Namespace:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # 每个进程仍然独占自己的 CUDA GPU；但进程间“控制面”统一使用 Gloo/CPU。
    # Qwen 本身没有 DDP/all-reduce，因此这里没有必要创建 NCCL communicator。
    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        if local_rank < 0 or local_rank >= visible:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} 超出当前进程可见 GPU 数量 {visible}；"
                f"请检查 CUDA_VISIBLE_DEVICES 和 NPROC_PER_NODE。"
            )
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist_is_initialized():
        dist.init_process_group(
            backend="gloo",
            timeout=timedelta(seconds=int(args.distributed_timeout_seconds)),
        )

    args.rank = get_rank()
    args.world_size = get_world_size()
    args.local_rank = local_rank
    args.device = resolve_device(args.device, local_rank)
    args.policy_device = resolve_device(args.policy_device, local_rank)
    if is_main_process() and world_size > 1:
        print(
            f"[distributed] world_size={world_size}; control_backend=gloo(cpu); "
            f"Qwen各rank独立运行，不创建NCCL process group。",
            flush=True,
        )
    return args


def cleanup_distributed() -> None:
    if dist_is_initialized():
        dist_barrier()
        dist.destroy_process_group()


def shard_by_index(rows: Sequence[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    rank = get_rank()
    world = get_world_size()
    return [(i, row) for i, row in enumerate(rows) if i % world == rank]


def serialize_transition(t: "Transition") -> Dict[str, Any]:
    return {
        "state": t.state.cpu(),
        "action": t.action.cpu(),
        "reward": float(t.reward),
        "next_state": t.next_state.cpu(),
        "done": bool(t.done),
        "step_index": int(t.step_index),
        "branch_index": int(t.branch_index),
        "eligible_layers": [int(x) for x in t.eligible_layers],
        "allocated_compute_ratios": [float(x) for x in t.allocated_compute_ratios],
        "allocated_compute_counts": [int(x) for x in t.allocated_compute_counts],
    }


def deserialize_transition(obj: Dict[str, Any]) -> "Transition":
    return Transition(
        state=obj["state"].float().cpu(),
        action=obj["action"].float().cpu(),
        reward=float(obj.get("reward", 0.0)),
        next_state=obj.get("next_state", torch.zeros_like(obj["state"])).float().cpu(),
        done=bool(obj.get("done", False)),
        step_index=int(obj.get("step_index", -1)),
        branch_index=int(obj.get("branch_index", 0)),
        eligible_layers=[int(x) for x in obj.get("eligible_layers", [])],
        allocated_compute_ratios=[float(x) for x in obj.get("allocated_compute_ratios", [])],
        allocated_compute_counts=[int(x) for x in obj.get("allocated_compute_counts", [])],
    )



# -----------------------------------------------------------------------------
# 强制断点续跑 / 原子落盘辅助函数
# -----------------------------------------------------------------------------
def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """先写同目录临时文件再 os.replace；只有完整写完的结果才会被后续识别。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.rank{get_rank()}.{os.getpid()}")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.rank{get_rank()}.{os.getpid()}")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.rank{get_rank()}.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_image_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pillow 依赖后缀推断格式，所以临时文件仍保留原扩展名。
    tmp = path.with_name(path.stem + f".tmp.rank{get_rank()}.{os.getpid()}" + path.suffix)
    image.save(tmp)
    os.replace(tmp, path)


def touch_done(path: Path) -> None:
    atomic_write_text(path, "done\n")


def done_exists(path: Path) -> bool:
    return path.is_file()


def profile_fieldnames(total_layers: int) -> List[str]:
    return (
        ["branch_index", "step_index"]
        + [f"image_b{b:03d}" for b in range(total_layers)]
        + [f"text_b{b:03d}" for b in range(total_layers)]
    )


def write_profile_table(
    path: Path,
    image_grid: np.ndarray,
    text_grid: np.ndarray,
) -> None:
    branches, steps, layers = image_grid.shape
    rows: List[Dict[str, Any]] = []
    for branch in range(branches):
        for step in range(steps):
            row: Dict[str, Any] = {"branch_index": branch, "step_index": step}
            for block in range(layers):
                row[f"image_b{block:03d}"] = float(image_grid[branch, step, block])
                row[f"text_b{block:03d}"] = float(text_grid[branch, step, block])
            rows.append(row)
    atomic_write_csv(path, profile_fieldnames(layers), rows)


def read_profile_chunk(
    path: Path,
    *,
    branch_count: int,
    num_steps: int,
    block_start: int,
    block_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """只从一个样本表读取当前 Block 小块；返回 CPU float32。"""
    width = block_end - block_start
    image = np.full((branch_count, num_steps, width), np.nan, dtype=np.float32)
    text = np.full_like(image, np.nan)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            branch = int(row["branch_index"])
            step = int(row["step_index"])
            if branch >= branch_count or step >= num_steps:
                continue
            for local, block in enumerate(range(block_start, block_end)):
                try:
                    image[branch, step, local] = float(row[f"image_b{block:03d}"])
                    text[branch, step, local] = float(row[f"text_b{block:03d}"])
                except (KeyError, ValueError, TypeError):
                    pass
    return image, text


def write_transition_table(path: Path, transitions: Sequence["Transition"]) -> None:
    latent_dim = max((int(t.action.numel()) for t in transitions), default=0)
    state_dim = max((int(t.state.numel()) for t in transitions), default=0)
    fields = [
        "decision_index", "step_index", "branch_index", "eligible_block_count",
        "reward", "done", "alloc_mean", "alloc_std", "alloc_min", "alloc_max",
        "eligible_layers_json", "allocated_ratios_json", "allocated_counts_json",
    ] + [f"action_{i}" for i in range(latent_dim)] + [f"state_{i}" for i in range(state_dim)]
    rows: List[Dict[str, Any]] = []
    for i, t in enumerate(transitions):
        alloc = np.asarray(t.allocated_compute_ratios, dtype=np.float64)
        row: Dict[str, Any] = {
            "decision_index": i,
            "step_index": int(t.step_index),
            "branch_index": int(t.branch_index),
            "eligible_block_count": len(t.eligible_layers),
            "reward": float(t.reward),
            "done": int(bool(t.done)),
            "alloc_mean": float(alloc.mean()) if alloc.size else 0.0,
            "alloc_std": float(alloc.std()) if alloc.size else 0.0,
            "alloc_min": float(alloc.min()) if alloc.size else 0.0,
            "alloc_max": float(alloc.max()) if alloc.size else 0.0,
            "eligible_layers_json": json.dumps(t.eligible_layers),
            "allocated_ratios_json": json.dumps(t.allocated_compute_ratios),
            "allocated_counts_json": json.dumps(t.allocated_compute_counts),
        }
        action = t.action.detach().cpu().tolist()
        state = t.state.detach().cpu().tolist()
        for j in range(latent_dim):
            row[f"action_{j}"] = float(action[j]) if j < len(action) else float("nan")
        for j in range(state_dim):
            row[f"state_{j}"] = float(state[j]) if j < len(state) else float("nan")
        rows.append(row)
    atomic_write_csv(path, fields, rows)


def completed_calibration_records(records_dir: Path) -> List[Path]:
    result: List[Path] = []
    if not records_dir.exists():
        return result
    for sample_dir in sorted(records_dir.glob("sample_*")):
        if (sample_dir / "PROFILE_DONE").is_file() and (sample_dir / "profile.csv").is_file():
            result.append(sample_dir / "profile.csv")
    return result


def rebuild_sample_summary(records_dir: Path, output_csv: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for sample_dir in sorted(records_dir.glob("sample_*")) if records_dir.exists() else []:
        meta = sample_dir / "meta.json"
        if not meta.is_file():
            continue
        try:
            rows.append(json.loads(meta.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    atomic_write_csv(output_csv, fields, rows)

# -----------------------------------------------------------------------------
# 二、数据加载：只保留最简单、可复现的图片 + prompt 采样
# -----------------------------------------------------------------------------
def parse_prompt_markdown(
    prompt_path: Path,
    language: str,
    include_viton: bool,
) -> Tuple[List[Dict[str, str]], str]:
    """解析项目现有 portrait_prompts.md；逻辑与旧版本一致。"""
    text = prompt_path.read_text(encoding="utf-8")
    group_pattern = re.compile(r"^##\s+(.+?)\s*$([\s\S]*?)(?=^##\s+|\Z)", re.MULTILINE)
    entry_pattern = re.compile(r"^###\s+(.+?)\s*$([\s\S]*?)(?=^###\s+|\Z)", re.MULTILINE)
    label = "英文" if language == "english" else "中文"
    prompts: List[Dict[str, str]] = []
    negative_entries: Dict[str, str] = {}
    for group_match in group_pattern.finditer(text):
        group_name = group_match.group(1).strip()
        for entry_match in entry_pattern.finditer(group_match.group(2)):
            prompt_id = entry_match.group(1).strip()
            value_match = re.search(
                rf"{label}：\s*```text\s*([\s\S]*?)\s*```",
                entry_match.group(2),
            )
            if value_match is None:
                continue
            prompt_text = value_match.group(1).strip()
            if group_name == "负向 prompt":
                negative_entries[prompt_id] = prompt_text
                continue
            if group_name == "试衣" and not include_viton:
                continue
            if not include_viton and group_name not in DEFAULT_PROMPT_GROUPS:
                continue
            prompts.append({"group": group_name, "prompt_id": prompt_id, "prompt": prompt_text})
    if not prompts:
        raise ValueError(f"没有从 {prompt_path} 解析出可用 prompt。")
    negative_parts = [
        negative_entries[key]
        for key in ("ffhq_negative", "ffhq_negative_occlusion")
        if key in negative_entries
    ]
    if include_viton and "viton_negative" in negative_entries:
        negative_parts.append(negative_entries["viton_negative"])
    return prompts, ", ".join(negative_parts) if negative_parts else " "


def scan_images(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.as_posix(),
    )


def _manifest_expected_counts(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "calibration": int(args.blueprint_calibration_count),
        "train": int(args.train_count),
        "eval": int(args.eval_count),
    }


def _manifest_actual_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return {
        name: sum(str(r.get("split")) == name for r in rows)
        for name in ("calibration", "train", "eval")
    }


def _select_manifest_prefixes(
    rows: Sequence[Dict[str, Any]],
    expected_counts: Dict[str, int],
) -> List[Dict[str, Any]]:
    """从 master manifest 对每个 split 取稳定前缀；缩小数量绝不重新抽样。"""
    selected: List[Dict[str, Any]] = []
    for split in ("calibration", "train", "eval"):
        candidates = [r for r in rows if str(r.get("split")) == split]
        candidates.sort(key=lambda r: (int(r.get("split_index", 10**12)), int(r.get("sample_index", 10**12))))
        need = int(expected_counts[split])
        if len(candidates) < need:
            raise RuntimeError(
                f"master manifest 的 {split} 只有 {len(candidates)} 条，但本次需要 {need} 条。"
            )
        selected.extend(candidates[:need])
    return selected


def _new_manifest_row(
    *,
    image_path: Path,
    prompt_item: Dict[str, str],
    negative_prompt: str,
    split: str,
    split_index: int,
    sample_index: int,
    generation_seed_base: int,
) -> Dict[str, Any]:
    return {
        "sample_index": int(sample_index),
        "split": str(split),
        "split_index": int(split_index),
        "image_path": str(image_path),
        "prompt_group": prompt_item["group"],
        "prompt_id": prompt_item["prompt_id"],
        "prompt": prompt_item["prompt"],
        "negative_prompt": negative_prompt,
        "generation_seed": int(generation_seed_base) + int(sample_index),
    }


def build_manifest(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    """维护一个只增不减的 master manifest，并为本次运行选择各 split 的稳定前缀。

    强制续跑语义：
      * 现有 100/200/20，本次 100/100/10 -> 不重抽、不覆盖 master，只取原 train 前100/eval前10；
      * 现有 100/100/10，本次 100/200/20 -> 保留已有行，只给不足的 split 追加；
      * 卡数变化不会改变 manifest。
    """
    manifest_path = output_dir / "manifest.jsonl"
    expected_counts = _manifest_expected_counts(args)
    rows: List[Dict[str, Any]] = []

    if manifest_path.exists():
        try:
            rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            # 基本完整性：已有行必须能唯一定位样本。
            sample_ids = [int(r["sample_index"]) for r in rows]
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError("sample_index 存在重复")
            for r in rows:
                if str(r.get("split")) not in {"calibration", "train", "eval"}:
                    raise ValueError(f"未知 split={r.get('split')}")
        except Exception as exc:
            print(f"[manifest] 发现未完整/损坏 manifest，自动重建：{exc}", flush=True)
            rows = []

    prompts, negative_prompt = parse_prompt_markdown(
        Path(args.prompt_file), args.prompt_language, args.include_viton_prompts
    )
    images = scan_images(Path(args.dataset_root))

    if not rows:
        total = sum(expected_counts.values())
        if len(images) < total:
            raise ValueError(f"数据集只找到 {len(images)} 张图，但本次需要 {total} 张。")
        rng = random.Random(args.sampling_seed)
        selected_images = rng.sample(images, total)
        cursor = 0
        sample_index = 0
        for split in ("calibration", "train", "eval"):
            for split_index in range(expected_counts[split]):
                image_path = selected_images[cursor]
                cursor += 1
                prompt_item = prompts[rng.randrange(len(prompts))]
                rows.append(_new_manifest_row(
                    image_path=image_path,
                    prompt_item=prompt_item,
                    negative_prompt=negative_prompt,
                    split=split,
                    split_index=split_index,
                    sample_index=sample_index,
                    generation_seed_base=int(args.generation_seed),
                ))
                sample_index += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            manifest_path,
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        )
        print(f"[manifest] 新建 master manifest：{_manifest_actual_counts(rows)}", flush=True)
        return _select_manifest_prefixes(rows, expected_counts)

    actual = _manifest_actual_counts(rows)
    missing = {name: max(0, expected_counts[name] - actual[name]) for name in expected_counts}
    if not any(missing.values()):
        if actual == expected_counts:
            print(f"[manifest] 完全复用现有 manifest：{actual}", flush=True)
        else:
            print(
                f"[manifest] master manifest={actual}；本次只取稳定前缀={expected_counts}，不重新抽样。",
                flush=True,
            )
        return _select_manifest_prefixes(rows, expected_counts)

    # 只增不减：不足的 split 追加新样本，已有行完全不动。
    used_images = {str(r.get("image_path")) for r in rows}
    available = [img for img in images if str(img) not in used_images]
    if len(available) < sum(missing.values()):
        raise ValueError(
            f"manifest 需要追加 {sum(missing.values())} 张，但数据集只剩 {len(available)} 张未使用图片。"
        )
    rng = random.Random(int(args.sampling_seed) ^ 0x6A09E667)
    rng.shuffle(available)
    next_sample_index = max(int(r["sample_index"]) for r in rows) + 1
    cursor = 0
    for split in ("calibration", "train", "eval"):
        existing_split_indices = [
            int(r.get("split_index", -1)) for r in rows if str(r.get("split")) == split
        ]
        next_split_index = (max(existing_split_indices) + 1) if existing_split_indices else 0
        for _ in range(missing[split]):
            image_path = available[cursor]
            cursor += 1
            prompt_item = prompts[rng.randrange(len(prompts))]
            rows.append(_new_manifest_row(
                image_path=image_path,
                prompt_item=prompt_item,
                negative_prompt=negative_prompt,
                split=split,
                split_index=next_split_index,
                sample_index=next_sample_index,
                generation_seed_base=int(args.generation_seed),
            ))
            next_split_index += 1
            next_sample_index += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        manifest_path,
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
    )
    print(
        f"[manifest] 原有={actual}，只追加={missing}，master现在={_manifest_actual_counts(rows)}。",
        flush=True,
    )
    return _select_manifest_prefixes(rows, expected_counts)


def build_manifest_distributed(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    """多卡只允许 rank0 创建/扩展 master manifest；其他 rank 等待后读取相同稳定前缀。"""
    manifest_path = output_dir / "manifest.jsonl"
    if is_main_process():
        build_manifest(args, output_dir)
    dist_barrier()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 未生成：{manifest_path}")
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return _select_manifest_prefixes(rows, _manifest_expected_counts(args))


# -----------------------------------------------------------------------------
# 三、Fresh Blueprint：当前模型上重新校准，不读取任何旧 schedule
# -----------------------------------------------------------------------------
def _smooth_grid(grid: np.ndarray, radius: int) -> np.ndarray:
    """与旧 Blueprint v4 相同：对 step×block 风险做局部中值平滑。"""
    if radius <= 0:
        return grid.copy()
    result = grid.copy()
    steps, layers = grid.shape
    for step in range(1, steps):
        for layer in range(1, layers - 1):
            s0, s1 = max(1, step - radius), min(steps, step + radius + 1)
            b0, b1 = max(1, layer - radius), min(layers - 1, layer + radius + 1)
            window = grid[s0:s1, b0:b1]
            finite = window[np.isfinite(window)]
            if finite.size:
                result[step, layer] = float(np.median(finite))
    return result


def _profile_one_calibration_sample(
    pipe,
    row: Dict[str, Any],
    args: argparse.Namespace,
    sample_dir: Path,
) -> None:
    """跑完一个 calibration 样本后立刻把图像和 step×block 统计表原子落盘。"""
    sample_args = make_sample_args(args, row)
    image = load_input_image(str(row["image_path"]))
    transformer = pipe.transformer
    blocks = list(transformer.transformer_blocks)
    forwards_per_step = infer_forwards_per_step(sample_args)
    controller = ResidualProfileController(
        transformer_blocks=blocks,
        original_transformer_forward=transformer.forward,
        args=sample_args,
        forwards_per_step=forwards_per_step,
    )

    print(
        f"[blueprint-calibration][rank{get_rank()}] sample={int(row['split_index']) + 1}/"
        f"{args.blueprint_calibration_count} 开始",
        flush=True,
    )
    started = time.perf_counter()
    with replace_transformer_forward(transformer, controller):
        generated_image = generate_image(pipe, [image], sample_args)
    elapsed = time.perf_counter() - started
    controller.validate_complete()

    image_grid = np.full(
        (forwards_per_step, args.num_inference_steps, len(blocks)),
        np.nan,
        dtype=np.float32,
    )
    text_grid = np.full_like(image_grid, np.nan)
    for item in controller.rows:
        step = int(item["step_index_0based"])
        branch = int(item["branch_index_0based"])
        block = int(item["block_index_0based"])
        if item.get("image_relative_l2") is not None:
            image_grid[branch, step, block] = float(item["image_relative_l2"])
        if item.get("text_relative_l2") is not None:
            text_grid[branch, step, block] = float(item["text_relative_l2"])

    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image_atomic(image, sample_dir / "input.png")
    save_image_atomic(generated_image, sample_dir / "rendered.png")
    write_profile_table(sample_dir / "profile.csv", image_grid, text_grid)
    atomic_write_json(sample_dir / "meta.json", {
        "sample_index": int(row["sample_index"]),
        "split_index": int(row["split_index"]),
        "image_path": str(row["image_path"]),
        "prompt_id": str(row["prompt_id"]),
        "prompt": str(row["prompt"]),
        "generation_seed": int(row["generation_seed"]),
        "forwards_per_step": int(forwards_per_step),
        "num_steps": int(args.num_inference_steps),
        "total_layers": int(len(blocks)),
        "elapsed_seconds": float(elapsed),
        "legacy_imported": False,
        "has_input_image": True,
        "has_rendered_image": True,
    })

    # PROFILE_DONE 是唯一完成标记，最后写。中途崩溃只会留下不完整目录，重启会自动覆盖重跑。
    touch_done(sample_dir / "PROFILE_DONE")

    controller.previous_caches.clear()
    controller.rows.clear()
    del controller, image, generated_image, image_grid, text_grid
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_cuda_memory("blueprint:sample_done", args.device)


def _inspect_profile_table(path: Path, total_layers: int) -> Tuple[int, int]:
    max_branch = -1
    max_step = -1
    expected_img = f"image_b{total_layers - 1:03d}"
    expected_txt = f"text_b{total_layers - 1:03d}"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or expected_img not in reader.fieldnames or expected_txt not in reader.fieldnames:
            raise RuntimeError(f"profile 表列不完整：{path}")
        for row in reader:
            max_branch = max(max_branch, int(row["branch_index"]))
            max_step = max(max_step, int(row["step_index"]))
    return max_branch + 1, max_step + 1


def _import_legacy_npz_records(blueprint_dir: Path, records_dir: Path, args: argparse.Namespace) -> None:
    """把 v5 已完成 npz 转成新 profile.csv，保证升级版本后统计可以断点续跑。

    旧版没有保存 input/rendered 图，因此这里只导入统计；新 v6 跑出的样本都会保存两张图。
    """
    legacy_dir = blueprint_dir / "calibration_samples"
    if not legacy_dir.exists():
        return
    for npz_path in sorted(legacy_dir.glob("[0-9][0-9][0-9][0-9][0-9].npz")):
        try:
            slot = int(npz_path.stem)
        except ValueError:
            continue
        if slot >= int(args.blueprint_calibration_count):
            continue
        sample_dir = records_dir / f"sample_{slot:05d}"
        if (sample_dir / "PROFILE_DONE").is_file() and (sample_dir / "profile.csv").is_file():
            continue
        try:
            with np.load(npz_path, allow_pickle=False) as payload:
                image_grid = np.asarray(payload["image_relative_l2"], dtype=np.float32)
                text_grid = np.asarray(payload["text_relative_l2"], dtype=np.float32)
                sample_index = int(payload["sample_index"]) if "sample_index" in payload else slot
                forwards_per_step = int(payload["forwards_per_step"])
                num_steps = int(payload["num_steps"])
                total_layers = int(payload["total_layers"])
            sample_dir.mkdir(parents=True, exist_ok=True)
            write_profile_table(sample_dir / "profile.csv", image_grid, text_grid)
            atomic_write_json(sample_dir / "meta.json", {
                "sample_index": sample_index,
                "split_index": slot,
                "image_path": "",
                "prompt_id": "",
                "prompt": "",
                "generation_seed": None,
                "forwards_per_step": forwards_per_step,
                "num_steps": num_steps,
                "total_layers": total_layers,
                "elapsed_seconds": None,
                "legacy_imported": True,
                "legacy_npz": str(npz_path),
                "has_input_image": False,
                "has_rendered_image": False,
            })
            touch_done(sample_dir / "PROFILE_DONE")
            print(f"[blueprint-resume] 导入旧统计 {npz_path.name} -> {sample_dir.name}/profile.csv", flush=True)
        except Exception as exc:
            print(f"[blueprint-resume] 跳过损坏旧 npz {npz_path}: {exc}", flush=True)


def _build_schedule_from_profile_tables(
    profile_files: Sequence[Path],
    args: argparse.Namespace,
    total_layers: int,
) -> Dict[str, Any]:
    """从逐样本 CSV 表按 Block 小块做精确 P90；任何时刻都不堆全部样本。"""
    expected = int(args.blueprint_calibration_count)
    if len(profile_files) < expected:
        raise RuntimeError(f"Fresh Blueprint calibration 完成 {len(profile_files)}/{expected}，不能生成 schedule。")
    profile_files = list(sorted(profile_files)[:expected])
    if not profile_files:
        raise RuntimeError("没有找到 calibration profile.csv。")

    branch_count, num_steps = _inspect_profile_table(profile_files[0], total_layers)
    if num_steps != int(args.num_inference_steps):
        raise RuntimeError(f"profile step 数={num_steps} 与当前 {args.num_inference_steps} 不一致。")
    observations = len(profile_files) * branch_count
    valid_steps = max(0, num_steps - 1)
    block_chunk = max(1, min(int(args.profile_aggregate_block_chunk), total_layers))

    image_q = np.full((num_steps, total_layers), np.nan, dtype=np.float64)
    text_q = np.full_like(image_q, np.nan)
    print(
        f"[blueprint-aggregate] 流式表格汇总：samples={len(profile_files)}, branches={branch_count}, "
        f"steps={num_steps}, blocks={total_layers}, chunk={block_chunk}",
        flush=True,
    )

    # 每一轮只分配 [N*branch, 49, chunk]，chunk 算完立即释放。
    for block_start in range(0, total_layers, block_chunk):
        block_end = min(total_layers, block_start + block_chunk)
        width = block_end - block_start
        image_values = np.full((observations, valid_steps, width), np.nan, dtype=np.float32)
        text_values = np.full_like(image_values, np.nan)
        cursor = 0
        for path in profile_files:
            image_one, text_one = read_profile_chunk(
                path,
                branch_count=branch_count,
                num_steps=num_steps,
                block_start=block_start,
                block_end=block_end,
            )
            image_values[cursor:cursor + branch_count] = image_one[:, 1:, :]
            text_values[cursor:cursor + branch_count] = text_one[:, 1:, :]
            cursor += branch_count
            del image_one, text_one

        image_q[1:, block_start:block_end] = np.nanquantile(
            image_values, args.profile_quantile, axis=0
        )
        del image_values
        gc.collect()
        text_q[1:, block_start:block_end] = np.nanquantile(
            text_values, args.profile_quantile, axis=0
        )
        del text_values
        gc.collect()
        print(
            f"[blueprint-aggregate] P{args.profile_quantile * 100:.1f} 完成 "
            f"block {block_start + 1}-{block_end}/{total_layers}",
            flush=True,
        )

    image_smooth = _smooth_grid(image_q, int(args.profile_smoothing_radius))
    text_smooth = _smooth_grid(text_q, int(args.profile_smoothing_radius))
    eligible = (slice(1, num_steps), slice(1, total_layers - 1))
    image_scale = max(float(np.nanmedian(image_smooth[eligible])), 1e-12)
    text_scale = max(float(np.nanmedian(text_smooth[eligible])), 1e-12)
    combined = np.maximum(image_smooth / image_scale, text_smooth / text_scale)
    finite_risk = combined[eligible]
    finite_risk = finite_risk[np.isfinite(finite_risk)]
    if finite_risk.size == 0:
        raise RuntimeError("Fresh Blueprint 没有得到有效 residual risk。")
    risk_threshold = float(np.quantile(finite_risk, args.target_cache_ratio))

    ages = [0] * total_layers
    schedule: List[Dict[str, Any]] = []
    for step in range(num_steps):
        force_full = (
            step < int(args.force_full_first_steps)
            or (
                int(args.force_full_last_steps) > 0
                and step >= num_steps - int(args.force_full_last_steps)
            )
        )
        internal = list(range(1, total_layers - 1))
        if step == 0 or force_full:
            base_executed = set(range(total_layers))
            left_boundary = 1
            right_boundary = total_layers
            boundary_reason = "forced_full_step"
        else:
            unstable = [layer for layer in internal if float(combined[step, layer]) > risk_threshold]
            if unstable:
                left = min(unstable)
                right = max(unstable)
                base_executed = {0, total_layers - 1, *range(left, right + 1)}
                left_boundary = left + 1
                right_boundary = right + 1
                boundary_reason = "continuous_interval_covering_all_high_risk_blocks"
            else:
                base_executed = {0, total_layers - 1}
                left_boundary = None
                right_boundary = None
                boundary_reason = "all_internal_blocks_below_threshold"

        base_skipped = set(range(total_layers)) - base_executed
        effective_executed = set(base_executed)
        forced_refresh: List[int] = []
        if int(args.blueprint_max_cache_age) > 0:
            for layer in sorted(base_skipped):
                if ages[layer] >= int(args.blueprint_max_cache_age):
                    effective_executed.add(layer)
                    forced_refresh.append(layer)
        effective_skipped = set(range(total_layers)) - effective_executed
        for layer in range(total_layers):
            ages[layer] = 0 if layer in effective_executed else ages[layer] + 1

        schedule.append({
            "step_index_0based": step,
            "step_number_1based": step + 1,
            "mode": "full_compute" if len(effective_executed) == total_layers else "blue_line_cache",
            "boundary_reason": boundary_reason,
            "blue_line_left_compute_boundary_1based": left_boundary,
            "blue_line_right_compute_boundary_1based": right_boundary,
            "base_executed_blocks_0based": sorted(base_executed),
            "base_skipped_blocks_0based": sorted(base_skipped),
            "forced_refresh_blocks_0based": forced_refresh,
            "executed_blocks_0based": sorted(effective_executed),
            "skipped_blocks_0based": sorted(effective_skipped),
            "executed_block_count": len(effective_executed),
            "skipped_block_count": len(effective_skipped),
            "smoothed_risk_by_block": [
                0.0 if not np.isfinite(combined[step, layer]) else float(combined[step, layer])
                for layer in range(total_layers)
            ],
            "max_cache_age_after_step": max(ages),
        })

    total_full = num_steps * total_layers
    executed = sum(int(item["executed_block_count"]) for item in schedule)
    payload = {
        "strategy_version": "fresh_blue_line_streaming_tables_exact_quantile_v6",
        "freshly_calibrated": True,
        "streaming_profile_tables": True,
        "profile_aggregate_block_chunk": block_chunk,
        "calibration_sample_count": expected,
        "profile_observation_count": observations,
        "profile_quantile": float(args.profile_quantile),
        "target_cache_ratio_before_contiguous_constraint": float(args.target_cache_ratio),
        "profile_smoothing_radius": int(args.profile_smoothing_radius),
        "max_cache_age": int(args.blueprint_max_cache_age),
        "force_full_first_steps": int(args.force_full_first_steps),
        "force_full_last_steps": int(args.force_full_last_steps),
        "image_normalization_scale": image_scale,
        "text_normalization_scale": text_scale,
        "combined_risk_threshold": risk_threshold,
        "total_layers": total_layers,
        "num_inference_steps": num_steps,
        "total_full_block_forwards": total_full,
        "effective_executed_block_forwards": executed,
        "effective_skipped_block_forwards": total_full - executed,
        "effective_executed_block_fraction": executed / total_full,
        "effective_cache_fraction": 1.0 - executed / total_full,
        "theoretical_block_speedup": total_full / max(1, executed),
        "schedule": schedule,
    }

    # 最终 step×block 汇总表仅 50*60 行，便于后续画图/检查，不含样本大对象。
    blueprint_dir = profile_files[0].parent.parent.parent
    aggregate_rows: List[Dict[str, Any]] = []
    schedule_map = {int(x["step_index_0based"]): x for x in schedule}
    for step in range(num_steps):
        executed_set = set(schedule_map[step]["executed_blocks_0based"])
        for block in range(total_layers):
            aggregate_rows.append({
                "step_index": step,
                "block_index": block,
                "image_p_quantile": float(image_q[step, block]) if np.isfinite(image_q[step, block]) else "",
                "text_p_quantile": float(text_q[step, block]) if np.isfinite(text_q[step, block]) else "",
                "image_smoothed": float(image_smooth[step, block]) if np.isfinite(image_smooth[step, block]) else "",
                "text_smoothed": float(text_smooth[step, block]) if np.isfinite(text_smooth[step, block]) else "",
                "combined_risk": float(combined[step, block]) if np.isfinite(combined[step, block]) else "",
                "risk_threshold": risk_threshold,
                "executed": int(block in executed_set),
            })
    atomic_write_csv(
        blueprint_dir / "profile_aggregate.csv",
        list(aggregate_rows[0].keys()),
        aggregate_rows,
    )
    return payload


def _schedule_matches_current(payload: Dict[str, Any], args: argparse.Namespace, total_layers: int) -> bool:
    checks = [
        int(payload.get("calibration_sample_count", -1)) == int(args.blueprint_calibration_count),
        int(payload.get("num_inference_steps", -1)) == int(args.num_inference_steps),
        int(payload.get("total_layers", -1)) == int(total_layers),
        abs(float(payload.get("profile_quantile", -1)) - float(args.profile_quantile)) < 1e-12,
        abs(float(payload.get("target_cache_ratio_before_contiguous_constraint", -1)) - float(args.target_cache_ratio)) < 1e-12,
        int(payload.get("profile_smoothing_radius", -1)) == int(args.profile_smoothing_radius),
        int(payload.get("max_cache_age", -1)) == int(args.blueprint_max_cache_age),
    ]
    return all(checks)


def build_fresh_blueprint(
    pipe,
    calibration_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    total_layers: int,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """强制断点续跑：每个 calibration 样本有独立表格和完成标记。"""
    blueprint_dir = output_dir / "blueprint"
    records_dir = blueprint_dir / "calibration_records"
    schedule_path = blueprint_dir / "blue_line_schedule.json"

    if bool(args.rebuild_blueprint) and is_main_process():
        if schedule_path.is_file():
            schedule_path.unlink()
        if records_dir.exists():
            shutil.rmtree(records_dir)
        # 显式重建时也清掉旧 v5 npz，避免它们被自动重新导入。
        legacy_dir = blueprint_dir / "calibration_samples"
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)
        print("[blueprint] 显式 --rebuild-blueprint：清空本次 calibration，从头重建。", flush=True)
    dist_barrier()

    records_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process() and not bool(args.rebuild_blueprint):
        _import_legacy_npz_records(blueprint_dir, records_dir, args)
    dist_barrier()

    existing_profiles = completed_calibration_records(records_dir)
    if schedule_path.is_file() and len(existing_profiles) >= int(args.blueprint_calibration_count):
        payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        if _schedule_matches_current(payload, args, total_layers):
            if is_main_process():
                print(f"[blueprint-resume] schedule 与当前参数一致，直接续用：{schedule_path}", flush=True)
            schedule = {int(item["step_index_0based"]): item for item in payload["schedule"]}
            return schedule, payload
        if is_main_process():
            print("[blueprint-resume] calibration 表可复用，但数量/统计参数变化；只重新汇总 schedule。", flush=True)

    # 样本级断点：已有 PROFILE_DONE 的 slot 永远不重跑；只补缺失 slot。
    pending: List[Tuple[int, Dict[str, Any]]] = []
    for row in calibration_rows:
        slot = int(row["split_index"])
        sample_dir = records_dir / f"sample_{slot:05d}"
        if (sample_dir / "PROFILE_DONE").is_file() and (sample_dir / "profile.csv").is_file():
            continue
        pending.append((slot, row))
    local_pending = [item for i, item in enumerate(pending) if i % get_world_size() == get_rank()]
    if is_main_process():
        print(
            f"[blueprint-resume] target={args.blueprint_calibration_count}, "
            f"completed={len(completed_calibration_records(records_dir))}, pending={len(pending)}",
            flush=True,
        )
    for slot, row in local_pending:
        sample_dir = records_dir / f"sample_{slot:05d}"
        _profile_one_calibration_sample(pipe, row, args, sample_dir)
    dist_barrier()

    if is_main_process():
        profile_files = completed_calibration_records(records_dir)
        if len(profile_files) < int(args.blueprint_calibration_count):
            raise RuntimeError(
                f"calibration 完成 {len(profile_files)}/{args.blueprint_calibration_count}，仍有缺失样本。"
            )
        payload = _build_schedule_from_profile_tables(profile_files, args, total_layers)
        atomic_write_json(schedule_path, payload)

        matrix_path = blueprint_dir / "blue_line_schedule_matrix.csv"
        matrix_rows: List[Dict[str, Any]] = []
        for item in payload["schedule"]:
            executed = set(int(v) for v in item["executed_blocks_0based"])
            row: Dict[str, Any] = {"step": item["step_number_1based"]}
            for i in range(total_layers):
                row[f"block_{i + 1:02d}"] = 1 if i in executed else 0
            matrix_rows.append(row)
        atomic_write_csv(matrix_path, list(matrix_rows[0].keys()), matrix_rows)
        rebuild_sample_summary(records_dir, blueprint_dir / "calibration_sample_summary.csv")
        print(f"[blueprint] Fresh Blueprint 完成：{schedule_path}", flush=True)
    dist_barrier()

    payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule = {int(item["step_index_0based"]): item for item in payload["schedule"]}
    return schedule, payload



def make_full_schedule(num_steps: int, total_layers: int) -> Dict[int, Dict[str, Any]]:
    """完整 Block 路径也包装成 schedule，方便和 Blueprint 复用同一 Token controller。"""
    layers = list(range(total_layers))
    return {
        step: {
            "step_index_0based": step,
            "mode": "full_compute",
            "executed_blocks_0based": layers,
            "base_executed_blocks_0based": layers,
            "base_skipped_blocks_0based": [],
            "forced_refresh_blocks_0based": [],
            "smoothed_risk_by_block": [0.0] * total_layers,
        }
        for step in range(num_steps)
    }


# -----------------------------------------------------------------------------
# 四、低维连续 TD3 + 可变长度 Block-budget controller
# -----------------------------------------------------------------------------
class TD3Actor(nn.Module):
    """固定维 state -> 固定维 latent action∈[-1,1]^K。零初始化输出严格对应 Fixed25。"""

    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state))


class TD3Critic(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(state_dim) + int(latent_dim), hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    step_index: int
    branch_index: int
    eligible_layers: List[int]
    allocated_compute_ratios: List[float]
    allocated_compute_counts: List[int]


class PolicyRuntime:
    """TD3 rollout wrapper。每个 timestep / branch 只记录一个 transition。"""

    def __init__(
        self,
        model: TD3Actor,
        device: str,
        explore: bool,
        *,
        exploration_noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.explore = bool(explore)
        self.exploration_noise = float(exploration_noise)
        self.latent_dim = int(model.latent_dim)
        self.state_dim = int(model.state_dim)
        self.transitions: List[Transition] = []
        self.pending_by_branch: Dict[int, int] = {}
        self.rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)

    @torch.no_grad()
    def act(self, state_values: Sequence[float]) -> np.ndarray:
        state = torch.tensor(state_values, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.model(state).squeeze(0).detach().cpu().numpy().astype(np.float64)
        if self.explore and self.exploration_noise > 0:
            action = action + self.rng.normal(0.0, self.exploration_noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def record_step(
        self,
        *,
        state_values: Sequence[float],
        action: Sequence[float],
        step_index: int,
        branch_index: int,
        eligible_layers: Sequence[int],
        allocated_compute_counts: Sequence[int],
        token_count: int,
    ) -> int:
        state = torch.tensor(state_values, dtype=torch.float32)
        # 当前 state 就是同 branch 上一个 transition 的 next_state。
        prev = self.pending_by_branch.get(int(branch_index))
        if prev is not None and not self.transitions[prev].done:
            self.transitions[prev].next_state = state.clone()
        counts = [int(x) for x in allocated_compute_counts]
        ratios = [float(x) / max(1, int(token_count)) for x in counts]
        idx = len(self.transitions)
        self.transitions.append(Transition(
            state=state,
            action=torch.tensor(action, dtype=torch.float32),
            reward=0.0,
            next_state=torch.zeros_like(state),
            done=False,
            step_index=int(step_index),
            branch_index=int(branch_index),
            eligible_layers=[int(x) for x in eligible_layers],
            allocated_compute_ratios=ratios,
            allocated_compute_counts=counts,
        ))
        self.pending_by_branch[int(branch_index)] = idx
        return idx

    def set_reward(self, index: int, reward: float) -> None:
        self.transitions[int(index)].reward = float(reward)

    def finish_branch(self, branch_index: int) -> None:
        idx = self.pending_by_branch.pop(int(branch_index), None)
        if idx is not None:
            self.transitions[idx].done = True
            self.transitions[idx].next_state = torch.zeros_like(self.transitions[idx].state)


class BlockBudgetTokenController(BlueLineTokenScheduledController):
    """Fixed25 或低维 TD3 joint allocation。

    TD3 模式下：
      1) 每个 timestep/branch 收集当前 schedule 真正执行且允许 token sparse 的 eligible layers；
      2) 无论 eligible 数量是 29、30、58 还是其它值，Actor 始终只输出 K 维 latent action；
      3) K 个控制点按当前 eligible 序列的相对位置 [0,1] 插值到 N 个 Block；
      4) softmax 得到相对额外预算权重，在每 Block 最低 min_compute_ratio 之上分配剩余预算；
      5) 整数化后严格保证总 compute token == Fixed25 总预算。

    action==0 -> uniform weights -> 每个 eligible Block 恰好 Fixed25。这样 TD3 从可靠基线出发，
    训练探索只是在 Fixed25 周围逐渐重新分配，而不是一开始就产生极端随机预算。
    """

    def __init__(
        self,
        *args,
        budget_mode: str,
        target_compute_ratio: float,
        policy_runtime: Optional[PolicyRuntime] = None,
        fixed_score_map: Optional[Dict[Tuple[int, int], float]] = None,
        base_is_blueprint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if budget_mode not in {"fixed25", "rl25"}:
            raise ValueError(f"未知 budget_mode={budget_mode}")
        self.budget_mode = budget_mode
        self.target_compute_ratio = float(target_compute_ratio)
        self.policy_runtime_new = policy_runtime
        self.fixed_score_map = fixed_score_map or {}
        self.base_is_blueprint = bool(base_is_blueprint)
        self.step_plans: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.step_transition_index: Dict[Tuple[int, int], int] = {}
        self.last_observed_change: Dict[Tuple[int, int], Tuple[float, float]] = {}

        self.rl_policy = None
        self.token_policy_mode = "fixed"
        self.token_budget_mode = "per_block"

    def _eligible_layers(self, step_index: int) -> List[int]:
        item = self.schedule[step_index]
        layers = [int(v) for v in item["executed_blocks_0based"]]
        if step_index == 0:
            return []
        if not self.cache_edge_blocks:
            layers = [v for v in layers if v not in {0, self.total_layers - 1}]
        return layers

    @staticmethod
    def _risk(item: Dict[str, Any], layer_index: int) -> float:
        values = item.get("smoothed_risk_by_block") or item.get("risk_by_block")
        if isinstance(values, list) and layer_index < len(values):
            return float(values[layer_index])
        return 0.0

    @staticmethod
    def _interp(values: Sequence[float], out_dim: int) -> List[float]:
        out_dim = max(1, int(out_dim))
        vals = np.asarray(list(values), dtype=np.float64)
        if vals.size == 0:
            return [0.0] * out_dim
        if vals.size == 1:
            return [float(vals[0])] * out_dim
        src = np.linspace(0.0, 1.0, vals.size)
        dst = np.linspace(0.0, 1.0, out_dim)
        return np.interp(dst, src, vals).astype(np.float64).tolist()

    def _age_features(self, branch_index: int, layer_index: int, step_index: int) -> Tuple[float, float]:
        refresh = self.token_last_refresh_steps.get(branch_index, {}).get(layer_index)
        if refresh is None or not torch.is_tensor(refresh) or refresh.numel() == 0:
            return 0.0, 0.0
        ages = step_index - refresh
        mean_age = float(ages.float().mean().item()) / max(1, int(self.args.num_inference_steps))
        expired = (
            float((ages > self.max_token_cache_age).float().mean().item())
            if self.max_token_cache_age > 0 else 0.0
        )
        return mean_age, expired

    def _build_joint_state(
        self,
        *,
        step_index: int,
        branch_index: int,
        item: Dict[str, Any],
        eligible_layers: Sequence[int],
    ) -> List[float]:
        if self.policy_runtime_new is None:
            raise RuntimeError("TD3 rl25 缺少 policy runtime。")
        k = int(self.policy_runtime_new.latent_dim)
        layers = [int(v) for v in eligible_layers]
        n = len(layers)
        internal_total = max(1, self.total_layers - (0 if self.cache_edge_blocks else 2))
        layer_pos = [v / max(1, self.total_layers - 1) for v in layers]
        risks = [math.log1p(max(0.0, self._risk(item, v))) for v in layers]
        prev_mean = [math.log1p(max(0.0, self.last_observed_change.get((branch_index, v), (0.0, 0.0))[0])) for v in layers]
        prev_max = [math.log1p(max(0.0, self.last_observed_change.get((branch_index, v), (0.0, 0.0))[1])) for v in layers]
        age_mean: List[float] = []
        expired: List[float] = []
        for v in layers:
            a, e = self._age_features(branch_index, v, step_index)
            age_mean.append(a)
            expired.append(e)

        globals_ = [
            step_index / max(1, int(self.args.num_inference_steps) - 1),
            1.0 if self.base_is_blueprint else 0.0,
            n / internal_total,
            (layers[0] / max(1, self.total_layers - 1)) if layers else 0.0,
            (layers[-1] / max(1, self.total_layers - 1)) if layers else 0.0,
            float(np.mean(risks)) if risks else 0.0,
        ]
        anchors = [
            self._interp(layer_pos, k),
            self._interp(risks, k),
            self._interp(prev_mean, k),
            self._interp(prev_max, k),
            self._interp(age_mean, k),
            self._interp(expired, k),
        ]
        state = list(globals_)
        for i in range(k):
            for feature in anchors:
                state.append(float(feature[i]))
        expected = td3_state_dim(k)
        if len(state) != expected:
            raise RuntimeError(f"TD3 state dim 错误：{len(state)} != {expected}")
        return state

    def _observe_current_change(self, branch_index: int, layer_index: int, image_input: torch.Tensor) -> None:
        source = self.token_source_inputs.get(branch_index, {}).get(layer_index)
        if source is None or source.shape != image_input.shape:
            self.last_observed_change[(branch_index, layer_index)] = (0.0, 0.0)
            return
        scores = self._score_values(image_input, source)
        self.last_observed_change[(branch_index, layer_index)] = (
            float(scores.mean().item()), float(scores.max().item())
        )

    @staticmethod
    def _bounded_integer_allocation(
        weights: np.ndarray,
        *,
        total_budget: int,
        min_count: int,
        max_count: int,
    ) -> List[int]:
        n = int(weights.size)
        if n <= 0:
            return []
        lower_total = n * int(min_count)
        upper_total = n * int(max_count)
        target = max(lower_total, min(upper_total, int(total_budget)))
        counts = np.full(n, int(min_count), dtype=np.int64)
        capacity = np.full(n, int(max_count - min_count), dtype=np.int64)
        remaining = int(target - lower_total)
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 1e-12)
        while remaining > 0:
            active = capacity > 0
            if not np.any(active):
                raise RuntimeError("TD3 预算投影失败：容量耗尽但仍有剩余预算。")
            wa = w.copy()
            wa[~active] = 0.0
            wa_sum = float(wa.sum())
            if wa_sum <= 0:
                wa[active] = 1.0
                wa_sum = float(wa.sum())
            raw = remaining * wa / wa_sum
            add = np.floor(raw).astype(np.int64)
            add = np.minimum(add, capacity)
            used = int(add.sum())
            if used > 0:
                counts += add
                capacity -= add
                remaining -= used
                continue
            # remaining 小于 active 数时按最大 fractional weight 每次补 1，保证精确闭合。
            order = np.argsort(-raw)
            progressed = False
            for idx in order:
                if remaining <= 0:
                    break
                if capacity[idx] <= 0:
                    continue
                counts[idx] += 1
                capacity[idx] -= 1
                remaining -= 1
                progressed = True
            if not progressed:
                raise RuntimeError("TD3 预算整数化无法继续。")
        if int(counts.sum()) != target:
            raise RuntimeError(f"TD3 总预算不闭合：{counts.sum()} != {target}")
        return [int(x) for x in counts.tolist()]

    def _ensure_step_plan(
        self,
        *,
        step_index: int,
        branch_index: int,
        item: Dict[str, Any],
        token_count: int,
    ) -> Dict[str, Any]:
        key = (step_index, branch_index)
        if key in self.step_plans:
            plan = self.step_plans[key]
            if int(plan["token_count"]) != int(token_count):
                raise RuntimeError("同一 timestep 不同 Block 的 image token 数不一致，无法使用 joint TD3 allocation。")
            return plan
        eligible = self._eligible_layers(step_index)
        n = len(eligible)
        if n <= 0:
            plan = {"eligible": [], "counts": {}, "token_count": int(token_count), "action": []}
            self.step_plans[key] = plan
            return plan
        fixed_count = int(math.ceil(token_count * self.target_compute_ratio))
        min_count = max(1, int(math.ceil(token_count * float(self.min_compute_ratio))))
        total_budget = max(n * min_count, min(n * token_count, n * fixed_count))
        state_values = self._build_joint_state(
            step_index=step_index, branch_index=branch_index, item=item, eligible_layers=eligible
        )
        if self.policy_runtime_new is None:
            raise RuntimeError("rl25 模式缺少 TD3 policy runtime。")
        action = self.policy_runtime_new.act(state_values)
        # K 维 action 在当前 N 个 eligible Block 的相对位置上插值。N 可以每 step 不同。
        if n == 1:
            decoded = np.asarray([float(np.mean(action))], dtype=np.float64)
        else:
            src = np.linspace(0.0, 1.0, len(action))
            dst = np.linspace(0.0, 1.0, n)
            decoded = np.interp(dst, src, np.asarray(action, dtype=np.float64))
        logits = decoded * float(self.args.td3_allocation_logit_scale)
        logits = logits - float(np.max(logits))
        weights = np.exp(logits)
        counts = self._bounded_integer_allocation(
            weights, total_budget=total_budget, min_count=min_count, max_count=token_count
        )
        # action=0 必须精确还原 Fixed25；这也是初始化时的行为。
        if np.allclose(action, 0.0, atol=1e-12):
            expected = [fixed_count] * n
            if counts != expected:
                raise RuntimeError(f"TD3 零动作未还原 Fixed25：前5={counts[:5]} vs {expected[:5]}")
        transition_index = self.policy_runtime_new.record_step(
            state_values=state_values,
            action=action,
            step_index=step_index,
            branch_index=branch_index,
            eligible_layers=eligible,
            allocated_compute_counts=counts,
            token_count=token_count,
        )
        self.step_transition_index[key] = transition_index
        plan = {
            "eligible": eligible,
            "counts": {int(layer): int(count) for layer, count in zip(eligible, counts)},
            "token_count": int(token_count),
            "action": [float(x) for x in action],
            "total_budget": int(total_budget),
        }
        self.step_plans[key] = plan
        return plan

    def _prepare_token_decision(self, **kwargs) -> Dict[str, Any]:
        step_index = int(kwargs["step_index"])
        branch_index = int(kwargs["branch_index"])
        item = kwargs["item"]
        layer_index = int(kwargs["layer_index"])
        image_input = kwargs["image_input"]
        token_count = int(image_input.shape[1])
        eligible_layers = self._eligible_layers(step_index)
        eligible = layer_index in set(eligible_layers)

        if self.budget_mode == "fixed25" or not eligible:
            old_ratio = self.token_cache_ratio
            self.token_cache_ratio = 1.0 - self.target_compute_ratio
            try:
                result = super()._prepare_token_decision(**kwargs)
            finally:
                self.token_cache_ratio = old_ratio
            return result

        # 先用“上一时刻已经观测到的全体 Block 特征”生成整步计划，再写入当前 Block 的新观测，
        # 避免 joint action 偷看尚未执行的当前 timestep 后续 Block。
        plan = self._ensure_step_plan(
            step_index=step_index, branch_index=branch_index, item=item, token_count=token_count
        )
        self._observe_current_change(branch_index, layer_index, image_input)
        compute_count = int(plan["counts"][layer_index])
        effective_ratio = max(0.0, (compute_count - 1e-6) / token_count)
        old_ratio = self.token_cache_ratio
        self.token_cache_ratio = 1.0 - effective_ratio
        try:
            result = super()._prepare_token_decision(**kwargs)
        finally:
            self.token_cache_ratio = old_ratio
        actual_compute = int(result["metadata"]["computed_image_token_count"])
        if actual_compute != compute_count:
            raise RuntimeError(
                f"TD3 joint budget 映射失败 step={step_index} block={layer_index}: "
                f"计划 {compute_count}，实际 {actual_compute}"
            )
        result["metadata"].update({
            "lowdim_td3": True,
            "td3_latent_action": plan["action"],
            "td3_eligible_block_count": len(plan["eligible"]),
            "td3_actual_compute_ratio": actual_compute / token_count,
        })
        return result

    def __call__(self, *positional_args, **keyword_args):
        step_index = self.call_index // self.forwards_per_step
        branch_index = self.call_index % self.forwards_per_step
        output = super().__call__(*positional_args, **keyword_args)
        row = self.branch_step_rows[-1]
        if self.budget_mode == "rl25" and self.policy_runtime_new is not None:
            key = (step_index, branch_index)
            fixed_score = self.fixed_score_map.get(key)
            reward: Optional[float] = None
            transition_index = self.step_transition_index.get(key)
            if fixed_score is not None and transition_index is not None:
                rl_score = float(row["score"])
                reward = (float(fixed_score) - rl_score) / (abs(float(fixed_score)) + 1e-8)
                reward = float(np.clip(reward, -2.0, 2.0))
                self.policy_runtime_new.set_reward(transition_index, reward)
            row["td3_relative_fixed_reward"] = reward
            if step_index >= int(self.args.num_inference_steps) - 1:
                self.policy_runtime_new.finish_branch(branch_index)
        return output

def release_controller_cuda_state(controller: Any) -> None:
    """尽快释放 controller 内部不再需要的 CUDA cache。

    Controller 会保存上一 timestep 的 Block residual、token source、KV cache 等大 Tensor。
    Python 变量即使后面不再使用，只要 controller 还活着，这些 Tensor 就不会释放。
    因此 fixed25 得到 score 后必须先清掉，再启动 RL25，避免两个 sparse controller
    的显存叠加。
    """
    if controller is None:
        return
    for name in (
        "previous_caches",
        "last_refresh_steps",
        "last_block_refresh_steps",
        "token_source_inputs",
        "token_last_refresh_steps",
        "image_kv_caches",
        "global_budget_states",
        "step_budget_state",
        "step_transition_indices",
        "last_actual_ratio",
        "step_plans",
        "step_transition_index",
        "last_observed_change",
        "branch_step_rows",
        "block_action_rows",
        "token_action_rows",
    ):
        value = getattr(controller, name, None)
        if hasattr(value, "clear"):
            value.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_teacher_references(teacher_refs: Optional[Dict[Any, Any]]) -> None:
    if teacher_refs is not None:
        teacher_refs.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_cuda_memory(label: str, device: str) -> None:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return
    dev = torch.device(device)
    allocated = torch.cuda.memory_allocated(dev) / 1024**3
    reserved = torch.cuda.memory_reserved(dev) / 1024**3
    peak = torch.cuda.max_memory_allocated(dev) / 1024**3
    print(
        f"[cuda-memory][rank{get_rank()}] {label}: "
        f"allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB, peak={peak:.2f} GiB",
        flush=True,
    )


# -----------------------------------------------------------------------------
# 六、TD3 replay / 更新 / 自动收敛
# -----------------------------------------------------------------------------
def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - float(tau)).add_(sp, alpha=float(tau))


def _hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def _flatten_alloc(transitions: Sequence[Transition]) -> np.ndarray:
    values: List[float] = []
    for t in transitions:
        values.extend(float(x) for x in t.allocated_compute_ratios)
    return np.asarray(values, dtype=np.float64)


def td3_update(
    *,
    actor: TD3Actor,
    actor_target: TD3Actor,
    critic1: TD3Critic,
    critic2: TD3Critic,
    critic1_target: TD3Critic,
    critic2_target: TD3Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    replay: List[Transition],
    rollout_transitions: Sequence[Transition],
    args: argparse.Namespace,
    gradient_update_index: int,
) -> Tuple[Dict[str, float], int]:
    device = torch.device(args.policy_device)
    rewards_rollout = np.asarray([float(t.reward) for t in rollout_transitions], dtype=np.float64)
    alloc = _flatten_alloc(rollout_transitions)
    base_stats: Dict[str, float] = {
        "critic_loss": 0.0,
        "actor_loss": 0.0,
        "q1_mean": 0.0,
        "mean_reward": float(rewards_rollout.mean()) if rewards_rollout.size else 0.0,
        "std_reward": float(rewards_rollout.std()) if rewards_rollout.size else 0.0,
        "mean_allocated_compute_ratio": float(alloc.mean()) if alloc.size else 0.0,
        "std_allocated_compute_ratio": float(alloc.std()) if alloc.size else 0.0,
        "policy_action_delta": 1.0,
        "actor_update_count": 0.0,
        "replay_size": float(len(replay)),
    }
    if len(replay) < max(int(args.td3_warmup_transitions), int(args.td3_batch_size)):
        return base_stats, gradient_update_index

    critic_losses: List[float] = []
    actor_losses: List[float] = []
    q_means: List[float] = []
    deltas: List[float] = []
    actor_update_count = 0
    batch_size = min(int(args.td3_batch_size), len(replay))
    for _ in range(max(1, int(args.td3_gradient_steps))):
        indices = np.random.randint(0, len(replay), size=batch_size)
        batch = [replay[int(i)] for i in indices]
        states = torch.stack([t.state for t in batch]).to(device)
        actions = torch.stack([t.action for t in batch]).to(device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        next_states = torch.stack([t.next_state for t in batch]).to(device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

        with torch.no_grad():
            noise = torch.randn_like(actions) * float(args.td3_target_noise)
            noise = noise.clamp(-float(args.td3_target_noise_clip), float(args.td3_target_noise_clip))
            next_actions = (actor_target(next_states) + noise).clamp(-1.0, 1.0)
            tq1 = critic1_target(next_states, next_actions)
            tq2 = critic2_target(next_states, next_actions)
            target_q = rewards + float(args.gamma) * (1.0 - dones) * torch.minimum(tq1, tq2)

        q1 = critic1(states, actions)
        q2 = critic2(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), args.max_grad_norm)
        critic_optimizer.step()
        critic_losses.append(float(critic_loss.item()))
        q_means.append(float(q1.mean().item()))
        gradient_update_index += 1

        if gradient_update_index % max(1, int(args.td3_policy_delay)) == 0:
            with torch.no_grad():
                before = actor(states).detach().clone()
            actor_loss = -critic1(states, actor(states)).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor_optimizer.step()
            with torch.no_grad():
                after = actor(states)
                deltas.append(float((after - before).abs().mean().item()))
            actor_losses.append(float(actor_loss.item()))
            actor_update_count += 1
            _soft_update(actor_target, actor, args.td3_tau)
            _soft_update(critic1_target, critic1, args.td3_tau)
            _soft_update(critic2_target, critic2, args.td3_tau)

    base_stats.update({
        "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
        "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
        "q1_mean": float(np.mean(q_means)) if q_means else 0.0,
        "policy_action_delta": float(np.mean(deltas)) if deltas else 1.0,
        "actor_update_count": float(actor_update_count),
        "replay_size": float(len(replay)),
    })
    return base_stats, gradient_update_index


def convergence_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "min_epochs": int(args.convergence_min_epochs),
        "window_updates": int(args.convergence_window_updates),
        "patience": int(args.convergence_patience),
        "reward_abs_tol": float(args.convergence_reward_abs_tol),
        "reward_rel_tol": float(args.convergence_reward_rel_tol),
        "policy_delta_threshold": float(args.convergence_policy_delta_threshold),
        "max_epochs": int(args.convergence_max_epochs),
    }


def _mean_metric(items: Sequence[Dict[str, Any]], key: str) -> float:
    values = [float(x.get(key, 0.0)) for x in items]
    return float(np.mean(values)) if values else 0.0


def assess_convergence(
    history: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    updates_per_epoch: int,
    previous_stable_checks: int,
) -> Dict[str, Any]:
    """TD3 收敛 = reward 平台且接近历史最佳 + deterministic Actor 动作基本不再变化。"""
    window = max(2, int(args.convergence_window_updates))
    min_updates = max(1, int(args.convergence_min_epochs)) * max(1, int(updates_per_epoch))
    enough = len(history) >= max(min_updates, 2 * window)
    result: Dict[str, Any] = {
        "enough_history": bool(enough), "stable": False, "converged": False,
        "stable_checks": 0 if not enough else int(previous_stable_checks),
        "recent_reward_mean": None, "previous_reward_mean": None,
        "reward_improvement": None, "reward_tolerance": None,
        "recent_policy_action_delta_mean": None, "best_window_reward_mean": None,
        "near_best": False, "reward_plateau": False, "policy_stable": False,
    }
    if not enough:
        return result
    recent = list(history[-window:])
    previous = list(history[-2 * window:-window])
    recent_reward = _mean_metric(recent, "mean_reward")
    previous_reward = _mean_metric(previous, "mean_reward")
    improvement = recent_reward - previous_reward
    tol = max(
        float(args.convergence_reward_abs_tol),
        float(args.convergence_reward_rel_tol) * max(abs(previous_reward), abs(recent_reward), 1e-3),
    )
    recent_delta = _mean_metric(recent, "policy_action_delta")
    rolling_means = [
        _mean_metric(history[end-window:end], "mean_reward")
        for end in range(window, len(history) + 1)
    ]
    best_window = max(rolling_means) if rolling_means else recent_reward
    near_best = recent_reward >= best_window - tol
    reward_plateau = abs(improvement) <= tol
    policy_stable = recent_delta <= float(args.convergence_policy_delta_threshold)
    stable = bool(reward_plateau and near_best and policy_stable)
    stable_checks = int(previous_stable_checks) + 1 if stable else 0
    converged = stable_checks >= max(1, int(args.convergence_patience))
    result.update({
        "stable": stable, "converged": converged, "stable_checks": stable_checks,
        "recent_reward_mean": recent_reward, "previous_reward_mean": previous_reward,
        "reward_improvement": improvement, "reward_tolerance": tol,
        "recent_policy_action_delta_mean": recent_delta,
        "best_window_reward_mean": best_window, "near_best": bool(near_best),
        "reward_plateau": bool(reward_plateau), "policy_stable": bool(policy_stable),
    })
    return result


def write_convergence_history_csv(state_dir: Path, history: Sequence[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for item in history:
        row = {k: v for k, v in item.items() if k != "convergence"}
        conv = item.get("convergence")
        if isinstance(conv, dict):
            for key, value in conv.items():
                row[f"conv_{key}"] = value
        rows.append(row)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    atomic_write_csv(state_dir / "convergence_history.csv", fields, rows)


def deterministic_epoch_order(total: int, seed: int, epoch_index: int, base_mode: str) -> List[int]:
    """每个 epoch 固定但不同的训练顺序，重启后能完全恢复同一顺序。"""
    salt = 0 if base_mode == "full" else 1000003
    rng = np.random.default_rng(int(seed) + salt + int(epoch_index) * 10007)
    return [int(x) for x in rng.permutation(total)]


# -----------------------------------------------------------------------------
# 七、公共运行函数
# -----------------------------------------------------------------------------
def make_sample_args(base: argparse.Namespace, row: Dict[str, Any]) -> argparse.Namespace:
    values = vars(base).copy()
    values.update({
        "sample_index": int(row["sample_index"]),
        "prompt": str(row["prompt"]),
        "negative_prompt": str(row["negative_prompt"]),
        "seed": int(row["generation_seed"]),
        # 父类 Token controller 读取这些字段。
        "image_token_cache_ratio": 1.0 - float(base.compute_ratio),
        "min_image_token_compute_ratio": float(base.min_compute_ratio),
        "max_token_cache_age": int(base.max_token_cache_age),
        "token_budget_mode": "per_block",
        "token_execution_mode": str(base.token_execution_mode),
        "token_policy_mode": "fixed",  # 旧 RL 永远关闭。
        "token_cache_edge_blocks": bool(base.token_cache_edge_blocks),
        "rl_policy_path": None,
        "rl_policy_device": base.policy_device,
        "rl_explore": False,
        "rl_temperature": 1.0,
        "rl_token_group_size": 16,
        "rl_hidden_dim": base.hidden_dim,
        "rl_transition_block_stride": 1,
    })
    return argparse.Namespace(**values)


def load_input_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def run_full_teacher(pipe, image: Image.Image, sample_args: argparse.Namespace, forwards_per_step: int):
    blocks = list(pipe.transformer.transformer_blocks)
    original_forward = pipe.transformer.forward
    controller = FullReferenceController(blocks, original_forward, sample_args, forwards_per_step)
    started = time.perf_counter()
    with replace_transformer_forward(pipe.transformer, controller):
        output = generate_image(pipe, [image], sample_args)
    elapsed = time.perf_counter() - started
    controller.validate_complete()
    return output, elapsed, controller.references


def run_blueprint_only(pipe, image, sample_args, forwards_per_step, schedule, teacher_refs):
    blocks = list(pipe.transformer.transformer_blocks)
    original_forward = pipe.transformer.forward
    controller = BlueLineScheduledController(
        blocks, original_forward, schedule, teacher_refs, sample_args, forwards_per_step
    )
    started = time.perf_counter()
    with replace_transformer_forward(pipe.transformer, controller):
        output = generate_image(pipe, [image], sample_args)
    elapsed = time.perf_counter() - started
    controller.validate_complete()
    return output, elapsed, controller


def run_token_method(
    pipe,
    image,
    sample_args,
    forwards_per_step,
    schedule,
    teacher_refs,
    *,
    budget_mode: str,
    policy_runtime: Optional[PolicyRuntime],
    fixed_score_map: Optional[Dict[Tuple[int, int], float]],
    base_is_blueprint: bool,
):
    blocks = list(pipe.transformer.transformer_blocks)
    original_forward = pipe.transformer.forward
    controller = BlockBudgetTokenController(
        transformer_blocks=blocks,
        original_transformer_forward=original_forward,
        schedule=schedule,
        teacher_references=teacher_refs,
        args=sample_args,
        forwards_per_step=forwards_per_step,
        budget_mode=budget_mode,
        target_compute_ratio=float(sample_args.compute_ratio),
        policy_runtime=policy_runtime,
        fixed_score_map=fixed_score_map,
        base_is_blueprint=base_is_blueprint,
    )
    started = time.perf_counter()
    with replace_transformer_forward(pipe.transformer, controller):
        output = generate_image(pipe, [image], sample_args)
    elapsed = time.perf_counter() - started
    controller.validate_complete()
    return output, elapsed, controller


def score_map(controller: BlockBudgetTokenController) -> Dict[Tuple[int, int], float]:
    return {
        (int(row["step_index_0based"]), int(row["branch_index_0based"])): float(row["score"])
        for row in controller.branch_step_rows
    }


def schedule_fingerprint(schedule: Dict[int, Dict[str, Any]]) -> str:
    compact = []
    for step in sorted(schedule):
        item = schedule[step]
        compact.append({
            "step": int(step),
            "executed": [int(v) for v in item.get("executed_blocks_0based", [])],
            "risk": [round(float(v), 10) for v in (item.get("smoothed_risk_by_block") or [])],
        })
    payload = json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_rows_fingerprint(rows: Optional[Sequence[Dict[str, Any]]]) -> str:
    if rows is None:
        return ""
    compact = [
        {
            "sample_index": int(r.get("sample_index", -1)),
            "image_path": str(r.get("image_path", "")),
            "prompt_id": str(r.get("prompt_id", "")),
            "prompt": str(r.get("prompt", "")),
            "generation_seed": int(r.get("generation_seed", -1)),
        }
        for r in rows
    ]
    text = json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rl_fingerprint(
    args: argparse.Namespace,
    base_mode: str,
    schedule: Dict[int, Dict[str, Any]],
    train_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """只描述 RL 语义。算法升级只失效 RL，不碰 Full Reference / Fixed25 静态缓存。"""
    return {
        "rl_algorithm_version": RL_ALGORITHM_VERSION,
        "base_mode": str(base_mode),
        "state_dim": td3_state_dim(args.td3_latent_dim),
        "latent_dim": int(args.td3_latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "compute_ratio": float(args.compute_ratio),
        "min_compute_ratio": float(args.min_compute_ratio),
        "num_inference_steps": int(args.num_inference_steps),
        "max_token_cache_age": int(args.max_token_cache_age),
        "token_execution_mode": str(args.token_execution_mode),
        "actor_lr": float(args.learning_rate),
        "critic_lr": float(args.td3_critic_learning_rate),
        "gamma": float(args.gamma),
        "tau": float(args.td3_tau),
        "policy_delay": int(args.td3_policy_delay),
        "target_noise": float(args.td3_target_noise),
        "target_noise_clip": float(args.td3_target_noise_clip),
        "allocation_logit_scale": float(args.td3_allocation_logit_scale),
        "noise_weight": float(args.noise_weight),
        "image_token_weight": float(args.image_token_weight),
        "text_token_weight": float(args.text_token_weight),
        "schedule_sha256": schedule_fingerprint(schedule),
        "train_manifest_sha256": manifest_rows_fingerprint(train_rows),
        "model_path": str(args.model_path), "dtype": str(args.dtype),
        "true_cfg_scale": float(args.true_cfg_scale), "guidance_scale": float(args.guidance_scale),
        "width": None if args.width is None else int(args.width),
        "height": None if args.height is None else int(args.height),
    }


def fingerprint_matches(saved: Any, expected: Dict[str, Any]) -> bool:
    return isinstance(saved, dict) and saved == expected


def policy_is_compatible(path: Path, expected: Dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu")
        return fingerprint_matches(payload.get("rl_fingerprint"), expected)
    except Exception:
        return False


def policy_is_converged_for_current_config(path: Path, expected: Dict[str, Any], args: argparse.Namespace) -> bool:
    if not policy_is_compatible(path, expected):
        return False
    try:
        payload = torch.load(path, map_location="cpu")
        return (
            bool(payload.get("training_converged", False))
            and payload.get("convergence_config") == convergence_config(args)
        )
    except Exception:
        return False


def checkpoint_is_compatible(path: Path, expected: Dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu")
        return fingerprint_matches(payload.get("rl_fingerprint"), expected)
    except Exception:
        return False


def _remove_if_exists(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except FileNotFoundError:
        pass


def invalidate_rl_training_only(output_dir: Path, base_mode: str, policy_path: Path) -> None:
    """算法/参数不兼容时只清 RL 产物；calibration/Blueprint/manifest/非RL图像全部保留。"""
    _remove_if_exists(policy_path)
    _remove_if_exists(output_dir / f"_train_state_{base_mode}")
    _remove_if_exists(output_dir / f"train_summary_{base_mode}.csv")
    train_root = output_dir / "train_samples"
    if train_root.exists():
        for sample_dir in train_root.glob("sample_*"):
            for name in (
                f"{base_mode}_rl25.png",
                f"decisions_{base_mode}.csv",
                f"record_{base_mode}.json",
                f"record_{base_mode}.csv",
            ):
                _remove_if_exists(sample_dir / name)


def ensure_rl_training_compatibility(
    output_dir: Path,
    base_mode: str,
    policy_path: Path,
    expected: Dict[str, Any],
) -> None:
    state_dir = output_dir / f"_train_state_{base_mode}"
    latest_ckpt = state_dir / "latest.pt"
    final_ok = policy_is_compatible(policy_path, expected)
    ckpt_ok = checkpoint_is_compatible(latest_ckpt, expected)
    has_old_rl = policy_path.exists() or latest_ckpt.exists() or state_dir.exists()
    if has_old_rl and not final_ok and not ckpt_ok:
        print(
            f"[rl-version:{base_mode}] 检测到旧/不兼容 RL 产物；只失效 RL 阶段。"
            f" Blueprint/calibration/manifest/固定基线结果继续复用。",
            flush=True,
        )
        invalidate_rl_training_only(output_dir, base_mode, policy_path)


def save_policy(
    path: Path,
    policy: TD3Actor,
    args: argparse.Namespace,
    base_mode: str,
    expected_fingerprint: Dict[str, Any],
    *,
    training_converged: bool = False,
    training_cursor: int = 0,
    update_index: int = 0,
    convergence_state: Optional[Dict[str, Any]] = None,
) -> None:
    atomic_torch_save(path, {
        "state_dict": policy.state_dict(),
        "hidden_dim": int(args.hidden_dim),
        "state_dim": td3_state_dim(args.td3_latent_dim),
        "latent_dim": int(args.td3_latent_dim),
        "compute_ratio": args.compute_ratio,
        "base_mode": base_mode,
        "rl_algorithm_version": RL_ALGORITHM_VERSION,
        "rl_fingerprint": expected_fingerprint,
        "training_converged": bool(training_converged),
        "training_cursor": int(training_cursor),
        "update_index": int(update_index),
        "convergence_config": convergence_config(args),
        "convergence_state": convergence_state or {},
    })


def load_policy(path: Path, args: argparse.Namespace) -> TD3Actor:
    payload = torch.load(path, map_location="cpu")
    if payload.get("rl_algorithm_version") != RL_ALGORITHM_VERSION:
        raise RuntimeError(
            f"Policy {path} 属于旧 RL 算法：{payload.get('rl_algorithm_version')}，当前需要 {RL_ALGORITHM_VERSION}。"
        )
    latent_dim = int(payload.get("latent_dim", args.td3_latent_dim))
    state_dim = int(payload.get("state_dim", td3_state_dim(latent_dim)))
    model = TD3Actor(state_dim=state_dim, latent_dim=latent_dim, hidden_dim=int(payload.get("hidden_dim", args.hidden_dim)))
    model.load_state_dict(payload["state_dict"])
    return model


# -----------------------------------------------------------------------------
# 八、训练：每个样本 Full Reference -> 同底座 Fixed25 -> 同底座 TD3-RL25 -> Replay/TD3
# -----------------------------------------------------------------------------

def _path_stat_signature(path_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    try:
        st = path.stat()
        return {
            "path": str(path.resolve()),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    except OSError:
        return {"path": str(path), "size": None, "mtime_ns": None}


def full_reference_cache_fingerprint(
    args: argparse.Namespace,
    row: Dict[str, Any],
    forwards_per_step: int,
) -> Dict[str, Any]:
    """Full Reference 只依赖完整推理语义，不依赖 RL / Fixed25 / Blueprint。"""
    return {
        "cache_version": TRAIN_STATIC_CACHE_VERSION,
        "kind": "full_reference",
        "model_path": str(Path(args.model_path).resolve()),
        "dtype": str(args.dtype),
        "num_inference_steps": int(args.num_inference_steps),
        "forwards_per_step": int(forwards_per_step),
        "true_cfg_scale": float(args.true_cfg_scale),
        "guidance_scale": float(args.guidance_scale),
        "width": None if args.width is None else int(args.width),
        "height": None if args.height is None else int(args.height),
        "image": _path_stat_signature(str(row["image_path"])),
        "prompt": str(row["prompt"]),
        "negative_prompt": str(row["negative_prompt"]),
        "generation_seed": int(row["generation_seed"]),
    }


def fixed25_cache_fingerprint(
    args: argparse.Namespace,
    base_mode: str,
    schedule: Dict[int, Dict[str, Any]],
    full_reference_fp: Dict[str, Any],
) -> Dict[str, Any]:
    """Fixed25 score 的静态依赖。RL 算法/Policy 改动不会让它失效。"""
    return {
        "cache_version": TRAIN_STATIC_CACHE_VERSION,
        "kind": "fixed25_score",
        "base_mode": str(base_mode),
        "full_reference_fingerprint": full_reference_fp,
        "schedule_sha256": schedule_fingerprint(schedule),
        "compute_ratio": float(args.compute_ratio),
        "min_compute_ratio": float(args.min_compute_ratio),
        "max_token_cache_age": int(args.max_token_cache_age),
        "token_execution_mode": str(args.token_execution_mode),
        "token_cache_edge_blocks": bool(args.token_cache_edge_blocks),
        "noise_weight": float(args.noise_weight),
        "image_token_weight": float(args.image_token_weight),
        "text_token_weight": float(args.text_token_weight),
    }


def _compact_teacher_references(
    refs: Dict[Tuple[int, int], Dict[str, Any]],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """只保存 score 真正会使用的 Full Reference。

    runtime 的 image last_tokens 可能包含“生成图 token + 参考图 token”；score_candidate 在
    比较时本来就只取生成图 token。这里落盘前提前裁掉后半部分，减少静态缓存磁盘占用，
    不改变 reward 数值语义。
    """
    compact: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for raw_key, item in refs.items():
        key = (int(raw_key[0]), int(raw_key[1]))
        sample = item["sample"].detach().cpu().contiguous()
        text, image = item["last_tokens"]
        text = text.detach().cpu().contiguous()
        image = image.detach().cpu()
        if sample.ndim == 3 and image.ndim == 3 and sample.shape[1] <= image.shape[1]:
            image = image[:, : int(sample.shape[1])]
        compact[key] = {
            "sample": sample,
            "last_tokens": (text, image.contiguous()),
        }
    return compact


def _tensor_tree_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_tree_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_nbytes(v) for v in value)
    return 0


def _load_static_cache(path: Path, expected_fp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[static-cache] 读取失败，将重建：{path} ({type(exc).__name__}: {exc})", flush=True)
        return None
    if payload.get("fingerprint") != expected_fp:
        return None
    return payload


def _train_static_cache_paths(output_dir: Path, sample_index: int, base_mode: str) -> Dict[str, Path]:
    root = output_dir / "train_static_cache" / f"sample_{sample_index:05d}"
    return {
        "dir": root,
        "full_reference": root / "full_reference.pt",
        "fixed25": root / f"{base_mode}_fixed25.pt",
    }


def _train_sample_record_paths(output_dir: Path, sample_index: int, base_mode: str) -> Dict[str, Path]:
    sample_dir = output_dir / "train_samples" / f"sample_{sample_index:05d}"
    return {
        "dir": sample_dir,
        "input": sample_dir / "input.png",
        "teacher": sample_dir / "full_teacher.png",
        "fixed": sample_dir / f"{base_mode}_fixed25.png",
        "rl": sample_dir / f"{base_mode}_rl25.png",
        "decisions": sample_dir / f"decisions_{base_mode}.csv",
        "record_json": sample_dir / f"record_{base_mode}.json",
        "record_csv": sample_dir / f"record_{base_mode}.csv",
    }


def _rebuild_train_summary(output_dir: Path, train_rows: Sequence[Dict[str, Any]], base_mode: str) -> None:
    records: List[Dict[str, Any]] = []
    for row in train_rows:
        paths = _train_sample_record_paths(output_dir, int(row["sample_index"]), base_mode)
        if paths["record_json"].is_file():
            try:
                records.append(json.loads(paths["record_json"].read_text(encoding="utf-8")))
            except Exception:
                pass
    if records:
        fields: List[str] = []
        for rec in records:
            for key in rec:
                if key not in fields:
                    fields.append(key)
        atomic_write_csv(output_dir / f"train_summary_{base_mode}.csv", fields, records)


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    dev = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(dev)


def train_one_base(
    *,
    base_mode: str,
    pipe,
    train_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    full_schedule: Dict[int, Dict[str, Any]],
    blueprint_schedule: Dict[int, Dict[str, Any]],
    policy_path: Path,
) -> None:
    """低维连续 TD3：固定训练池循环 rollout，直到自动收敛。

    Full Reference 与 Fixed25 沿用 v7.4 静态 cache；RL 算法升级只重建 TD3 policy/replay。
    """
    assert base_mode in {"full", "blueprint"}
    schedule = blueprint_schedule if base_mode == "blueprint" else full_schedule
    output_dir = policy_path.parent
    expected_fp = rl_fingerprint(args, base_mode, schedule, train_rows)
    if is_main_process():
        ensure_rl_training_compatibility(output_dir, base_mode, policy_path, expected_fp)
    dist_barrier()

    state_dim = td3_state_dim(args.td3_latent_dim)
    actor = TD3Actor(state_dim, args.td3_latent_dim, args.hidden_dim).to(args.policy_device)
    actor_target = TD3Actor(state_dim, args.td3_latent_dim, args.hidden_dim).to(args.policy_device)
    critic1 = TD3Critic(state_dim, args.td3_latent_dim, args.hidden_dim).to(args.policy_device)
    critic2 = TD3Critic(state_dim, args.td3_latent_dim, args.hidden_dim).to(args.policy_device)
    critic1_target = TD3Critic(state_dim, args.td3_latent_dim, args.hidden_dim).to(args.policy_device)
    critic2_target = TD3Critic(state_dim, args.td3_latent_dim, args.hidden_dim).to(args.policy_device)
    _hard_update(actor_target, actor)
    _hard_update(critic1_target, critic1)
    _hard_update(critic2_target, critic2)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    critic_optimizer = torch.optim.AdamW(
        list(critic1.parameters()) + list(critic2.parameters()),
        lr=args.td3_critic_learning_rate, weight_decay=1e-5,
    )
    forwards_per_step = int(args.forwards_per_step) if args.forwards_per_step is not None else (2 if args.true_cfg_scale > 1.0 else 1)

    state_dir = output_dir / f"_train_state_{base_mode}"
    state_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = state_dir / "latest.pt"
    latest_policy = state_dir / "latest_policy.pt"
    rollout_cache_dir = state_dir / "rollout_cache"
    rollout_cache_dir.mkdir(parents=True, exist_ok=True)
    history_path = state_dir / "convergence_history.json"

    total = len(train_rows)
    if total <= 0:
        raise RuntimeError(f"train:{base_mode} 训练池为空。")
    global_chunk = max(1, int(args.update_every_samples))
    updates_per_epoch = int(math.ceil(total / global_chunk))
    training_cursor = 0
    update_index = 0
    gradient_update_index = 0
    convergence_history: List[Dict[str, Any]] = []
    stable_checks = 0
    replay: List[Transition] = []

    if checkpoint_is_compatible(latest_ckpt, expected_fp):
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        actor.load_state_dict(ckpt["actor_state"])
        actor_target.load_state_dict(ckpt["actor_target_state"])
        critic1.load_state_dict(ckpt["critic1_state"])
        critic2.load_state_dict(ckpt["critic2_state"])
        critic1_target.load_state_dict(ckpt["critic1_target_state"])
        critic2_target.load_state_dict(ckpt["critic2_target_state"])
        actor_optimizer.load_state_dict(ckpt["actor_optimizer_state"])
        critic_optimizer.load_state_dict(ckpt["critic_optimizer_state"])
        _optimizer_state_to_device(actor_optimizer, args.policy_device)
        _optimizer_state_to_device(critic_optimizer, args.policy_device)
        replay = [deserialize_transition(x) for x in ckpt.get("replay", [])]
        training_cursor = int(ckpt.get("training_cursor", 0))
        update_index = int(ckpt.get("update_index", 0))
        gradient_update_index = int(ckpt.get("gradient_update_index", 0))
        if ckpt.get("convergence_config") == convergence_config(args):
            convergence_history = list(ckpt.get("convergence_history", []))
            stable_checks = int(ckpt.get("stable_checks", 0))
        if bool(ckpt.get("training_converged", False)) and ckpt.get("convergence_config") == convergence_config(args):
            if is_main_process():
                save_policy(policy_path, actor, args, base_mode, expected_fp, training_converged=True,
                            training_cursor=training_cursor, update_index=update_index,
                            convergence_state=ckpt.get("convergence_state", {}))
                print(f"[train:{base_mode}] TD3 checkpoint 已收敛，直接复用。", flush=True)
            dist_barrier()
            return
        print(
            f"[train:{base_mode}] TD3 强制断点续跑 cursor={training_cursor} epoch={training_cursor/total:.2f} "
            f"update={update_index} replay={len(replay)}",
            flush=True,
        )
    elif policy_is_compatible(policy_path, expected_fp):
        payload = torch.load(policy_path, map_location="cpu")
        actor.load_state_dict(payload["state_dict"])
        _hard_update(actor_target, actor)
        if policy_is_converged_for_current_config(policy_path, expected_fp, args):
            if is_main_process():
                print(f"[train:{base_mode}] compatible TD3 policy 已收敛，直接复用。", flush=True)
            dist_barrier()
            return

    broadcast_model_from_rank0(actor)
    if is_main_process():
        print(
            f"[train:{base_mode}] TD3 latent_dim={args.td3_latent_dim}, state_dim={state_dim}, "
            f"train_pool={total}, replay_capacity={args.td3_replay_capacity}, "
            f"static_cache={'ON' if args.cache_train_static else 'OFF'}；训练终点=自动收敛。",
            flush=True,
        )

    converged = False
    final_convergence_state: Dict[str, Any] = {}
    while not converged:
        epoch_index = training_cursor // total
        position_in_epoch = training_cursor % total
        if int(args.convergence_max_epochs) > 0 and epoch_index >= int(args.convergence_max_epochs):
            if is_main_process():
                print(f"[train:{base_mode}] 达到安全上限 {args.convergence_max_epochs} epoch，尚未收敛。", flush=True)
                save_policy(latest_policy, actor, args, base_mode, expected_fp, training_converged=False,
                            training_cursor=training_cursor, update_index=update_index,
                            convergence_state=final_convergence_state)
            dist_barrier()
            return

        order = deterministic_epoch_order(total, args.seed, epoch_index, base_mode)
        chunk_len = min(global_chunk, total - position_in_epoch)
        positions = list(range(position_in_epoch, position_in_epoch + chunk_len))
        occurrence_ids = [training_cursor + j for j in range(chunk_len)]
        dataset_indices = [order[pos] for pos in positions]
        chunk_dir = rollout_cache_dir / f"update_{update_index:06d}_cursor_{training_cursor:09d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        tasks = list(zip(occurrence_ids, dataset_indices, positions))
        pending_tasks = [t for t in tasks if not (chunk_dir / f"occ_{t[0]:09d}.pt").is_file()]
        local_tasks = [t for j, t in enumerate(pending_tasks) if j % get_world_size() == get_rank()]

        exploration_noise = max(
            float(args.td3_exploration_noise_min),
            float(args.td3_exploration_noise) * (float(args.td3_exploration_noise_decay) ** update_index),
        )
        if is_main_process():
            print(
                f"[train:{base_mode}:resume] epoch={epoch_index+1} pos={position_in_epoch}:{position_in_epoch+chunk_len}/{total} "
                f"update={update_index} noise={exploration_noise:.5f} 已有={len(tasks)-len(pending_tasks)} 待跑={len(pending_tasks)}",
                flush=True,
            )

        for occurrence_id, dataset_index, epoch_position in local_tasks:
            row = train_rows[dataset_index]
            sample_index = int(row["sample_index"])
            sample_args = make_sample_args(args, row)
            image = load_input_image(row["image_path"])
            paths = _train_sample_record_paths(output_dir, sample_index, base_mode)
            paths["dir"].mkdir(parents=True, exist_ok=True)
            print(
                f"\n[train:{base_mode}][rank{get_rank()}] epoch={epoch_index+1} "
                f"pos={epoch_position+1}/{total} sample_id={sample_index}", flush=True,
            )
            if not paths["input"].is_file():
                save_image_atomic(image, paths["input"])

            static_paths = _train_static_cache_paths(output_dir, sample_index, base_mode)
            static_paths["dir"].mkdir(parents=True, exist_ok=True)
            full_ref_fp = full_reference_cache_fingerprint(args, row, forwards_per_step)
            full_payload = _load_static_cache(static_paths["full_reference"], full_ref_fp) if args.cache_train_static else None
            full_cache_hit = full_payload is not None and paths["teacher"].is_file()
            teacher_time = teacher_first_compute_time = 0.0
            if full_cache_hit:
                teacher_refs = full_payload["teacher_refs"]
                teacher_first_compute_time = float(full_payload.get("first_compute_elapsed", 0.0))
                teacher_img = _load_png(paths["teacher"])
                print(f"[static-cache][rank{get_rank()}] sample={sample_index:05d} Full Reference HIT", flush=True)
            else:
                teacher_img, teacher_time, raw_teacher_refs = run_full_teacher(pipe, image, sample_args, forwards_per_step)
                teacher_first_compute_time = float(teacher_time)
                if not paths["teacher"].is_file(): save_image_atomic(teacher_img, paths["teacher"])
                teacher_refs = _compact_teacher_references(raw_teacher_refs)
                release_teacher_references(raw_teacher_refs); del raw_teacher_refs
                if args.cache_train_static:
                    cache_bytes = _tensor_tree_nbytes(teacher_refs)
                    atomic_torch_save(static_paths["full_reference"], {
                        "fingerprint": full_ref_fp, "teacher_refs": teacher_refs,
                        "first_compute_elapsed": float(teacher_time), "tensor_bytes": int(cache_bytes),
                    })
                    print(f"[static-cache][rank{get_rank()}] sample={sample_index:05d} Full Reference MISS→缓存 {cache_bytes/1024**3:.3f} GiB", flush=True)

            fixed_fp = fixed25_cache_fingerprint(args, base_mode, schedule, full_ref_fp)
            fixed_payload = _load_static_cache(static_paths["fixed25"], fixed_fp) if args.cache_train_static else None
            fixed_cache_hit = fixed_payload is not None and paths["fixed"].is_file()
            fixed_time = fixed_first_compute_time = 0.0
            if fixed_cache_hit:
                fixed_scores = {(int(k[0]), int(k[1])): float(v) for k, v in fixed_payload["fixed_scores"].items()}
                fixed_metric = dict(fixed_payload.get("fixed_metric", {}))
                fixed_first_compute_time = float(fixed_payload.get("first_compute_elapsed", 0.0))
                print(f"[static-cache][rank{get_rank()}] sample={sample_index:05d} {base_mode} Fixed25 HIT", flush=True)
            else:
                fixed_img, fixed_time, fixed_controller = run_token_method(
                    pipe, image, sample_args, forwards_per_step, schedule, teacher_refs,
                    budget_mode="fixed25", policy_runtime=None, fixed_score_map=None,
                    base_is_blueprint=(base_mode == "blueprint"),
                )
                if not paths["fixed"].is_file(): save_image_atomic(fixed_img, paths["fixed"])
                fixed_metric = metric_row(teacher_img, fixed_img)
                fixed_scores = score_map(fixed_controller)
                fixed_first_compute_time = float(fixed_time)
                release_controller_cuda_state(fixed_controller); del fixed_controller, fixed_img
                if args.cache_train_static:
                    atomic_torch_save(static_paths["fixed25"], {
                        "fingerprint": fixed_fp, "fixed_scores": fixed_scores,
                        "fixed_metric": fixed_metric, "first_compute_elapsed": float(fixed_time),
                    })

            runtime = PolicyRuntime(
                actor, args.policy_device, explore=True,
                exploration_noise=exploration_noise,
                seed=int(args.seed) + int(occurrence_id) * 1009,
            )
            rl_img, rl_time, rl_controller = run_token_method(
                pipe, image, sample_args, forwards_per_step, schedule, teacher_refs,
                budget_mode="rl25", policy_runtime=runtime, fixed_score_map=fixed_scores,
                base_is_blueprint=(base_mode == "blueprint"),
            )
            # 若最后几步被强制 full 而没有新的 TD3 action，也要结束每个 branch 的最后 transition。
            for branch in range(forwards_per_step):
                runtime.finish_branch(branch)
            save_image_atomic(rl_img, paths["rl"])
            rl_metric = metric_row(teacher_img, rl_img)
            rewards = [float(t.reward) for t in runtime.transitions]
            all_alloc = _flatten_alloc(runtime.transitions)
            mean_reward = float(np.mean(rewards)) if rewards else 0.0
            mean_alloc = float(all_alloc.mean()) if all_alloc.size else 0.0
            std_alloc = float(all_alloc.std()) if all_alloc.size else 0.0
            write_transition_table(paths["decisions"], runtime.transitions)
            record = {
                "sample_index": sample_index, "dataset_train_index": int(dataset_index),
                "training_occurrence_id": int(occurrence_id), "training_epoch": int(epoch_index),
                "epoch_position": int(epoch_position), "td3_update_index": int(update_index),
                "base_mode": base_mode, "prompt_id": str(row["prompt_id"]),
                "image_path": str(row["image_path"]), "generation_seed": int(row["generation_seed"]),
                "rl_algorithm_version": RL_ALGORITHM_VERSION,
                "transition_count": len(runtime.transitions), "step_reward_count": len(rewards),
                "mean_reward": mean_reward, "mean_allocated_compute_ratio": mean_alloc,
                "std_allocated_compute_ratio": std_alloc,
                "exploration_noise": float(exploration_noise),
                "full_reference_cache_hit": bool(full_cache_hit), "fixed25_cache_hit": bool(fixed_cache_hit),
                "teacher_elapsed": float(teacher_time), "teacher_first_compute_elapsed": float(teacher_first_compute_time),
                "fixed25_elapsed": float(fixed_time), "fixed25_first_compute_elapsed": float(fixed_first_compute_time),
                "rl25_elapsed": float(rl_time),
                "fixed25_psnr_vs_teacher": float(fixed_metric["psnr"]), "fixed25_ssim_vs_teacher": float(fixed_metric["ssim"]),
                "rl25_psnr_vs_teacher": float(rl_metric["psnr"]), "rl25_ssim_vs_teacher": float(rl_metric["ssim"]),
                "rank": get_rank(),
            }
            atomic_write_json(paths["record_json"], record)
            atomic_write_csv(paths["record_csv"], list(record.keys()), [record])
            rollout_file = chunk_dir / f"occ_{occurrence_id:09d}.pt"
            atomic_torch_save(rollout_file, {
                "rl_algorithm_version": RL_ALGORITHM_VERSION, "rl_fingerprint": expected_fp,
                "record": record, "transitions": [serialize_transition(t) for t in runtime.transitions],
            })
            release_controller_cuda_state(rl_controller); del rl_controller, fixed_scores
            release_teacher_references(teacher_refs); del teacher_refs, teacher_img, rl_img, image, runtime
            full_payload = fixed_payload = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        dist_barrier()
        if is_main_process():
            missing = [occ for occ, _, _ in tasks if not (chunk_dir / f"occ_{occ:09d}.pt").is_file()]
            if missing: raise RuntimeError(f"TD3 update 缺 rollout occurrence：{missing}")
            gathered: List[Transition] = []
            for occurrence_id, _, _ in tasks:
                payload = torch.load(chunk_dir / f"occ_{occurrence_id:09d}.pt", map_location="cpu")
                if not fingerprint_matches(payload.get("rl_fingerprint"), expected_fp):
                    raise RuntimeError("发现不兼容 TD3 rollout")
                gathered.extend(deserialize_transition(x) for x in payload.get("transitions", []))
            replay.extend(gathered)
            cap = max(1, int(args.td3_replay_capacity))
            if len(replay) > cap:
                replay = replay[-cap:]
            stats, gradient_update_index = td3_update(
                actor=actor, actor_target=actor_target, critic1=critic1, critic2=critic2,
                critic1_target=critic1_target, critic2_target=critic2_target,
                actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                replay=replay, rollout_transitions=gathered, args=args,
                gradient_update_index=gradient_update_index,
            )
            stats["exploration_noise"] = float(exploration_noise)
            training_cursor += chunk_len
            update_index += 1
            stats_record = {
                "update_index": int(update_index), "gradient_update_index": int(gradient_update_index),
                "training_cursor": int(training_cursor), "epoch_index": int((training_cursor-1)//total),
                "epoch_fraction": float(training_cursor/total), "sample_occurrences": int(chunk_len), **stats,
            }
            convergence_history.append(stats_record)
            convergence_state = assess_convergence(convergence_history, args, updates_per_epoch, stable_checks)
            stable_checks = int(convergence_state["stable_checks"])
            final_convergence_state = convergence_state
            stats_record["convergence"] = dict(convergence_state)
            atomic_write_json(state_dir / f"td3_update_{update_index:06d}.json", {
                "rl_algorithm_version": RL_ALGORITHM_VERSION, **stats_record,
            })
            atomic_write_json(history_path, {
                "base_mode": base_mode, "convergence_config": convergence_config(args),
                "updates_per_epoch": updates_per_epoch, "stable_checks": stable_checks,
                "converged": bool(convergence_state["converged"]), "history": convergence_history,
            })
            write_convergence_history_csv(state_dir, convergence_history)
            atomic_torch_save(latest_ckpt, {
                "actor_state": actor.state_dict(), "actor_target_state": actor_target.state_dict(),
                "critic1_state": critic1.state_dict(), "critic2_state": critic2.state_dict(),
                "critic1_target_state": critic1_target.state_dict(), "critic2_target_state": critic2_target.state_dict(),
                "actor_optimizer_state": actor_optimizer.state_dict(), "critic_optimizer_state": critic_optimizer.state_dict(),
                "replay": [serialize_transition(t) for t in replay],
                "training_cursor": int(training_cursor), "update_index": int(update_index),
                "gradient_update_index": int(gradient_update_index), "base_mode": base_mode,
                "rl_algorithm_version": RL_ALGORITHM_VERSION, "rl_fingerprint": expected_fp,
                "training_converged": bool(convergence_state["converged"]),
                "convergence_config": convergence_config(args), "convergence_history": convergence_history,
                "stable_checks": stable_checks, "convergence_state": convergence_state,
            })
            save_policy(latest_policy, actor, args, base_mode, expected_fp,
                        training_converged=bool(convergence_state["converged"]),
                        training_cursor=training_cursor, update_index=update_index,
                        convergence_state=convergence_state)
            print(
                f"[TD3:{base_mode}] update={update_index} epoch={training_cursor/total:.2f} "
                f"critic_loss={stats['critic_loss']:.6f} actor_loss={stats['actor_loss']:+.6f} "
                f"Q1={stats['q1_mean']:+.6f} replay={int(stats['replay_size'])} "
                f"actor_updates={int(stats['actor_update_count'])} noise={exploration_noise:.5f} "
                f"policy_delta={stats['policy_action_delta']:.7f}", flush=True,
            )
            print(
                f"[convergence:{base_mode}] batch_reward={stats['mean_reward']:+.6f}±{stats['std_reward']:.6f} "
                f"recent={convergence_state.get('recent_reward_mean')} previous={convergence_state.get('previous_reward_mean')} "
                f"improve={convergence_state.get('reward_improvement')} tol={convergence_state.get('reward_tolerance')} "
                f"best={convergence_state.get('best_window_reward_mean')} near_best={convergence_state.get('near_best')} "
                f"plateau={convergence_state.get('reward_plateau')} delta_now={stats['policy_action_delta']:.7f} "
                f"delta_recent={convergence_state.get('recent_policy_action_delta_mean')} "
                f"policy_stable={convergence_state.get('policy_stable')} "
                f"alloc_mean={stats['mean_allocated_compute_ratio']:.6f} alloc_std={stats['std_allocated_compute_ratio']:.6f} "
                f"stable={convergence_state['stable']} patience={stable_checks}/{args.convergence_patience} "
                f"converged={convergence_state['converged']}", flush=True,
            )
            shutil.rmtree(chunk_dir, ignore_errors=True)
            converged = bool(convergence_state["converged"])
            if converged:
                save_policy(policy_path, actor, args, base_mode, expected_fp, training_converged=True,
                            training_cursor=training_cursor, update_index=update_index,
                            convergence_state=convergence_state)
                _rebuild_train_summary(output_dir, train_rows, base_mode)
                print(
                    f"[train:{base_mode}] TD3 自动收敛：epoch={training_cursor/total:.2f}, updates={update_index}, "
                    f"recent_reward={convergence_state.get('recent_reward_mean')}, "
                    f"recent_policy_delta={convergence_state.get('recent_policy_action_delta_mean')}", flush=True,
                )
        dist_barrier()
        sync_ckpt = torch.load(latest_ckpt, map_location="cpu")
        actor.load_state_dict(sync_ckpt["actor_state"])
        actor.to(args.policy_device)
        training_cursor = int(sync_ckpt.get("training_cursor", training_cursor))
        update_index = int(sync_ckpt.get("update_index", update_index))
        stable_checks = int(sync_ckpt.get("stable_checks", stable_checks))
        converged = bool(sync_ckpt.get("training_converged", False))
        del sync_ckpt
        gc.collect()
    dist_barrier()


# -----------------------------------------------------------------------------
# 九、六组评估与汇总
# -----------------------------------------------------------------------------
def metric_row(reference: Image.Image, candidate: Image.Image) -> Dict[str, float]:
    result = image_metrics(reference, candidate, compute_lpips=False)
    return {"psnr": float(result["psnr"]), "ssim": float(result["ssim"])}


def fingerprint_digest(fp: Dict[str, Any]) -> str:
    text = json.dumps(fp, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_eval_record(sample_dir: Path) -> Dict[str, Any]:
    path = sample_dir / "record.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _eval_nonrl_ready(sample_dir: Path, rec: Dict[str, Any]) -> bool:
    required_files = [
        "input.png", "full_dense.png", "blueprint_only.png",
        "blueprint_fixed25.png", "full_fixed25.png",
    ]
    required_keys = [
        "full_dense_elapsed",
        "blueprint_only_elapsed", "blueprint_only_psnr", "blueprint_only_ssim",
        "blueprint_fixed25_elapsed", "blueprint_fixed25_psnr", "blueprint_fixed25_ssim",
        "full_fixed25_elapsed", "full_fixed25_psnr", "full_fixed25_ssim",
    ]
    return all((sample_dir / name).is_file() for name in required_files) and all(k in rec for k in required_keys)


def _eval_bp_rl_ready(sample_dir: Path, rec: Dict[str, Any], bp_fp_digest: str) -> bool:
    return (
        (sample_dir / "blueprint_rl25.png").is_file()
        and (sample_dir / "decisions_blueprint_rl25.csv").is_file()
        and rec.get("rl_algorithm_version") == RL_ALGORITHM_VERSION
        and rec.get("blueprint_rl_fingerprint_sha256") == bp_fp_digest
        and all(k in rec for k in ("blueprint_rl25_elapsed", "blueprint_rl25_psnr", "blueprint_rl25_ssim"))
    )


def _eval_full_rl_ready(sample_dir: Path, rec: Dict[str, Any], full_fp_digest: str) -> bool:
    return (
        (sample_dir / "full_rl25.png").is_file()
        and (sample_dir / "decisions_full_rl25.csv").is_file()
        and rec.get("rl_algorithm_version") == RL_ALGORITHM_VERSION
        and rec.get("full_rl_fingerprint_sha256") == full_fp_digest
        and all(k in rec for k in ("full_rl25_elapsed", "full_rl25_psnr", "full_rl25_ssim"))
    )


def _eval_sample_complete(
    sample_dir: Path,
    full_fp_digest: str,
    bp_fp_digest: str,
) -> bool:
    rec = _load_eval_record(sample_dir)
    return (
        _eval_nonrl_ready(sample_dir, rec)
        and _eval_bp_rl_ready(sample_dir, rec, bp_fp_digest)
        and _eval_full_rl_ready(sample_dir, rec, full_fp_digest)
        and (sample_dir / "metrics.csv").is_file()
        and (sample_dir / "DONE").is_file()
    )


def _load_png(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def evaluate_sixway(
    *,
    pipe,
    eval_rows: Sequence[Dict[str, Any]],
    train_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    full_schedule: Dict[int, Dict[str, Any]],
    blueprint_schedule: Dict[int, Dict[str, Any]],
    full_policy_path: Path,
    blueprint_policy_path: Path,
    output_dir: Path,
) -> None:
    full_fp = rl_fingerprint(args, "full", full_schedule, train_rows)
    bp_fp = rl_fingerprint(args, "blueprint", blueprint_schedule, train_rows)
    full_fp_digest = fingerprint_digest(full_fp)
    bp_fp_digest = fingerprint_digest(bp_fp)
    if not policy_is_compatible(full_policy_path, full_fp) or not policy_is_compatible(blueprint_policy_path, bp_fp):
        raise FileNotFoundError(
            "eval 需要与当前算法、schedule、manifest和训练配置兼容的 full/blueprint 两套 policy；若刚完成训练仍看到此错误，请检查 policy fingerprint。"
        )

    full_policy = load_policy(full_policy_path, args).to(args.policy_device).eval()
    blueprint_policy = load_policy(blueprint_policy_path, args).to(args.policy_device).eval()
    forwards_per_step = int(args.forwards_per_step) if args.forwards_per_step is not None else (2 if args.true_cfg_scale > 1.0 else 1)

    methods = [
        "blueprint_only", "full_dense", "blueprint_fixed25",
        "blueprint_rl25", "full_fixed25", "full_rl25",
    ]
    eval_root = output_dir / "eval_samples"
    eval_root.mkdir(parents=True, exist_ok=True)

    pending: List[Tuple[int, Dict[str, Any]]] = []
    for eval_index, row in enumerate(eval_rows):
        sample_dir = eval_root / f"sample_{int(row['sample_index']):05d}"
        if _eval_sample_complete(sample_dir, full_fp_digest, bp_fp_digest):
            continue
        pending.append((eval_index, row))
    local_pending = [item for i, item in enumerate(pending) if i % get_world_size() == get_rank()]
    if is_main_process():
        print(
            f"[eval-resume] target={len(eval_rows)}, compatible_done={len(eval_rows)-len(pending)}, pending={len(pending)}",
            flush=True,
        )

    for eval_index, row in local_pending:
        sample_index = int(row["sample_index"])
        sample_args = make_sample_args(args, row)
        image = load_input_image(row["image_path"])
        sample_dir = eval_root / f"sample_{sample_index:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        rec = _load_eval_record(sample_dir)
        nonrl_ready = _eval_nonrl_ready(sample_dir, rec)
        bp_rl_ready = _eval_bp_rl_ready(sample_dir, rec, bp_fp_digest)
        full_rl_ready = _eval_full_rl_ready(sample_dir, rec, full_fp_digest)

        print(
            f"\n[eval][rank{get_rank()}] sample {eval_index + 1}/{len(eval_rows)} id={sample_index} "
            f"reuse_nonrl={int(nonrl_ready)} reuse_bp_rl={int(bp_rl_ready)} reuse_full_rl={int(full_rl_ready)}",
            flush=True,
        )
        if not (sample_dir / "input.png").is_file():
            save_image_atomic(image, sample_dir / "input.png")

        # 任意 RL 方法要重算时都需要当前 full teacher reference。若 full_dense 已存在，
        # 只把新 forward 当作 reference 生产器，不覆盖旧图/旧计时。
        generated_full, generated_full_time, teacher_refs = run_full_teacher(
            pipe, image, sample_args, forwards_per_step
        )
        if (sample_dir / "full_dense.png").is_file():
            full_img = _load_png(sample_dir / "full_dense.png")
        else:
            full_img = generated_full
            save_image_atomic(full_img, sample_dir / "full_dense.png")
            rec["full_dense_elapsed"] = float(generated_full_time)
        if "full_dense_elapsed" not in rec:
            rec["full_dense_elapsed"] = float(generated_full_time)

        # 非 RL 四组只在缺文件/缺指标时才补；v6 已有结果直接复用。
        if not (sample_dir / "blueprint_only.png").is_file() or not all(
            k in rec for k in ("blueprint_only_elapsed", "blueprint_only_psnr", "blueprint_only_ssim")
        ):
            bp_img, bp_time, bp_ctrl = run_blueprint_only(
                pipe, image, sample_args, forwards_per_step, blueprint_schedule, teacher_refs
            )
            save_image_atomic(bp_img, sample_dir / "blueprint_only.png")
            bp_metric = metric_row(full_img, bp_img)
            rec.update({
                "blueprint_only_elapsed": float(bp_time),
                "blueprint_only_psnr": float(bp_metric["psnr"]),
                "blueprint_only_ssim": float(bp_metric["ssim"]),
            })
            release_controller_cuda_state(bp_ctrl)
            del bp_ctrl, bp_img

        if not (sample_dir / "blueprint_fixed25.png").is_file() or not all(
            k in rec for k in ("blueprint_fixed25_elapsed", "blueprint_fixed25_psnr", "blueprint_fixed25_ssim")
        ):
            bpf_img, bpf_time, bpf_ctrl = run_token_method(
                pipe, image, sample_args, forwards_per_step, blueprint_schedule, teacher_refs,
                budget_mode="fixed25", policy_runtime=None, fixed_score_map=None, base_is_blueprint=True,
            )
            save_image_atomic(bpf_img, sample_dir / "blueprint_fixed25.png")
            bpf_metric = metric_row(full_img, bpf_img)
            rec.update({
                "blueprint_fixed25_elapsed": float(bpf_time),
                "blueprint_fixed25_psnr": float(bpf_metric["psnr"]),
                "blueprint_fixed25_ssim": float(bpf_metric["ssim"]),
            })
            release_controller_cuda_state(bpf_ctrl)
            del bpf_ctrl, bpf_img

        if not (sample_dir / "full_fixed25.png").is_file() or not all(
            k in rec for k in ("full_fixed25_elapsed", "full_fixed25_psnr", "full_fixed25_ssim")
        ):
            ff_img, ff_time, ff_ctrl = run_token_method(
                pipe, image, sample_args, forwards_per_step, full_schedule, teacher_refs,
                budget_mode="fixed25", policy_runtime=None, fixed_score_map=None, base_is_blueprint=False,
            )
            save_image_atomic(ff_img, sample_dir / "full_fixed25.png")
            ff_metric = metric_row(full_img, ff_img)
            rec.update({
                "full_fixed25_elapsed": float(ff_time),
                "full_fixed25_psnr": float(ff_metric["psnr"]),
                "full_fixed25_ssim": float(ff_metric["ssim"]),
            })
            release_controller_cuda_state(ff_ctrl)
            del ff_ctrl, ff_img

        # 只有 RL policy/fingerprint 不兼容时才重算对应 RL 图。
        if not bp_rl_ready:
            bp_runtime = PolicyRuntime(blueprint_policy, args.policy_device, explore=False, exploration_noise=0.0, seed=args.seed)
            bpr_img, bpr_time, bpr_ctrl = run_token_method(
                pipe, image, sample_args, forwards_per_step, blueprint_schedule, teacher_refs,
                budget_mode="rl25", policy_runtime=bp_runtime, fixed_score_map=None, base_is_blueprint=True,
            )
            save_image_atomic(bpr_img, sample_dir / "blueprint_rl25.png")
            bpr_metric = metric_row(full_img, bpr_img)
            write_transition_table(sample_dir / "decisions_blueprint_rl25.csv", bp_runtime.transitions)
            alloc = _flatten_alloc(bp_runtime.transitions)
            rec.update({
                "blueprint_rl25_elapsed": float(bpr_time),
                "blueprint_rl25_psnr": float(bpr_metric["psnr"]),
                "blueprint_rl25_ssim": float(bpr_metric["ssim"]),
                "blueprint_rl25_mean_block_compute_ratio": float(np.mean(alloc)) if alloc.size else 0.0,
                "blueprint_rl25_std_block_compute_ratio": float(np.std(alloc)) if alloc.size else 0.0,
            })
            release_controller_cuda_state(bpr_ctrl)
            del bpr_ctrl, bpr_img, bp_runtime

        if not full_rl_ready:
            full_runtime = PolicyRuntime(full_policy, args.policy_device, explore=False, exploration_noise=0.0, seed=args.seed)
            fr_img, fr_time, fr_ctrl = run_token_method(
                pipe, image, sample_args, forwards_per_step, full_schedule, teacher_refs,
                budget_mode="rl25", policy_runtime=full_runtime, fixed_score_map=None, base_is_blueprint=False,
            )
            save_image_atomic(fr_img, sample_dir / "full_rl25.png")
            fr_metric = metric_row(full_img, fr_img)
            write_transition_table(sample_dir / "decisions_full_rl25.csv", full_runtime.transitions)
            alloc = _flatten_alloc(full_runtime.transitions)
            rec.update({
                "full_rl25_elapsed": float(fr_time),
                "full_rl25_psnr": float(fr_metric["psnr"]),
                "full_rl25_ssim": float(fr_metric["ssim"]),
                "full_rl25_mean_block_compute_ratio": float(np.mean(alloc)) if alloc.size else 0.0,
                "full_rl25_std_block_compute_ratio": float(np.std(alloc)) if alloc.size else 0.0,
            })
            release_controller_cuda_state(fr_ctrl)
            del fr_ctrl, fr_img, full_runtime

        rec.update({
            "sample_index": sample_index,
            "prompt_id": str(row["prompt_id"]),
            "image_path": str(row["image_path"]),
            "generation_seed": int(row["generation_seed"]),
            "rl_algorithm_version": RL_ALGORITHM_VERSION,
            "full_rl_fingerprint_sha256": full_fp_digest,
            "blueprint_rl_fingerprint_sha256": bp_fp_digest,
        })
        atomic_write_json(sample_dir / "record.json", rec)
        atomic_write_csv(sample_dir / "metrics.csv", list(rec.keys()), [rec])
        touch_done(sample_dir / "DONE")

        release_teacher_references(teacher_refs)
        del teacher_refs, image, generated_full, full_img
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_cuda_memory("eval:sample_done", args.device)

    dist_barrier()

    if is_main_process():
        rows_out: List[Dict[str, Any]] = []
        sums: Dict[str, Dict[str, float]] = {
            m: {"elapsed": 0.0, "psnr": 0.0, "ssim": 0.0, "count": 0.0} for m in methods
        }
        for row in eval_rows:
            sample_dir = eval_root / f"sample_{int(row['sample_index']):05d}"
            if not _eval_sample_complete(sample_dir, full_fp_digest, bp_fp_digest):
                raise RuntimeError(f"eval 样本仍未完成或 RL fingerprint 不兼容：{sample_dir}")
            rec = _load_eval_record(sample_dir)
            rows_out.append(rec)
            sums["full_dense"]["elapsed"] += float(rec["full_dense_elapsed"])
            sums["full_dense"]["count"] += 1
            for method in [m for m in methods if m != "full_dense"]:
                sums[method]["elapsed"] += float(rec[f"{method}_elapsed"])
                sums[method]["psnr"] += float(rec[f"{method}_psnr"])
                sums[method]["ssim"] += float(rec[f"{method}_ssim"])
                sums[method]["count"] += 1

        full_mean_time = sums["full_dense"]["elapsed"] / max(1.0, sums["full_dense"]["count"])
        summary: List[Dict[str, Any]] = []
        for method in methods:
            count = int(sums[method]["count"])
            mean_time = sums[method]["elapsed"] / max(1, count)
            if method == "full_dense":
                mean_psnr, mean_ssim = float("inf"), 1.0
            else:
                mean_psnr = sums[method]["psnr"] / max(1, count)
                mean_ssim = sums[method]["ssim"] / max(1, count)
            summary.append({
                "method": method,
                "completed": count,
                "mean_elapsed_seconds": mean_time,
                "measured_speedup_vs_full_dense": full_mean_time / max(mean_time, 1e-9),
                "psnr_vs_full_dense": mean_psnr,
                "ssim_vs_full_dense": mean_ssim,
                "compute_ratio": 1.0 if method in {"full_dense", "blueprint_only"} else args.compute_ratio,
                "rl_algorithm_version": RL_ALGORITHM_VERSION if "rl25" in method else "",
            })

        if rows_out:
            fields: List[str] = []
            for rec in rows_out:
                for key in rec:
                    if key not in fields:
                        fields.append(key)
            atomic_write_csv(output_dir / "sixway_per_sample.csv", fields, rows_out)
        atomic_write_csv(output_dir / "sixway_summary.csv", list(summary[0].keys()), summary)
        atomic_write_json(output_dir / "sixway_summary.json", summary)

        print("\n========== 六组结果汇总 ==========", flush=True)
        for item in summary:
            print(f"\n== {item['method']} ==", flush=True)
            print(f"completed: {item['completed']} / {len(eval_rows)}", flush=True)
            print(f"measured_speedup_vs_full_dense: {item['measured_speedup_vs_full_dense']}", flush=True)
            print(f"psnr_vs_full_dense: {item['psnr_vs_full_dense']}", flush=True)
            print(f"ssim_vs_full_dense: {item['ssim_vs_full_dense']}", flush=True)


# -----------------------------------------------------------------------------
# 十、入口
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args = setup_distributed(args)
    random.seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

    if not (0.0 < args.compute_ratio <= 1.0):
        raise ValueError("--compute-ratio 必须在 (0,1]。")
    if not (0.0 < args.min_compute_ratio <= args.compute_ratio):
        raise ValueError("--min-compute-ratio 必须 >0 且 <= compute-ratio。")
    if args.blueprint_calibration_count <= 0:
        raise ValueError("--blueprint-calibration-count 必须 > 0。")
    if not (0.0 < args.profile_quantile <= 1.0):
        raise ValueError("--profile-quantile 必须在 (0,1]。")
    if not (0.0 < args.target_cache_ratio < 1.0):
        raise ValueError("--target-cache-ratio 必须在 (0,1)。")
    if args.profile_smoothing_radius < 0:
        raise ValueError("--profile-smoothing-radius 不能 < 0。")
    if args.profile_aggregate_block_chunk <= 0:
        raise ValueError("--profile-aggregate-block-chunk 必须 > 0。")
    if args.convergence_min_epochs <= 0:
        raise ValueError("--convergence-min-epochs 必须 > 0。")
    if args.convergence_window_updates < 2:
        raise ValueError("--convergence-window-updates 必须 >= 2。")
    if args.convergence_patience <= 0:
        raise ValueError("--convergence-patience 必须 > 0。")
    if args.convergence_reward_abs_tol < 0 or args.convergence_reward_rel_tol < 0:
        raise ValueError("收敛 reward tolerance 不能 < 0。")
    if args.convergence_policy_delta_threshold < 0:
        raise ValueError("--convergence-policy-delta-threshold 不能 < 0。")
    if args.td3_latent_dim < 2:
        raise ValueError("--td3-latent-dim 必须 >= 2。")
    if args.td3_replay_capacity <= 0 or args.td3_batch_size <= 0 or args.td3_gradient_steps <= 0:
        raise ValueError("TD3 replay/batch/gradient 参数必须 > 0。")
    if not (0.0 < args.td3_tau <= 1.0):
        raise ValueError("--td3-tau 必须在 (0,1]。")
    if not (0.0 < args.td3_exploration_noise_decay <= 1.0):
        raise ValueError("--td3-exploration-noise-decay 必须在 (0,1]。")
    if args.convergence_max_epochs < 0:
        raise ValueError("--convergence-max-epochs 不能 < 0。")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest_distributed(args, output_dir)
    calibration_rows = [r for r in manifest if r["split"] == "calibration"]
    train_rows = [r for r in manifest if r["split"] == "train"]
    eval_rows = [r for r in manifest if r["split"] == "eval"]

    pipe = load_pipeline(args)
    total_layers = len(pipe.transformer.transformer_blocks)
    full_schedule = make_full_schedule(args.num_inference_steps, total_layers)

    if args.mode in {"all", "train", "blueprint"}:
        blueprint_schedule, blueprint_payload = build_fresh_blueprint(
            pipe=pipe,
            calibration_rows=calibration_rows,
            args=args,
            output_dir=output_dir,
            total_layers=total_layers,
        )
    else:
        schedule_path = output_dir / "blueprint" / "blue_line_schedule.json"
        if not schedule_path.is_file():
            raise FileNotFoundError(
                f"eval 找不到本次实验 Blueprint：{schedule_path}。请先运行 MODE=all 或 MODE=blueprint/train。"
            )
        blueprint_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        blueprint_schedule = {int(item["step_index_0based"]): item for item in blueprint_payload["schedule"]}

    if is_main_process():
        atomic_write_json(output_dir / "fresh_blueprint_summary.json", {
            "freshly_calibrated": bool(blueprint_payload.get("freshly_calibrated", False)),
            "calibration_sample_count": blueprint_payload.get("calibration_sample_count"),
            "profile_quantile": blueprint_payload.get("profile_quantile"),
            "target_cache_ratio": blueprint_payload.get("target_cache_ratio_before_contiguous_constraint"),
            "effective_cache_fraction": blueprint_payload.get("effective_cache_fraction"),
            "theoretical_block_speedup": blueprint_payload.get("theoretical_block_speedup"),
            "schedule": str((output_dir / "blueprint" / "blue_line_schedule.json").resolve()),
            "total_layers": total_layers,
            "num_steps": args.num_inference_steps,
            "world_size": get_world_size(),
        })

    if args.mode == "blueprint":
        if is_main_process():
            print("[done] Fresh Blueprint 已生成；MODE=blueprint 到此结束。", flush=True)
        cleanup_distributed()
        return

    full_policy_path = output_dir / "policy_full_rl25.pt"
    blueprint_policy_path = output_dir / "policy_blueprint_rl25.pt"

    if args.mode in {"all", "train"}:
        train_one_base(
            base_mode="full", pipe=pipe, train_rows=train_rows, args=args,
            full_schedule=full_schedule, blueprint_schedule=blueprint_schedule,
            policy_path=full_policy_path,
        )
        train_one_base(
            base_mode="blueprint", pipe=pipe, train_rows=train_rows, args=args,
            full_schedule=full_schedule, blueprint_schedule=blueprint_schedule,
            policy_path=blueprint_policy_path,
        )

    if args.mode in {"all", "eval"}:
        evaluate_sixway(
            pipe=pipe, eval_rows=eval_rows, train_rows=train_rows, args=args,
            full_schedule=full_schedule, blueprint_schedule=blueprint_schedule,
            full_policy_path=full_policy_path,
            blueprint_policy_path=blueprint_policy_path,
            output_dir=output_dir,
        )

    cleanup_distributed()


if __name__ == "__main__":
    main()
