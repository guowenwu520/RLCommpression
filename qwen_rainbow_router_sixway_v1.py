#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit-2511: fixed-total-budget teacher-free-observation macro router v10

核心设计
========
1. 整个 denoising episode 的 image-token/block 代理计算量严格固定为 --compute-ratio（默认 25% Full）。
2. Rainbow-style Router 每个 timestep 只选择一次宏动作；full 分支只做 temporal token budget routing，
   blueprint 分支联合选择 Full/Safe/Normal/Aggressive Block 模式与当前 step 总预算。
3. Policy state 不再包含 Full Teacher 的 prev_score。替换为当前压缩模型自己可观测的 token hidden-change
   统计（上一 step 的 mean/max），因此 Teacher 只用于训练 reward/评估，不进入部署时 policy observation。
4. 新增严格公平的 global-uniform25 基线：整条 episode 总代理预算与 RL 完全相同，Blueprint 仍固定使用
   Normal Blueprint schedule，只通过静态水位投影分配每个 step 的 token 数。
5. 新增 static-learned25 基线：从 best validation epoch 的 deterministic action 分布中，通过 DP 在同样硬预算
   下提取一条全样本共享的静态 timestep action schedule，用于区分“普适 timestep 规律”和“sample-adaptive routing”。
6. 训练和评估自动输出较全面诊断图：收敛曲线、action 概率热力图、平均 step 预算、action entropy、
   static-vs-RL disagreement、剩余预算、Blueprint 执行 Block 数、speed-quality Pareto、paired PSNR gain 等。
7. 仍使用 Dueling Double-DQN + Prioritized Replay + n-step return（Rainbow-style，不宣称完整 Rainbow）。

八组评估：
- full_dense
- blueprint_only
- full_uniform25
- full_static25
- full_rl25
- blueprint_uniform25
- blueprint_static25
- blueprint_rl25
"""
from __future__ import annotations

import contextlib
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qwen_rl_blueprint_sixway_fresh_v2 as core

ALGO_VERSION = "rainbow_macro_router_teacher_free_obs_global_budget_v2"
core.RL_ALGORITHM_VERSION = ALGO_VERSION

# -----------------------------------------------------------------------------
# 环境变量配置：不侵入 core argparse，原 shell/CLI 仍可继续使用。
# -----------------------------------------------------------------------------
def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


ROUTER_VAL_COUNT = max(1, env_int("ROUTER_VAL_COUNT", 8))
ROUTER_REPLAY_CAPACITY = max(1000, env_int("ROUTER_REPLAY_CAPACITY", 50000))
ROUTER_BATCH_SIZE = max(32, env_int("ROUTER_BATCH_SIZE", 256))
ROUTER_GRADIENT_STEPS = max(1, env_int("ROUTER_GRADIENT_STEPS", 160))
ROUTER_WARMUP = max(128, env_int("ROUTER_WARMUP", 1000))
ROUTER_N_STEP = max(1, env_int("ROUTER_N_STEP", 3))
ROUTER_GAMMA = env_float("ROUTER_GAMMA", 0.99)
ROUTER_LR = env_float("ROUTER_LR", 1e-4)
ROUTER_EPS_START = env_float("ROUTER_EPS_START", 0.30)
ROUTER_EPS_MIN = env_float("ROUTER_EPS_MIN", 0.05)
ROUTER_EPS_DECAY = env_float("ROUTER_EPS_DECAY", 0.85)
ROUTER_TARGET_UPDATE = max(1, env_int("ROUTER_TARGET_UPDATE", 250))
ROUTER_PER_ALPHA = env_float("ROUTER_PER_ALPHA", 0.6)
ROUTER_PER_BETA_START = env_float("ROUTER_PER_BETA_START", 0.4)
ROUTER_PER_BETA_END = env_float("ROUTER_PER_BETA_END", 1.0)
ROUTER_MIN_EPOCHS = max(1, env_int("ROUTER_MIN_EPOCHS", 3))
ROUTER_PATIENCE = max(1, env_int("ROUTER_PATIENCE", 3))
ROUTER_MAX_EPOCHS = max(0, env_int("ROUTER_MAX_EPOCHS", 20))
ROUTER_VAL_ABS_TOL = max(0.0, env_float("ROUTER_VAL_ABS_TOL", 1e-4))
ROUTER_VAL_REL_TOL = max(0.0, env_float("ROUTER_VAL_REL_TOL", 0.01))
ROUTER_SAFE_EXEC_FRAC = min(0.95, max(0.35, env_float("ROUTER_SAFE_EXEC_FRAC", 0.65)))
ROUTER_AGGR_EXEC_FRAC = min(0.70, max(0.10, env_float("ROUTER_AGGR_EXEC_FRAC", 0.30)))
ROUTER_QUIET_INNER = env_bool("ROUTER_QUIET_INNER", True)
ROUTER_SEED = env_int("ROUTER_SEED", 20260820)
ROUTER_STATIC_LAPLACE = max(1e-6, env_float("ROUTER_STATIC_LAPLACE", 0.5))
ROUTER_WRITE_PLOTS = env_bool("ROUTER_WRITE_PLOTS", True)

# 所有动作的总 step 计算比例都是 5% 的整数倍，因此 50-step + 25% 总预算可以离散精确闭合。
# 对 blueprint 分支，Block schedule 不同，但 token 数会自动调整，使该 step 的总代理计算量仍是此比例。
ACTION_NAMES = [
    "A0_FULL_100",
    "A1_SAFE_50",
    "A2_NORMAL_35",
    "A3_NORMAL_25",
    "A4_AGGRESSIVE_15",
    "A5_AGGRESSIVE_05",
]
ACTION_STEP_RATIOS = np.asarray([1.00, 0.50, 0.35, 0.25, 0.15, 0.05], dtype=np.float64)
BUDGET_UNIT = 0.05
ACTION_UNITS = np.rint(ACTION_STEP_RATIOS / BUDGET_UNIT).astype(np.int64)  # [20,10,7,5,3,1]
NUM_ACTIONS = len(ACTION_NAMES)
STATE_NAMES = [
    "step_fraction",
    "remaining_budget_fraction",
    "remaining_steps_fraction",
    "prev_action",
    "prev_compute_ratio",
    "normal_block_fraction",
    "safe_block_fraction",
    "aggressive_block_fraction",
    "risk_mean_log1p",
    "risk_max_log1p",
    "risk_p90_log1p",
    "block_cache_age_mean",
    "block_cache_age_max",
    "block_cache_expired_fraction",
    "prev_token_change_mean_log1p",
    "prev_token_change_max_log1p",
]
STATE_DIM = len(STATE_NAMES)

# -----------------------------------------------------------------------------
# schedule 变体：Normal = 当前 Fresh Blueprint；Safe 更宽；Aggressive 更窄。
# -----------------------------------------------------------------------------
def _risk_vector(item: Dict[str, Any], total_layers: int) -> np.ndarray:
    values = item.get("smoothed_risk_by_block") or item.get("risk_by_block") or []
    arr = np.zeros(total_layers, dtype=np.float64)
    if isinstance(values, list):
        n = min(total_layers, len(values))
        if n:
            arr[:n] = np.asarray(values[:n], dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _best_contiguous_internal_window(risk: np.ndarray, count: int, total_layers: int) -> List[int]:
    internal = list(range(1, max(1, total_layers - 1)))
    if total_layers <= 2 or count <= 0:
        return []
    count = max(1, min(int(count), len(internal)))
    best_start = 1
    best_score = -float("inf")
    # log1p 防止单个极端风险值完全控制窗口。
    rr = np.log1p(np.maximum(risk, 0.0))
    for start in range(1, total_layers - count):
        end = start + count
        score = float(rr[start:end].sum())
        if score > best_score:
            best_score = score
            best_start = start
    return list(range(best_start, best_start + count))


def _schedule_item_from_count(
    base_item: Dict[str, Any],
    *,
    internal_count: int,
    total_layers: int,
    mode_name: str,
) -> Dict[str, Any]:
    risk = _risk_vector(base_item, total_layers)
    internal = _best_contiguous_internal_window(risk, internal_count, total_layers)
    executed = sorted(set(([0, total_layers - 1] if total_layers >= 2 else [0]) + internal))
    item = copy.deepcopy(base_item)
    item["mode"] = mode_name
    item["executed_blocks_0based"] = executed
    item["base_executed_blocks_0based"] = executed
    item["base_skipped_blocks_0based"] = sorted(set(range(total_layers)) - set(executed))
    # 动态 Router 已经负责刷新模式选择；不沿用 Normal schedule 的静态 forced_refresh 列表，
    # 否则会偷偷改变动作实际成本。Token 级 max-cache-age 仍保留并在固定预算内优先刷新。
    item["forced_refresh_blocks_0based"] = []
    return item


def build_router_schedule_variants(
    full_schedule: Dict[int, Dict[str, Any]],
    blueprint_schedule: Dict[int, Dict[str, Any]],
    total_layers: int,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    safe: Dict[int, Dict[str, Any]] = {}
    normal: Dict[int, Dict[str, Any]] = {}
    aggressive: Dict[int, Dict[str, Any]] = {}
    internal_total = max(1, total_layers - 2)
    safe_floor = int(math.ceil(internal_total * ROUTER_SAFE_EXEC_FRAC))
    aggr_target = int(math.ceil(internal_total * ROUTER_AGGR_EXEC_FRAC))
    for step in sorted(full_schedule):
        if step == 0:
            safe[step] = copy.deepcopy(full_schedule[step])
            normal[step] = copy.deepcopy(full_schedule[step])
            aggressive[step] = copy.deepcopy(full_schedule[step])
            continue
        base = copy.deepcopy(blueprint_schedule[step])
        normal_exec = [int(v) for v in base.get("executed_blocks_0based", []) if int(v) not in {0, total_layers - 1}]
        normal_count = max(1, len(normal_exec))
        safe_count = max(normal_count, safe_floor)
        aggr_count = min(normal_count, max(1, aggr_target))
        safe[step] = _schedule_item_from_count(
            base, internal_count=safe_count, total_layers=total_layers, mode_name="router_safe"
        )
        normal[step] = copy.deepcopy(base)
        normal[step]["mode"] = "router_normal"
        normal[step]["forced_refresh_blocks_0based"] = []
        aggressive[step] = _schedule_item_from_count(
            base, internal_count=aggr_count, total_layers=total_layers, mode_name="router_aggressive"
        )
    return {"full": full_schedule, "safe": safe, "normal": normal, "aggressive": aggressive}


def _mode_schedule_name(base_mode: str, action: int) -> str:
    if base_mode == "full":
        return "full"
    return ["full", "safe", "normal", "normal", "aggressive", "aggressive"][int(action)]


def _item_for_action(
    variants: Dict[str, Dict[int, Dict[str, Any]]],
    base_mode: str,
    action: int,
    step: int,
) -> Dict[str, Any]:
    return copy.deepcopy(variants[_mode_schedule_name(base_mode, action)][int(step)])


def _required_internal_ratio(
    *, desired_step_ratio: float, item: Dict[str, Any], total_layers: int, cache_edge_blocks: bool
) -> Tuple[float, int, int]:
    executed = sorted({int(v) for v in item.get("executed_blocks_0based", [])})
    edge_set = {0, total_layers - 1} if total_layers >= 2 else {0}
    mandatory_edges = 0 if cache_edge_blocks else len([v for v in executed if v in edge_set])
    internal = [v for v in executed if cache_edge_blocks or v not in edge_set]
    n = len(internal)
    desired_equiv = float(desired_step_ratio) * float(total_layers)
    if n == 0:
        return (0.0 if abs(desired_equiv - mandatory_edges) < 1e-9 else float("inf"), n, mandatory_edges)
    ratio = (desired_equiv - mandatory_edges) / n
    return float(ratio), n, mandatory_edges


def action_feasible(
    *,
    base_mode: str,
    action: int,
    step: int,
    variants: Dict[str, Dict[int, Dict[str, Any]]],
    total_layers: int,
    min_compute_ratio: float,
    cache_edge_blocks: bool,
) -> bool:
    if step == 0:
        return int(action) == 0  # token cache 初始化 step 必须 Full。
    item = _item_for_action(variants, base_mode, action, step)
    req, n, _ = _required_internal_ratio(
        desired_step_ratio=float(ACTION_STEP_RATIOS[action]), item=item,
        total_layers=total_layers, cache_edge_blocks=cache_edge_blocks,
    )
    if n <= 0:
        return False
    return (float(min_compute_ratio) - 1e-9) <= req <= (1.0 + 1e-9)


def build_budget_reachability(
    *,
    base_mode: str,
    variants: Dict[str, Dict[int, Dict[str, Any]]],
    total_layers: int,
    num_steps: int,
    target_ratio: float,
    min_compute_ratio: float,
    cache_edge_blocks: bool,
) -> Tuple[List[List[int]], List[set], int, float]:
    feasible: List[List[int]] = []
    for step in range(num_steps):
        acts = [
            a for a in range(NUM_ACTIONS)
            if action_feasible(
                base_mode=base_mode, action=a, step=step, variants=variants,
                total_layers=total_layers, min_compute_ratio=min_compute_ratio,
                cache_edge_blocks=cache_edge_blocks,
            )
        ]
        if not acts:
            raise RuntimeError(f"Router step={step} 没有任何可行动作。")
        feasible.append(acts)

    reachable: List[set] = [set() for _ in range(num_steps + 1)]
    reachable[num_steps] = {0}
    for step in range(num_steps - 1, -1, -1):
        sums = set()
        for a in feasible[step]:
            u = int(ACTION_UNITS[a])
            for tail in reachable[step + 1]:
                sums.add(u + int(tail))
        reachable[step] = sums

    requested_units = int(round(float(target_ratio) * num_steps / BUDGET_UNIT))
    if requested_units in reachable[0]:
        target_units = requested_units
    else:
        # 理论上默认50 steps / 25%可以精确命中；自定义步数或比例若不可达，取最近可达值并明确打印。
        target_units = min(reachable[0], key=lambda x: abs(int(x) - requested_units))
    effective_ratio = float(target_units) * BUDGET_UNIT / max(1, num_steps)
    return feasible, reachable, int(target_units), float(effective_ratio)

# -----------------------------------------------------------------------------
# Dueling Double-DQN + PER
# -----------------------------------------------------------------------------
class DuelingQNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = NUM_ACTIONS, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.adv = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=-1, keepdim=True)


@dataclass
class RouteTransition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    next_valid_mask: torch.Tensor
    step_index: int
    mode_name: str
    step_compute_ratio: float
    quality_score: float


def serialize_transition(t: RouteTransition) -> Dict[str, Any]:
    return {
        "state": t.state.cpu(), "action": int(t.action), "reward": float(t.reward),
        "next_state": t.next_state.cpu(), "done": bool(t.done),
        "next_valid_mask": t.next_valid_mask.cpu(), "step_index": int(t.step_index),
        "mode_name": str(t.mode_name), "step_compute_ratio": float(t.step_compute_ratio),
        "quality_score": float(t.quality_score),
    }


def deserialize_transition(x: Dict[str, Any]) -> RouteTransition:
    return RouteTransition(
        state=x["state"].float().cpu(), action=int(x["action"]), reward=float(x["reward"]),
        next_state=x["next_state"].float().cpu(), done=bool(x["done"]),
        next_valid_mask=x["next_valid_mask"].bool().cpu(), step_index=int(x["step_index"]),
        mode_name=str(x.get("mode_name", "")), step_compute_ratio=float(x.get("step_compute_ratio", 0.0)),
        quality_score=float(x.get("quality_score", 0.0)),
    )


class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.data: List[RouteTransition] = []
        self.priorities: List[float] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, t: RouteTransition, priority: Optional[float] = None) -> None:
        p = float(priority if priority is not None else (max(self.priorities) if self.priorities else 1.0))
        p = max(p, 1e-6)
        if len(self.data) < self.capacity:
            self.data.append(t)
            self.priorities.append(p)
        else:
            self.data[self.pos] = t
            self.priorities[self.pos] = p
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float, rng: np.random.RandomState):
        n = len(self.data)
        p = np.asarray(self.priorities, dtype=np.float64) ** self.alpha
        p /= max(p.sum(), 1e-12)
        replace = n < batch_size
        idx = rng.choice(n, size=batch_size, replace=replace, p=p)
        prob = p[idx]
        weights = (n * prob) ** (-float(beta))
        weights /= max(float(weights.max()), 1e-12)
        return [self.data[int(i)] for i in idx], idx.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: Sequence[int], values: Sequence[float]) -> None:
        for i, v in zip(indices, values):
            self.priorities[int(i)] = max(float(v), 1e-6)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity, "alpha": self.alpha, "pos": self.pos,
            "data": [serialize_transition(t) for t in self.data],
            "priorities": list(self.priorities),
        }

    @classmethod
    def from_state_dict(cls, payload: Dict[str, Any]) -> "PrioritizedReplay":
        obj = cls(int(payload.get("capacity", ROUTER_REPLAY_CAPACITY)), float(payload.get("alpha", ROUTER_PER_ALPHA)))
        obj.data = [deserialize_transition(x) for x in payload.get("data", [])]
        obj.priorities = [float(x) for x in payload.get("priorities", [1.0] * len(obj.data))]
        obj.pos = int(payload.get("pos", 0)) % max(1, obj.capacity)
        return obj


def make_n_step(transitions: Sequence[RouteTransition], n_step: int, gamma: float) -> List[RouteTransition]:
    out: List[RouteTransition] = []
    for i, src in enumerate(transitions):
        reward = 0.0
        last = src
        discount = 1.0
        for k in range(n_step):
            j = i + k
            if j >= len(transitions):
                break
            cur = transitions[j]
            reward += discount * float(cur.reward)
            last = cur
            if cur.done:
                break
            discount *= gamma
        steps_used = max(1, int(last.step_index - src.step_index + 1))
        # target 中使用 gamma**steps_used；为了不改结构，把步数存在 mode_name 后缀之外不优雅。
        # 这里 reward 已聚合，下一状态用 last；训练时统一 gamma**n，terminal/尾部误差很小。
        out.append(RouteTransition(
            state=src.state.clone(), action=src.action, reward=float(reward),
            next_state=last.next_state.clone(), done=bool(last.done),
            next_valid_mask=last.next_valid_mask.clone(), step_index=src.step_index,
            mode_name=src.mode_name, step_compute_ratio=src.step_compute_ratio, quality_score=src.quality_score,
        ))
    return out

# -----------------------------------------------------------------------------
# Router runtime：每 timestep 一次动作，同 step 所有 CFG branch 共用。
# -----------------------------------------------------------------------------
class RouterRuntime:
    def __init__(
        self, qnet: DuelingQNet, device: str, *, epsilon: float, seed: int,
        feasible: List[List[int]], reachable: List[set], target_units: int,
    ) -> None:
        self.qnet = qnet.to(device)
        self.device = torch.device(device)
        self.epsilon = float(epsilon)
        self.rng = np.random.RandomState(int(seed) & 0x7fffffff)
        self.feasible = feasible
        self.reachable = reachable
        self.target_units = int(target_units)
        self.remaining_units = int(target_units)
        self.transitions: List[RouteTransition] = []
        self._pending_index: Optional[int] = None
        self._current_step: Optional[int] = None

    def valid_mask(self, step: int) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        for a in self.feasible[int(step)]:
            u = int(ACTION_UNITS[a])
            rem = self.remaining_units - u
            if rem >= 0 and rem in self.reachable[int(step) + 1]:
                mask[a] = True
        if not mask.any():
            raise RuntimeError(
                f"预算约束无可行动作：step={step}, remaining_units={self.remaining_units}, "
                f"feasible={self.feasible[int(step)]}"
            )
        return mask

    @torch.no_grad()
    def begin_step(self, state_values: Sequence[float], step: int) -> Tuple[int, np.ndarray]:
        state = torch.tensor(state_values, dtype=torch.float32)
        mask_np = self.valid_mask(step)
        # 当前 state 成为上一 transition 的 next_state。
        if self._pending_index is not None:
            prev = self.transitions[self._pending_index]
            prev.next_state = state.clone()
            prev.next_valid_mask = torch.tensor(mask_np, dtype=torch.bool)
        valid = np.flatnonzero(mask_np)
        if self.epsilon > 0 and self.rng.rand() < self.epsilon:
            action = int(self.rng.choice(valid))
        else:
            q = self.qnet(state.to(self.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
            q[~mask_np] = -1e30
            action = int(np.argmax(q))
        self.remaining_units -= int(ACTION_UNITS[action])
        idx = len(self.transitions)
        self.transitions.append(RouteTransition(
            state=state, action=action, reward=0.0,
            next_state=torch.zeros_like(state), done=False,
            next_valid_mask=torch.zeros(NUM_ACTIONS, dtype=torch.bool),
            step_index=int(step), mode_name=ACTION_NAMES[action],
            step_compute_ratio=float(ACTION_STEP_RATIOS[action]), quality_score=0.0,
        ))
        self._pending_index = idx
        self._current_step = int(step)
        return action, mask_np

    def finish_step(self, reward: float, quality_score: float, *, terminal: bool) -> None:
        if self._pending_index is None:
            raise RuntimeError("finish_step 没有 pending transition。")
        t = self.transitions[self._pending_index]
        t.reward = float(reward)
        t.quality_score = float(quality_score)
        if terminal:
            t.done = True
            t.next_state = torch.zeros_like(t.state)
            t.next_valid_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
            self._pending_index = None
            if self.remaining_units != 0:
                raise RuntimeError(f"episode 预算没有闭合：remaining_units={self.remaining_units}")

# -----------------------------------------------------------------------------
# 动态 Block+Token controller
# -----------------------------------------------------------------------------
class MacroRouterController(core.BlueLineTokenScheduledController):
    def __init__(
        self,
        *,
        transformer_blocks,
        original_transformer_forward,
        normal_schedule,
        schedule_variants,
        teacher_references,
        args,
        forwards_per_step,
        base_mode: str,
        router_runtime: RouterRuntime,
    ) -> None:
        # 父类 schedule 后面会被逐 step 动态替换；初始使用 normal/full 仅用于初始化。
        initial_schedule = normal_schedule if base_mode == "blueprint" else schedule_variants["full"]
        super().__init__(
            transformer_blocks=transformer_blocks,
            original_transformer_forward=original_transformer_forward,
            schedule=copy.deepcopy(initial_schedule),
            teacher_references=teacher_references,
            args=args,
            forwards_per_step=forwards_per_step,
        )
        self.base_mode = str(base_mode)
        self.variants = schedule_variants
        self.router_runtime = router_runtime
        self.step_action: Dict[int, int] = {}
        self.step_valid_mask: Dict[int, np.ndarray] = {}
        self.step_desired_ratio: Dict[int, float] = {}
        self.step_plans: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.step_scores: Dict[int, List[float]] = {}
        self.prev_action = 0
        self.prev_compute_ratio = 1.0
        # Teacher-free online feedback: 上一步执行 Block 内 token hidden 相对其缓存来源的变化。
        # 这两个量由当前压缩模型自己的 token selector 产生，不需要 Full Teacher。
        self.prev_token_change_mean = 0.0
        self.prev_token_change_max = 0.0
        self.total_layers = len(transformer_blocks)
        self._normal_schedule_ref = normal_schedule
        # 禁用旧 token-RL；只使用 fixed token selector + 本 Router 给出的严格 count。
        self.rl_policy = None
        self.token_policy_mode = "fixed"
        self.token_budget_mode = "per_block"

    def _current_block_age_features(self, step: int) -> Tuple[float, float, float]:
        refresh = self.last_block_refresh_steps.get(0, {})
        ages = []
        for b in range(self.total_layers):
            s = refresh.get(b)
            ages.append(float(step - s) if s is not None else 0.0)
        arr = np.asarray(ages, dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0, 0.0
        denom = max(1.0, float(self.args.num_inference_steps))
        expired = float(np.mean(arr > max(1, int(getattr(self.args, "blueprint_max_cache_age", 5)))))
        return float(arr.mean() / denom), float(arr.max() / denom), expired

    def _build_state(self, step: int) -> List[float]:
        normal = self.variants["normal"][step]
        safe = self.variants["safe"][step]
        aggr = self.variants["aggressive"][step]
        risk = np.log1p(np.maximum(_risk_vector(normal, self.total_layers), 0.0))
        age_mean, age_max, expired = self._current_block_age_features(step)
        remaining_steps = max(1, int(self.args.num_inference_steps) - step)
        target = max(1, self.router_runtime.target_units)
        state = [
            step / max(1, int(self.args.num_inference_steps) - 1),
            self.router_runtime.remaining_units / target,
            remaining_steps / max(1, int(self.args.num_inference_steps)),
            float(self.prev_action) / max(1, NUM_ACTIONS - 1),
            float(self.prev_compute_ratio),
            len(normal.get("executed_blocks_0based", [])) / max(1, self.total_layers),
            len(safe.get("executed_blocks_0based", [])) / max(1, self.total_layers),
            len(aggr.get("executed_blocks_0based", [])) / max(1, self.total_layers),
            float(np.mean(risk)) if risk.size else 0.0,
            float(np.max(risk)) if risk.size else 0.0,
            float(np.quantile(risk, 0.90)) if risk.size else 0.0,
            age_mean,
            age_max,
            expired,
            math.log1p(max(0.0, float(self.prev_token_change_mean))),
            math.log1p(max(0.0, float(self.prev_token_change_max))),
        ]
        if len(state) != STATE_DIM:
            raise RuntimeError(f"Router state dim {len(state)} != {STATE_DIM}")
        return state

    def _prepare_plan(self, step: int, branch: int, item: Dict[str, Any], token_count: int) -> Dict[str, Any]:
        key = (step, branch)
        if key in self.step_plans:
            return self.step_plans[key]
        action = self.step_action[step]
        desired_ratio = float(ACTION_STEP_RATIOS[action])
        executed = sorted({int(v) for v in item.get("executed_blocks_0based", [])})
        edge_set = {0, self.total_layers - 1} if self.total_layers >= 2 else {0}
        edge_layers = [v for v in executed if v in edge_set and not self.cache_edge_blocks]
        sparse_layers = [v for v in executed if self.cache_edge_blocks or v not in edge_set]
        desired_total = int(round(desired_ratio * self.total_layers * token_count))
        mandatory = len(edge_layers) * token_count
        remaining = desired_total - mandatory
        min_count = max(1, int(math.ceil(token_count * float(self.min_compute_ratio))))
        n = len(sparse_layers)
        if n <= 0 or remaining < n * min_count or remaining > n * token_count:
            raise RuntimeError(
                f"Router action不可实现 step={step} action={ACTION_NAMES[action]} desired={desired_ratio:.2f} "
                f"executed={len(executed)} sparse_layers={n} remaining={remaining} "
                f"range=[{n*min_count},{n*token_count}]"
            )
        counts = core.BlockBudgetTokenController._bounded_integer_allocation(
            np.ones(n, dtype=np.float64), total_budget=remaining, min_count=min_count, max_count=token_count
        )
        plan = {
            "counts": {int(b): int(c) for b, c in zip(sparse_layers, counts)},
            "edge_layers": set(edge_layers), "desired_total": int(desired_total),
            "token_count": int(token_count), "action": int(action), "desired_ratio": desired_ratio,
        }
        self.step_plans[key] = plan
        return plan

    def _prepare_token_decision(self, **kwargs) -> Dict[str, Any]:
        step = int(kwargs["step_index"])
        branch = int(kwargs["branch_index"])
        layer = int(kwargs["layer_index"])
        image_input = kwargs["image_input"]
        token_count = int(image_input.shape[1])
        item = kwargs["item"]
        if step == 0:
            return super()._prepare_token_decision(**kwargs)
        plan = self._prepare_plan(step, branch, item, token_count)
        if layer in plan["edge_layers"]:
            return super()._prepare_token_decision(**kwargs)
        count = plan["counts"].get(layer)
        if count is None:
            return super()._prepare_token_decision(**kwargs)
        # 父类使用 ceil(token_count * compute_ratio)，减一个极小量可精确得到整数 count。
        effective_compute = max(0.0, (int(count) - 1e-6) / max(1, token_count))
        old = self.token_cache_ratio
        self.token_cache_ratio = 1.0 - effective_compute
        try:
            result = super()._prepare_token_decision(**kwargs)
        finally:
            self.token_cache_ratio = old
        actual = int(result["metadata"]["computed_image_token_count"])
        if actual != int(count):
            raise RuntimeError(f"Router token budget失配 step={step} block={layer}: {actual}!={count}")
        result["metadata"].update({
            "router_action": ACTION_NAMES[plan["action"]],
            "router_step_compute_ratio": float(plan["desired_ratio"]),
        })
        return result

    def __call__(self, *args, **kwargs):
        step = self.call_index // self.forwards_per_step
        branch = self.call_index % self.forwards_per_step
        if branch == 0:
            state = self._build_state(step)
            action, valid_mask = self.router_runtime.begin_step(state, step)
            self.step_action[step] = int(action)
            self.step_valid_mask[step] = valid_mask
            self.step_desired_ratio[step] = float(ACTION_STEP_RATIOS[action])
            item = _item_for_action(self.variants, self.base_mode, action, step)
            self.schedule[step] = item
        # branch>0 必须沿用 branch0 的同一个动作和同一个 schedule。
        output = super().__call__(*args, **kwargs)
        row = self.branch_step_rows[-1]
        row["router_action_index"] = int(self.step_action[step])
        row["router_action_name"] = ACTION_NAMES[self.step_action[step]]
        row["router_planned_step_compute_ratio"] = float(self.step_desired_ratio[step])
        self.step_scores.setdefault(step, []).append(float(row["score"]))

        # 从父 controller 本 step/branch 的 token_action_rows 中读取 teacher-free hidden-change。
        # selection_score_all_mean = 当前 image hidden 与该 token 上次真实计算来源 hidden 的归一化变化，
        # 它本来就用于 fixed token selector，因此部署时无需额外 Full Teacher forward。
        drift_vals = []
        for token_row in self.token_action_rows:
            if (
                int(token_row.get("step_index_0based", -1)) == step
                and int(token_row.get("branch_index_0based", -1)) == branch
            ):
                value = token_row.get("selection_score_all_mean")
                if value is not None and math.isfinite(float(value)):
                    drift_vals.append(float(value))
        branch_drift_mean = float(np.mean(drift_vals)) if drift_vals else 0.0
        branch_drift_max = float(np.max(drift_vals)) if drift_vals else 0.0
        self.__dict__.setdefault("step_drift_mean", {}).setdefault(step, []).append(branch_drift_mean)
        self.__dict__.setdefault("step_drift_max", {}).setdefault(step, []).append(branch_drift_max)

        if branch == self.forwards_per_step - 1:
            score = float(np.mean(self.step_scores[step]))
            # Teacher 只用于训练 reward/离线评估，不进入下一步 policy state。
            reward = -math.log1p(max(0.0, score))
            terminal = step >= int(self.args.num_inference_steps) - 1
            self.router_runtime.finish_step(reward, score, terminal=terminal)
            self.prev_action = int(self.step_action[step])
            self.prev_compute_ratio = float(self.step_desired_ratio[step])
            self.prev_token_change_mean = float(np.mean(self.step_drift_mean.get(step, [0.0])))
            self.prev_token_change_max = float(np.max(self.step_drift_max.get(step, [0.0])))
        return output


def _quiet_context():
    if ROUTER_QUIET_INNER:
        return contextlib.redirect_stdout(io.StringIO())
    return contextlib.nullcontext()


def run_router_method(
    *, pipe, image, sample_args, forwards_per_step, normal_schedule, variants, teacher_refs,
    base_mode: str, runtime: RouterRuntime,
):
    blocks = list(pipe.transformer.transformer_blocks)
    ctrl = MacroRouterController(
        transformer_blocks=blocks,
        original_transformer_forward=pipe.transformer.forward,
        normal_schedule=normal_schedule,
        schedule_variants=variants,
        teacher_references=teacher_refs,
        args=sample_args,
        forwards_per_step=forwards_per_step,
        base_mode=base_mode,
        router_runtime=runtime,
    )
    import time
    started = time.perf_counter()
    with _quiet_context():
        with core.replace_transformer_forward(pipe.transformer, ctrl):
            out = core.generate_image(pipe, [image], sample_args)
    elapsed = time.perf_counter() - started
    ctrl.validate_complete()
    return out, elapsed, ctrl

# -----------------------------------------------------------------------------
# 公平 global-uniform 基线 + static learned schedule runtime
# -----------------------------------------------------------------------------
def _ratio_bounds_for_item(
    *, item: Dict[str, Any], total_layers: int, min_compute_ratio: float,
    cache_edge_blocks: bool, step: int,
) -> Tuple[float, float]:
    """给定固定 Block schedule，返回该 step 可实现的 Full-equivalent proxy ratio 区间。"""
    if int(step) == 0:
        return 1.0, 1.0
    executed = sorted({int(v) for v in item.get("executed_blocks_0based", [])})
    edge_set = {0, total_layers - 1} if total_layers >= 2 else {0}
    mandatory_edges = 0 if cache_edge_blocks else len([v for v in executed if v in edge_set])
    sparse_layers = [v for v in executed if cache_edge_blocks or v not in edge_set]
    n = len(sparse_layers)
    if n <= 0:
        ratio = mandatory_edges / max(1, total_layers)
        return float(ratio), float(ratio)
    lo = (mandatory_edges + n * float(min_compute_ratio)) / max(1, total_layers)
    hi = (mandatory_edges + n * 1.0) / max(1, total_layers)
    return float(lo), float(hi)


def build_uniform_ratio_plan(
    *, schedule: Dict[int, Dict[str, Any]], total_layers: int, num_steps: int,
    target_ratio: float, min_compute_ratio: float, cache_edge_blocks: bool,
) -> Tuple[List[float], float]:
    """在固定 schedule 下构造“尽量每 step 相同”的严格 global-budget 比例计划。

    step0 因缓存初始化固定 Full。其它 step 通过 water-filling / box projection 找共同水位 c：
        ratio_t = clip(c, lower_t, upper_t)
    并让 sum_t ratio_t == target_ratio * num_steps。
    这样 Blueprint baseline 不再是“执行 Block 内 25%”，而是相对 Full episode 的同一个全局预算。
    """
    lows: List[float] = []
    highs: List[float] = []
    for step in range(num_steps):
        lo, hi = _ratio_bounds_for_item(
            item=schedule[step], total_layers=total_layers,
            min_compute_ratio=min_compute_ratio, cache_edge_blocks=cache_edge_blocks, step=step,
        )
        lows.append(lo); highs.append(hi)
    target_sum = float(target_ratio) * float(num_steps)
    lo_sum, hi_sum = float(sum(lows)), float(sum(highs))
    if target_sum < lo_sum - 1e-9 or target_sum > hi_sum + 1e-9:
        raise RuntimeError(
            f"global-uniform预算不可实现：target_sum={target_sum:.6f}, "
            f"feasible=[{lo_sum:.6f},{hi_sum:.6f}]"
        )

    left = min(lows) - 1.0
    right = max(highs) + 1.0
    for _ in range(100):
        mid = (left + right) * 0.5
        cur = sum(min(hi, max(lo, mid)) for lo, hi in zip(lows, highs))
        if cur < target_sum:
            left = mid
        else:
            right = mid
    level = (left + right) * 0.5
    plan = [min(hi, max(lo, level)) for lo, hi in zip(lows, highs)]

    # 消除浮点残差，保证计划比例求和严格等于 target_sum（数值精度内）。
    residual = target_sum - float(sum(plan))
    if abs(residual) > 1e-12:
        if residual > 0:
            for i in range(num_steps - 1, -1, -1):
                room = highs[i] - plan[i]
                if room <= 0:
                    continue
                add = min(room, residual)
                plan[i] += add; residual -= add
                if residual <= 1e-12: break
        else:
            need = -residual
            for i in range(num_steps - 1, -1, -1):
                room = plan[i] - lows[i]
                if room <= 0:
                    continue
                sub = min(room, need)
                plan[i] -= sub; need -= sub
                if need <= 1e-12: break
            residual = -need
    effective = float(sum(plan)) / max(1, num_steps)
    if abs(effective - float(target_ratio)) > 1e-8:
        raise RuntimeError(f"global-uniform预算闭合失败：effective={effective}, target={target_ratio}")
    return [float(x) for x in plan], effective


class FixedRatioController(core.BlueLineTokenScheduledController):
    """固定 Block schedule + 固定 per-step Full-equivalent ratio plan。

    用于公平 global-uniform25 baseline。Block 不由 RL 改，Token 数通过与 Router 相同的反解逻辑分配。
    """
    def __init__(self, *args, step_ratio_plan: Sequence[float], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step_ratio_plan = [float(x) for x in step_ratio_plan]
        self.step_plans: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._prefix_ratio_sum = np.cumsum(np.asarray(self.step_ratio_plan, dtype=np.float64))
        self._branch_used_block_token_units: Dict[int, int] = {}
        self.rl_policy = None
        self.token_policy_mode = "fixed"
        self.token_budget_mode = "per_block"

    def _prepare_plan(self, step: int, branch: int, item: Dict[str, Any], token_count: int) -> Dict[str, Any]:
        key = (step, branch)
        if key in self.step_plans:
            return self.step_plans[key]
        desired_ratio = float(self.step_ratio_plan[step])
        executed = sorted({int(v) for v in item.get("executed_blocks_0based", [])})
        edge_set = {0, self.total_layers - 1} if self.total_layers >= 2 else {0}
        edge_layers = [v for v in executed if v in edge_set and not self.cache_edge_blocks]
        sparse_layers = [v for v in executed if self.cache_edge_blocks or v not in edge_set]
        # 用累计目标做整数闭合：每个 step 的浮点计划先转换为 cumulative integer budget，
        # 当前 step 取“累计目标 - 已使用量”。因此最后一 step 后整条 episode 的实际
        # image-token×Block proxy 与 global target 在整数 token 层面也严格一致。
        if branch not in self._branch_used_block_token_units:
            self._branch_used_block_token_units[branch] = int(self.total_layers * token_count)  # step0 Full
        cumulative_target = int(round(float(self._prefix_ratio_sum[step]) * self.total_layers * token_count))
        desired_total = int(cumulative_target - self._branch_used_block_token_units[branch])
        mandatory = len(edge_layers) * token_count
        remaining = desired_total - mandatory
        min_count = max(1, int(math.ceil(token_count * float(self.min_compute_ratio))))
        n = len(sparse_layers)
        if n <= 0 or remaining < n * min_count or remaining > n * token_count:
            raise RuntimeError(
                f"Uniform预算不可实现 step={step} desired={desired_ratio:.8f} executed={len(executed)} "
                f"remaining={remaining} range=[{n*min_count},{n*token_count}]"
            )
        counts = core.BlockBudgetTokenController._bounded_integer_allocation(
            np.ones(n, dtype=np.float64), total_budget=remaining, min_count=min_count, max_count=token_count
        )
        plan = {
            "counts": {int(b): int(c) for b, c in zip(sparse_layers, counts)},
            "edge_layers": set(edge_layers), "desired_total": int(desired_total),
            "token_count": int(token_count), "desired_ratio": desired_ratio,
        }
        self.step_plans[key] = plan
        self._branch_used_block_token_units[branch] += int(desired_total)
        return plan

    def _prepare_token_decision(self, **kwargs) -> Dict[str, Any]:
        step = int(kwargs["step_index"]); branch = int(kwargs["branch_index"])
        layer = int(kwargs["layer_index"]); image_input = kwargs["image_input"]
        token_count = int(image_input.shape[1]); item = kwargs["item"]
        if step == 0:
            return super()._prepare_token_decision(**kwargs)
        plan = self._prepare_plan(step, branch, item, token_count)
        if layer in plan["edge_layers"]:
            return super()._prepare_token_decision(**kwargs)
        count = plan["counts"].get(layer)
        if count is None:
            return super()._prepare_token_decision(**kwargs)
        effective_compute = max(0.0, (int(count) - 1e-6) / max(1, token_count))
        old = self.token_cache_ratio
        self.token_cache_ratio = 1.0 - effective_compute
        try:
            result = super()._prepare_token_decision(**kwargs)
        finally:
            self.token_cache_ratio = old
        actual = int(result["metadata"]["computed_image_token_count"])
        if actual != int(count):
            raise RuntimeError(f"Uniform token budget失配 step={step} block={layer}: {actual}!={count}")
        result["metadata"].update({"uniform_step_compute_ratio": float(plan["desired_ratio"])})
        return result


def run_uniform_method(
    *, pipe, image, sample_args, forwards_per_step, schedule, teacher_refs, step_ratio_plan,
):
    ctrl = FixedRatioController(
        transformer_blocks=list(pipe.transformer.transformer_blocks),
        original_transformer_forward=pipe.transformer.forward,
        schedule=copy.deepcopy(schedule), teacher_references=teacher_refs,
        args=sample_args, forwards_per_step=forwards_per_step,
        step_ratio_plan=step_ratio_plan,
    )
    import time
    started = time.perf_counter()
    with _quiet_context():
        with core.replace_transformer_forward(pipe.transformer, ctrl):
            out = core.generate_image(pipe, [image], sample_args)
    elapsed = time.perf_counter() - started
    ctrl.validate_complete()
    return out, elapsed, ctrl


class StaticSequenceRuntime:
    """执行一条训练集/validation 提炼出的固定 action 序列；不读取 policy state 做决策。"""
    def __init__(self, *, actions: Sequence[int], feasible: List[List[int]], reachable: List[set], target_units: int) -> None:
        self.actions = [int(a) for a in actions]
        self.feasible = feasible
        self.reachable = reachable
        self.target_units = int(target_units)
        self.remaining_units = int(target_units)
        self.transitions: List[RouteTransition] = []
        self._pending_index: Optional[int] = None

    def valid_mask(self, step: int) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        for a in self.feasible[int(step)]:
            rem = self.remaining_units - int(ACTION_UNITS[a])
            if rem >= 0 and rem in self.reachable[int(step) + 1]:
                mask[a] = True
        return mask

    def begin_step(self, state_values: Sequence[float], step: int) -> Tuple[int, np.ndarray]:
        state = torch.tensor(state_values, dtype=torch.float32)
        mask_np = self.valid_mask(step)
        if self._pending_index is not None:
            prev = self.transitions[self._pending_index]
            prev.next_state = state.clone()
            prev.next_valid_mask = torch.tensor(mask_np, dtype=torch.bool)
        action = int(self.actions[step])
        if action < 0 or action >= NUM_ACTIONS or not mask_np[action]:
            raise RuntimeError(
                f"Static schedule不可执行：step={step}, action={action}, remaining={self.remaining_units}, "
                f"valid={np.flatnonzero(mask_np).tolist()}"
            )
        self.remaining_units -= int(ACTION_UNITS[action])
        idx = len(self.transitions)
        self.transitions.append(RouteTransition(
            state=state, action=action, reward=0.0, next_state=torch.zeros_like(state), done=False,
            next_valid_mask=torch.zeros(NUM_ACTIONS, dtype=torch.bool), step_index=int(step),
            mode_name=ACTION_NAMES[action], step_compute_ratio=float(ACTION_STEP_RATIOS[action]), quality_score=0.0,
        ))
        self._pending_index = idx
        return action, mask_np

    def finish_step(self, reward: float, quality_score: float, *, terminal: bool) -> None:
        if self._pending_index is None:
            raise RuntimeError("Static finish_step 没有 pending transition")
        t = self.transitions[self._pending_index]
        t.reward = float(reward); t.quality_score = float(quality_score)
        if terminal:
            t.done = True
            t.next_state = torch.zeros_like(t.state)
            t.next_valid_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
            self._pending_index = None
            if self.remaining_units != 0:
                raise RuntimeError(f"Static episode预算没有闭合：remaining_units={self.remaining_units}")

# -----------------------------------------------------------------------------
# 静态 Full Reference cache 复用（旧 Fixed25 cache 不删除，但 v10 不再依赖）
# -----------------------------------------------------------------------------
def load_or_build_static(
    *, pipe, row, args, schedule, base_mode: str, output_dir: Path, forwards_per_step: int
):
    sample_index = int(row["sample_index"])
    sample_args = core.make_sample_args(args, row)
    image = core.load_input_image(row["image_path"])
    paths = core._train_sample_record_paths(output_dir, sample_index, base_mode)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    if not paths["input"].is_file():
        core.save_image_atomic(image, paths["input"])
    static_paths = core._train_static_cache_paths(output_dir, sample_index, base_mode)
    static_paths["dir"].mkdir(parents=True, exist_ok=True)
    full_fp = core.full_reference_cache_fingerprint(args, row, forwards_per_step)
    full_payload = core._load_static_cache(static_paths["full_reference"], full_fp) if args.cache_train_static else None
    if full_payload is not None:
        teacher_refs = full_payload["teacher_refs"]
        teacher_img = core._load_png(paths["teacher"]) if paths["teacher"].is_file() else None
    else:
        with _quiet_context():
            teacher_img, elapsed, raw_refs = core.run_full_teacher(pipe, image, sample_args, forwards_per_step)
        teacher_refs = core._compact_teacher_references(raw_refs)
        core.release_teacher_references(raw_refs)
        if not paths["teacher"].is_file(): core.save_image_atomic(teacher_img, paths["teacher"])
        if args.cache_train_static:
            core.atomic_torch_save(static_paths["full_reference"], {
                "fingerprint": full_fp, "teacher_refs": teacher_refs,
                "first_compute_elapsed": float(elapsed), "tensor_bytes": int(core._tensor_tree_nbytes(teacher_refs)),
            })
    # v10 的 Router reward 只需要 Full Teacher。旧 fixed25 cache 保留在磁盘但不再新建，
    # 避免为了已经不使用的旧“执行Block内25%”定义额外跑一遍。
    return image, sample_args, teacher_refs

# -----------------------------------------------------------------------------
# fingerprint / policy IO
# -----------------------------------------------------------------------------
def router_config_dict(args, base_mode: str, variants, train_rows) -> Dict[str, Any]:
    return {
        "algorithm": ALGO_VERSION,
        "base_mode": base_mode,
        "state_dim": STATE_DIM,
        "state_names": STATE_NAMES,
        "observation_teacher_free": True,
        "teacher_free_feedback": "prev_token_selection_change_mean_max",
        "actions": ACTION_NAMES,
        "action_step_ratios": [float(x) for x in ACTION_STEP_RATIOS],
        "budget_unit": BUDGET_UNIT,
        "target_global_ratio": float(args.compute_ratio),
        "safe_exec_frac": ROUTER_SAFE_EXEC_FRAC,
        "aggr_exec_frac": ROUTER_AGGR_EXEC_FRAC,
        "min_compute_ratio": float(args.min_compute_ratio),
        "n_step": ROUTER_N_STEP, "gamma": ROUTER_GAMMA,
        "lr": ROUTER_LR, "hidden_dim": int(args.hidden_dim),
        "normal_schedule_sha256": core.schedule_fingerprint(variants["normal"]),
        "safe_schedule_sha256": core.schedule_fingerprint(variants["safe"]),
        "aggressive_schedule_sha256": core.schedule_fingerprint(variants["aggressive"]),
        "train_manifest_sha256": core.manifest_rows_fingerprint(train_rows),
        "model_path": str(args.model_path), "dtype": str(args.dtype),
        "num_steps": int(args.num_inference_steps),
    }


def router_fingerprint(args, base_mode: str, schedule, train_rows=None) -> Dict[str, Any]:
    # core.eval/main 兼容入口。schedule 本身不够构建变体，所以独立 eval 使用 policy 文件里的 fingerprint；
    # train/eval override 中会使用完整 variants 的 router_config_dict。
    return {
        "algorithm": ALGO_VERSION, "base_mode": str(base_mode),
        "target_global_ratio": float(args.compute_ratio),
        "train_manifest_sha256": core.manifest_rows_fingerprint(train_rows),
        "schedule_sha256": core.schedule_fingerprint(schedule),
    }


def save_router_policy(
    path: Path, qnet: DuelingQNet, fingerprint: Dict[str, Any], *,
    converged: bool, best_score: float, epoch: int, best_epoch: Optional[int] = None,
) -> None:
    core.atomic_torch_save(path, {
        "rl_algorithm_version": ALGO_VERSION,
        "state_dict": qnet.state_dict(),
        "state_dim": STATE_DIM, "action_dim": NUM_ACTIONS,
        "hidden_dim": int(qnet.feature[0].out_features),
        "rl_fingerprint": fingerprint,
        "training_converged": bool(converged),
        "best_validation_score": float(best_score), "epoch": int(epoch),
        "best_epoch": int(best_epoch if best_epoch is not None else epoch),
    })


def load_router_policy(path: Path, expected_fp: Dict[str, Any], device: str) -> DuelingQNet:
    p = torch.load(path, map_location="cpu")
    if p.get("rl_algorithm_version") != ALGO_VERSION or p.get("rl_fingerprint") != expected_fp:
        raise RuntimeError(f"Router policy 不兼容：{path}")
    net = DuelingQNet(STATE_DIM, NUM_ACTIONS, int(p.get("hidden_dim", 128)))
    net.load_state_dict(p["state_dict"])
    return net.to(device).eval()


def router_policy_compatible(path: Path, fp: Dict[str, Any]) -> bool:
    if not path.is_file(): return False
    try:
        p = torch.load(path, map_location="cpu")
        return p.get("rl_algorithm_version") == ALGO_VERSION and p.get("rl_fingerprint") == fp
    except Exception:
        return False

# -----------------------------------------------------------------------------
# 训练历史、静态 schedule 提炼与绘图
# -----------------------------------------------------------------------------
def _plot_module():
    if not ROUTER_WRITE_PLOTS:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"[plot] matplotlib不可用，跳过绘图：{type(exc).__name__}: {exc}", flush=True)
        return None


def _save_training_history(state_dir: Path, history: Sequence[Dict[str, Any]]) -> None:
    rows = [dict(x) for x in history]
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    core.atomic_write_csv(state_dir / "training_history.csv", fields, rows)
    core.atomic_write_json(state_dir / "training_history.json", {"history": rows})


def _plot_training_history(state_dir: Path, base_mode: str, history: Sequence[Dict[str, Any]]) -> None:
    plt = _plot_module()
    if plt is None or not history:
        return
    diag = state_dir / "plots"; diag.mkdir(parents=True, exist_ok=True)
    root_diag = state_dir.parent / "diagnostics"; root_diag.mkdir(parents=True, exist_ok=True)
    epochs = [int(x["epoch"]) for x in history]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, [float(x["val_score"]) for x in history], marker="o", label="validation score")
    plt.plot(epochs, [float(x["best_score"]) for x in history], linestyle="--", label="best score")
    plt.xlabel("Epoch"); plt.ylabel("Teacher quality score (lower is better)")
    plt.title(f"Router convergence - {base_mode}"); plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
    fig.savefig(diag / f"convergence_val_score_{base_mode}.png", dpi=180)
    fig.savefig(root_diag / f"convergence_val_score_{base_mode}.png", dpi=180); plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, [float(x["train_loss"]) for x in history], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("DQN TD loss")
    plt.title(f"Training loss - {base_mode}"); plt.grid(True, alpha=0.25); plt.tight_layout()
    fig.savefig(diag / f"training_loss_{base_mode}.png", dpi=180)
    fig.savefig(root_diag / f"training_loss_{base_mode}.png", dpi=180); plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(epochs, [float(x["epsilon"]) for x in history], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Epsilon")
    plt.title(f"Exploration schedule - {base_mode}"); plt.grid(True, alpha=0.25); plt.tight_layout()
    fig.savefig(diag / f"epsilon_{base_mode}.png", dpi=180)
    fig.savefig(root_diag / f"epsilon_{base_mode}.png", dpi=180); plt.close(fig)


def _static_schedule_path(output_dir: Path, base_mode: str) -> Path:
    return output_dir / f"static_learned_schedule_{base_mode}.json"


def _derive_static_schedule_from_validation(
    *, base_mode: str, state_dir: Path, output_dir: Path, val_rows: Sequence[Dict[str, Any]],
    best_epoch: int, args, variants: Dict[str, Dict[int, Dict[str, Any]]], total_layers: int,
    policy_fp: Dict[str, Any],
) -> Dict[str, Any]:
    """从 best validation epoch 的 deterministic action 分布提炼一条固定 action schedule。

    不是逐 step 直接取众数，而是在同一 global budget、同一 action feasibility 下做 DP，
    最大化各 step 的 log empirical probability。这样 static baseline 与 RL 拥有完全相同的离散预算约束。
    """
    num_steps = int(args.num_inference_steps)
    val_dir = state_dir / "validation" / f"epoch_{int(best_epoch):03d}"
    counts = np.zeros((num_steps, NUM_ACTIONS), dtype=np.int64)
    loaded = 0
    for row in val_rows:
        path = val_dir / f"sample_{int(row['sample_index']):05d}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        actions = payload.get("actions", [])
        if len(actions) != num_steps:
            continue
        for step, action in enumerate(actions):
            a = int(action)
            if 0 <= a < NUM_ACTIONS:
                counts[step, a] += 1
        loaded += 1
    if loaded <= 0:
        raise RuntimeError(f"无法从 best validation epoch={best_epoch} 提炼 static schedule：没有完整 action 记录。")

    feasible, reachable, target_units, effective_ratio = build_budget_reachability(
        base_mode=base_mode, variants=variants, total_layers=total_layers,
        num_steps=num_steps, target_ratio=float(args.compute_ratio),
        min_compute_ratio=float(args.min_compute_ratio), cache_edge_blocks=bool(args.token_cache_edge_blocks),
    )
    probs = (counts.astype(np.float64) + ROUTER_STATIC_LAPLACE) / (
        counts.sum(axis=1, keepdims=True).astype(np.float64) + ROUTER_STATIC_LAPLACE * NUM_ACTIONS
    )

    # dp[used_units] = (score, sequence)
    dp: Dict[int, Tuple[float, List[int]]] = {0: (0.0, [])}
    for step in range(num_steps):
        nxt: Dict[int, Tuple[float, List[int]]] = {}
        for used, (score, seq) in dp.items():
            for action in feasible[step]:
                new_used = int(used + int(ACTION_UNITS[action]))
                if new_used > target_units:
                    continue
                # 保证后续仍能精确闭合到 target_units。
                if (target_units - new_used) not in reachable[step + 1]:
                    continue
                new_score = score + math.log(max(float(probs[step, action]), 1e-12))
                old = nxt.get(new_used)
                if old is None or new_score > old[0]:
                    nxt[new_used] = (new_score, seq + [int(action)])
        if not nxt:
            raise RuntimeError(f"static schedule DP在 step={step} 无可行路径")
        dp = nxt
    if target_units not in dp:
        raise RuntimeError(f"static schedule DP无法闭合 target_units={target_units}")
    best_logprob, actions = dp[target_units]
    if len(actions) != num_steps:
        raise RuntimeError("static schedule长度错误")

    rows = []
    for step, action in enumerate(actions):
        rows.append({
            "step": step + 1,
            "action": int(action),
            "action_name": ACTION_NAMES[action],
            "step_compute_ratio": float(ACTION_STEP_RATIOS[action]),
            "validation_action_probability": float(probs[step, action]),
            "validation_action_count": int(counts[step, action]),
            "validation_sample_count": int(loaded),
        })
    payload = {
        "algorithm": ALGO_VERSION,
        "base_mode": base_mode,
        "source": "best_validation_deterministic_action_distribution_dp",
        "best_epoch": int(best_epoch),
        "validation_sample_count": int(loaded),
        "target_units": int(target_units),
        "effective_global_compute_ratio": float(effective_ratio),
        "best_log_probability": float(best_logprob),
        "actions": [int(a) for a in actions],
        "action_names": [ACTION_NAMES[a] for a in actions],
        "step_compute_ratios": [float(ACTION_STEP_RATIOS[a]) for a in actions],
        "action_counts": counts.tolist(),
        "action_probabilities": probs.tolist(),
        "policy_fingerprint_sha256": core.fingerprint_digest(policy_fp),
    }
    core.atomic_write_json(_static_schedule_path(output_dir, base_mode), payload)
    core.atomic_write_csv(output_dir / f"static_learned_schedule_{base_mode}.csv", list(rows[0].keys()), rows)
    _plot_static_schedule(output_dir, base_mode, payload)
    return payload


def _plot_static_schedule(output_dir: Path, base_mode: str, payload: Dict[str, Any]) -> None:
    plt = _plot_module()
    if plt is None:
        return
    diag = output_dir / "diagnostics"; diag.mkdir(parents=True, exist_ok=True)
    ratios = [float(x) for x in payload.get("step_compute_ratios", [])]
    actions = [int(x) for x in payload.get("actions", [])]
    if ratios:
        x = np.arange(1, len(ratios) + 1)
        fig = plt.figure(figsize=(10, 5))
        plt.step(x, ratios, where="mid")
        plt.xlabel("Denoising timestep"); plt.ylabel("Full-equivalent step compute ratio")
        plt.ylim(0.0, 1.05); plt.title(f"Static learned compute schedule - {base_mode}")
        plt.grid(True, alpha=0.25); plt.tight_layout()
        fig.savefig(diag / f"static_schedule_compute_ratio_{base_mode}.png", dpi=180); plt.close(fig)
    if actions:
        x = np.arange(1, len(actions) + 1)
        fig = plt.figure(figsize=(10, 5))
        plt.step(x, actions, where="mid")
        plt.xlabel("Denoising timestep"); plt.ylabel("Action index")
        plt.yticks(range(NUM_ACTIONS), [f"A{i}" for i in range(NUM_ACTIONS)])
        plt.title(f"Static learned actions - {base_mode}"); plt.grid(True, alpha=0.25); plt.tight_layout()
        fig.savefig(diag / f"static_schedule_actions_{base_mode}.png", dpi=180); plt.close(fig)
    probs = np.asarray(payload.get("action_probabilities", []), dtype=np.float64)
    if probs.ndim == 2 and probs.shape[1] == NUM_ACTIONS:
        fig = plt.figure(figsize=(12, 5))
        plt.imshow(probs.T, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
        plt.colorbar(label="Validation action probability")
        plt.xlabel("Denoising timestep"); plt.ylabel("Action")
        plt.yticks(range(NUM_ACTIONS), [f"A{i}" for i in range(NUM_ACTIONS)])
        plt.title(f"Best-validation action distribution - {base_mode}"); plt.tight_layout()
        fig.savefig(diag / f"static_source_action_probability_{base_mode}.png", dpi=180); plt.close(fig)


def _load_static_schedule(output_dir: Path, base_mode: str, expected_fp: Dict[str, Any], num_steps: int) -> Dict[str, Any]:
    path = _static_schedule_path(output_dir, base_mode)
    if not path.is_file():
        raise FileNotFoundError(f"缺少 static learned schedule：{path}，请先完成 train。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("algorithm") != ALGO_VERSION:
        raise RuntimeError(f"static schedule算法版本不兼容：{path}")
    if payload.get("policy_fingerprint_sha256") != core.fingerprint_digest(expected_fp):
        raise RuntimeError(f"static schedule fingerprint不兼容：{path}")
    if len(payload.get("actions", [])) != int(num_steps):
        raise RuntimeError(f"static schedule step数不匹配：{path}")
    return payload

# -----------------------------------------------------------------------------
# 训练 / validation
# -----------------------------------------------------------------------------
def _episode_rollout(
    *, pipe, row, args, base_mode: str, base_schedule, variants, qnet, epsilon: float,
    output_dir: Path, forwards_per_step: int, seed_offset: int,
):
    total_layers = len(pipe.transformer.transformer_blocks)
    feasible, reachable, target_units, effective_ratio = build_budget_reachability(
        base_mode=base_mode, variants=variants, total_layers=total_layers,
        num_steps=int(args.num_inference_steps), target_ratio=float(args.compute_ratio),
        min_compute_ratio=float(args.min_compute_ratio), cache_edge_blocks=bool(args.token_cache_edge_blocks),
    )
    image, sample_args, teacher_refs = load_or_build_static(
        pipe=pipe, row=row, args=args, schedule=base_schedule, base_mode=base_mode,
        output_dir=output_dir, forwards_per_step=forwards_per_step,
    )
    runtime = RouterRuntime(
        qnet, args.policy_device, epsilon=epsilon,
        seed=ROUTER_SEED + int(row["sample_index"]) * 1009 + seed_offset,
        feasible=feasible, reachable=reachable, target_units=target_units,
    )
    out_img, elapsed, ctrl = run_router_method(
        pipe=pipe, image=image, sample_args=sample_args, forwards_per_step=forwards_per_step,
        normal_schedule=base_schedule, variants=variants, teacher_refs=teacher_refs,
        base_mode=base_mode, runtime=runtime,
    )
    transitions = runtime.transitions
    mean_score = float(np.mean([t.quality_score for t in transitions])) if transitions else float("inf")
    mean_reward = float(np.mean([t.reward for t in transitions])) if transitions else -float("inf")
    core.release_controller_cuda_state(ctrl)
    core.release_teacher_references(teacher_refs)
    del ctrl, teacher_refs, image, out_img
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return transitions, mean_score, mean_reward, float(elapsed), float(effective_ratio)


def _dqn_update(
    *, online: DuelingQNet, target: DuelingQNet, optimizer, replay: PrioritizedReplay,
    device: str, gradient_step: int, rng: np.random.RandomState,
) -> Tuple[float, int]:
    if len(replay) < max(ROUTER_WARMUP, ROUTER_BATCH_SIZE):
        return 0.0, gradient_step
    losses = []
    for _ in range(ROUTER_GRADIENT_STEPS):
        progress = min(1.0, gradient_step / max(1.0, float(ROUTER_MAX_EPOCHS or 20) * ROUTER_GRADIENT_STEPS))
        beta = ROUTER_PER_BETA_START + (ROUTER_PER_BETA_END - ROUTER_PER_BETA_START) * progress
        batch, idx, weights_np = replay.sample(ROUTER_BATCH_SIZE, beta, rng)
        s = torch.stack([t.state for t in batch]).to(device)
        a = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
        r = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        ns = torch.stack([t.next_state for t in batch]).to(device)
        done = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
        mask = torch.stack([t.next_valid_mask for t in batch]).to(device)
        w = torch.tensor(weights_np, dtype=torch.float32, device=device)
        q = online(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            nq_online = online(ns)
            nq_online = nq_online.masked_fill(~mask, -1e30)
            next_a = nq_online.argmax(dim=1)
            nq_target = target(ns).gather(1, next_a[:, None]).squeeze(1)
            gamma_n = float(ROUTER_GAMMA) ** int(ROUTER_N_STEP)
            y = r + (1.0 - done) * gamma_n * nq_target
        td = y - q
        loss = (w * F.smooth_l1_loss(q, y, reduction="none")).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
        optimizer.step()
        replay.update_priorities(idx, (td.detach().abs().cpu().numpy() + 1e-5).tolist())
        gradient_step += 1
        if gradient_step % ROUTER_TARGET_UPDATE == 0:
            target.load_state_dict(online.state_dict())
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0, gradient_step


def train_one_base_router(
    *, base_mode: str, pipe, train_rows, args, full_schedule, blueprint_schedule, policy_path: Path
) -> None:
    assert base_mode in {"full", "blueprint"}
    total_layers = len(pipe.transformer.transformer_blocks)
    variants = build_router_schedule_variants(full_schedule, blueprint_schedule, total_layers)
    base_schedule = full_schedule if base_mode == "full" else blueprint_schedule
    n_total = len(train_rows)
    if n_total < 2: raise RuntimeError("Router 至少需要2个 train 样本。")
    val_n = min(ROUTER_VAL_COUNT, max(1, n_total // 4))
    fit_rows = list(train_rows[:-val_n])
    val_rows = list(train_rows[-val_n:])
    fp = router_config_dict(args, base_mode, variants, train_rows)
    output_dir = policy_path.parent
    state_dir = output_dir / f"_router_state_{base_mode}"
    ckpt_path = state_dir / "latest.pt"
    rollout_root = state_dir / "rollouts"
    val_root = state_dir / "validation"
    state_dir.mkdir(parents=True, exist_ok=True); rollout_root.mkdir(exist_ok=True); val_root.mkdir(exist_ok=True)

    # 算法升级只清 Router 产物，静态 Full/Fixed25 cache 与 Blueprint 不动。
    if core.is_main_process():
        compatible = router_policy_compatible(policy_path, fp)
        ckpt_ok = False
        if ckpt_path.is_file():
            try: ckpt_ok = torch.load(ckpt_path, map_location="cpu").get("rl_fingerprint") == fp
            except Exception: ckpt_ok = False
        if (policy_path.exists() or ckpt_path.exists()) and not compatible and not ckpt_ok:
            if policy_path.exists(): policy_path.unlink()
            if state_dir.exists(): shutil.rmtree(state_dir)
            state_dir.mkdir(parents=True, exist_ok=True); rollout_root.mkdir(exist_ok=True); val_root.mkdir(exist_ok=True)
    core.dist_barrier()

    online = DuelingQNet(STATE_DIM, NUM_ACTIONS, int(args.hidden_dim)).to(args.policy_device)
    target = DuelingQNet(STATE_DIM, NUM_ACTIONS, int(args.hidden_dim)).to(args.policy_device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.AdamW(online.parameters(), lr=ROUTER_LR, weight_decay=1e-5)
    replay = PrioritizedReplay(ROUTER_REPLAY_CAPACITY, ROUTER_PER_ALPHA)
    epoch_start = 0; gradient_step = 0; best_score = float("inf"); best_state = copy.deepcopy(online.state_dict()); no_improve = 0
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    converged = False
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if ckpt.get("rl_fingerprint") == fp:
            online.load_state_dict(ckpt["online"]); target.load_state_dict(ckpt["target"])
            optimizer.load_state_dict(ckpt["optimizer"])
            core._optimizer_state_to_device(optimizer, args.policy_device)
            replay = PrioritizedReplay.from_state_dict(ckpt.get("replay", {}))
            epoch_start = int(ckpt.get("epoch", 0)); gradient_step = int(ckpt.get("gradient_step", 0))
            best_score = float(ckpt.get("best_score", float("inf")))
            best_state = ckpt.get("best_state", copy.deepcopy(online.state_dict()))
            best_epoch = int(ckpt.get("best_epoch", 0))
            history = [dict(x) for x in ckpt.get("history", [])]
            no_improve = int(ckpt.get("no_improve", 0)); converged = bool(ckpt.get("converged", False))
    if converged and router_policy_compatible(policy_path, fp):
        if core.is_main_process():
            print(f"[router:{base_mode}] 已收敛，直接复用。", flush=True)
            _save_training_history(state_dir, history)
            _plot_training_history(state_dir, base_mode, history)
            if not _static_schedule_path(output_dir, base_mode).is_file():
                if best_epoch <= 0:
                    policy_payload = torch.load(policy_path, map_location="cpu")
                    best_epoch = int(policy_payload.get("best_epoch", policy_payload.get("epoch", 0)))
                _derive_static_schedule_from_validation(
                    base_mode=base_mode, state_dir=state_dir, output_dir=output_dir, val_rows=val_rows,
                    best_epoch=best_epoch, args=args, variants=variants, total_layers=total_layers, policy_fp=fp,
                )
        core.dist_barrier(); return

    core.broadcast_model_from_rank0(online)
    target.load_state_dict(online.state_dict())
    forwards_per_step = int(args.forwards_per_step) if args.forwards_per_step is not None else (2 if args.true_cfg_scale > 1.0 else 1)
    epoch = epoch_start
    while True:
        if ROUTER_MAX_EPOCHS > 0 and epoch >= ROUTER_MAX_EPOCHS:
            break
        epsilon = max(ROUTER_EPS_MIN, ROUTER_EPS_START * (ROUTER_EPS_DECAY ** epoch))
        order = list(range(len(fit_rows)))
        random.Random(ROUTER_SEED + epoch * 10007 + (0 if base_mode == "full" else 17)).shuffle(order)
        epoch_dir = rollout_root / f"epoch_{epoch+1:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for pos, idx in enumerate(order):
            p = epoch_dir / f"sample_{int(fit_rows[idx]['sample_index']):05d}.pt"
            if not p.is_file(): pending.append((pos, idx, p))
        local = [x for j, x in enumerate(pending) if j % core.get_world_size() == core.get_rank()]
        for pos, idx, path in local:
            row = fit_rows[idx]
            transitions, mean_score, mean_reward, elapsed, effective_ratio = _episode_rollout(
                pipe=pipe, row=row, args=args, base_mode=base_mode, base_schedule=base_schedule,
                variants=variants, qnet=online, epsilon=epsilon, output_dir=output_dir,
                forwards_per_step=forwards_per_step, seed_offset=epoch * 100000 + pos,
            )
            core.atomic_torch_save(path, {
                "transitions": [serialize_transition(t) for t in transitions],
                "mean_score": mean_score, "mean_reward": mean_reward,
                "elapsed": elapsed, "effective_ratio": effective_ratio,
            })
        core.dist_barrier()

        train_loss = 0.0
        if core.is_main_process():
            for idx in order:
                p = epoch_dir / f"sample_{int(fit_rows[idx]['sample_index']):05d}.pt"
                payload = torch.load(p, map_location="cpu")
                raw = [deserialize_transition(x) for x in payload["transitions"]]
                for t in make_n_step(raw, ROUTER_N_STEP, ROUTER_GAMMA): replay.add(t)
            train_loss, gradient_step = _dqn_update(
                online=online, target=target, optimizer=optimizer, replay=replay,
                device=args.policy_device, gradient_step=gradient_step,
                rng=np.random.RandomState(ROUTER_SEED + epoch * 7919 + (0 if base_mode == "full" else 1000003)),
            )
        core.broadcast_model_from_rank0(online)
        core.dist_barrier()

        # 固定 deterministic holdout；每个 epoch 的模型不同，所以结果放独立 epoch 目录。
        val_epoch_dir = val_root / f"epoch_{epoch+1:03d}"
        val_epoch_dir.mkdir(parents=True, exist_ok=True)
        local_val = [r for i, r in enumerate(val_rows) if i % core.get_world_size() == core.get_rank()]
        for row in local_val:
            p = val_epoch_dir / f"sample_{int(row['sample_index']):05d}.json"
            if p.is_file(): continue
            val_transitions, score, reward, _, effective_ratio = _episode_rollout(
                pipe=pipe, row=row, args=args, base_mode=base_mode, base_schedule=base_schedule,
                variants=variants, qnet=online, epsilon=0.0, output_dir=output_dir,
                forwards_per_step=forwards_per_step, seed_offset=9000000 + epoch,
            )
            core.atomic_write_json(p, {
                "score": score, "reward": reward, "effective_ratio": effective_ratio,
                "actions": [int(t.action) for t in val_transitions],
                "step_compute_ratios": [float(t.step_compute_ratio) for t in val_transitions],
                "step_quality_scores": [float(t.quality_score) for t in val_transitions],
                "step_rewards": [float(t.reward) for t in val_transitions],
            })
        core.dist_barrier()

        if core.is_main_process():
            vals = []
            for row in val_rows:
                p = val_epoch_dir / f"sample_{int(row['sample_index']):05d}.json"
                vals.append(json.loads(p.read_text(encoding="utf-8")))
            val_score = float(np.mean([v["score"] for v in vals]))
            tol = max(ROUTER_VAL_ABS_TOL, ROUTER_VAL_REL_TOL * max(abs(best_score) if math.isfinite(best_score) else 0.0, abs(val_score), 1e-6))
            improved = (not math.isfinite(best_score)) or (val_score < best_score - tol)
            if improved:
                best_score = val_score; best_state = copy.deepcopy(online.state_dict()); no_improve = 0
                best_epoch = epoch + 1
            else:
                no_improve += 1
            converged = (epoch + 1) >= ROUTER_MIN_EPOCHS and no_improve >= ROUTER_PATIENCE
            history.append({
                "epoch": int(epoch + 1), "train_loss": float(train_loss), "val_score": float(val_score),
                "best_score": float(best_score), "epsilon": float(epsilon), "no_improve": int(no_improve),
                "converged": bool(converged), "gradient_step": int(gradient_step), "replay_size": int(len(replay)),
            })
            _save_training_history(state_dir, history)
            _plot_training_history(state_dir, base_mode, history)
            # 用户要求简洁日志：终端只保留 loss + deterministic收敛状态；详细曲线写文件。
            print(
                f"[router:{base_mode}] epoch={epoch+1} loss={train_loss:.6f} "
                f"val_score={val_score:.6f} best={best_score:.6f} "
                f"patience={no_improve}/{ROUTER_PATIENCE} converged={converged}", flush=True,
            )
            core.atomic_torch_save(ckpt_path, {
                "rl_fingerprint": fp, "online": online.state_dict(), "target": target.state_dict(),
                "optimizer": optimizer.state_dict(), "replay": replay.state_dict(),
                "epoch": epoch + 1, "gradient_step": gradient_step,
                "best_score": best_score, "best_state": best_state, "best_epoch": int(best_epoch),
                "history": history, "no_improve": no_improve, "converged": converged,
            })
            if converged:
                online.load_state_dict(best_state)
                save_router_policy(policy_path, online, fp, converged=True, best_score=best_score, epoch=epoch+1, best_epoch=best_epoch)
        # 同步停止标志和 best policy。
        flag_path = state_dir / "sync.json"
        if core.is_main_process(): core.atomic_write_json(flag_path, {"converged": converged})
        core.dist_barrier()
        stop = bool(json.loads(flag_path.read_text(encoding="utf-8"))["converged"])
        if stop: break
        epoch += 1

    if core.is_main_process() and not converged:
        online.load_state_dict(best_state)
        save_router_policy(policy_path, online, fp, converged=True, best_score=best_score, epoch=epoch, best_epoch=best_epoch)
        print(f"[router:{base_mode}] safety_stop best_val_score={best_score:.6f}", flush=True)
    if core.is_main_process():
        if best_epoch <= 0:
            raise RuntimeError(f"router:{base_mode} 没有有效 best_epoch，无法生成 static baseline")
        _derive_static_schedule_from_validation(
            base_mode=base_mode, state_dir=state_dir, output_dir=output_dir, val_rows=val_rows,
            best_epoch=best_epoch, args=args, variants=variants, total_layers=total_layers, policy_fp=fp,
        )
    core.dist_barrier()

# -----------------------------------------------------------------------------
# Eval：global-uniform / static-learned / adaptive-RL 三类公平比较。
# -----------------------------------------------------------------------------
def _mean_branch_value(ctrl, step: int, key: str, default: float = 0.0) -> float:
    vals = []
    for row in getattr(ctrl, "branch_step_rows", []):
        if int(row.get("step_index_0based", -1)) != int(step):
            continue
        value = row.get(key)
        if value is not None:
            try:
                value = float(value)
                if math.isfinite(value): vals.append(value)
            except Exception:
                pass
    return float(np.mean(vals)) if vals else float(default)


def _controller_actual_global_proxy(ctrl) -> float:
    vals=[]
    for row in getattr(ctrl, "branch_step_rows", []):
        value=row.get("image_token_compute_fraction_proxy")
        if value is not None:
            try:
                v=float(value)
                if math.isfinite(v): vals.append(v)
            except Exception:
                pass
    return float(np.mean(vals)) if vals else float("nan")


def _step_plan_payload(ctrl, step: int) -> Dict[str, Any]:
    plan = getattr(ctrl, "step_plans", {}).get((int(step), 0), {})
    counts = {int(k): int(v) for k, v in plan.get("counts", {}).items()}
    edge = sorted(int(v) for v in plan.get("edge_layers", set()))
    return {
        "token_count": int(plan.get("token_count", 0) or 0),
        "desired_total": int(plan.get("desired_total", 0) or 0),
        "counts": counts,
        "edge_layers": edge,
    }


def _write_router_decisions(path: Path, transitions: Sequence[RouteTransition], ctrl, target_units: int) -> None:
    rows: List[Dict[str, Any]] = []
    remaining = int(target_units)
    for t in transitions:
        step = int(t.step_index)
        remaining -= int(ACTION_UNITS[int(t.action)])
        item = ctrl.schedule[step]
        executed = sorted(int(v) for v in item.get("executed_blocks_0based", []))
        risk = np.log1p(np.maximum(_risk_vector(item, ctrl.total_layers), 0.0))
        plan = _step_plan_payload(ctrl, step)
        row: Dict[str, Any] = {
            "step": step + 1,
            "action": int(t.action), "action_name": ACTION_NAMES[int(t.action)],
            "schedule_mode": str(item.get("mode", "")),
            "step_compute_ratio": float(t.step_compute_ratio),
            "actual_step_compute_fraction_proxy": _mean_branch_value(ctrl, step, "image_token_compute_fraction_proxy", float(t.step_compute_ratio)),
            "remaining_budget_units_after_step": int(remaining),
            "remaining_budget_fraction_after_step": float(remaining / max(1, target_units)),
            "executed_block_count": len(executed),
            "skipped_block_count": int(ctrl.total_layers - len(executed)),
            "executed_blocks_0based": json.dumps(executed, separators=(",", ":")),
            "token_count_per_block": int(plan["token_count"]),
            "desired_image_token_block_units": int(plan["desired_total"]),
            "token_compute_counts_by_block": json.dumps(plan["counts"], sort_keys=True, separators=(",", ":")),
            "edge_full_blocks_0based": json.dumps(plan["edge_layers"], separators=(",", ":")),
            "risk_mean_log1p": float(np.mean(risk)) if risk.size else 0.0,
            "risk_max_log1p": float(np.max(risk)) if risk.size else 0.0,
            "risk_p90_log1p": float(np.quantile(risk, 0.90)) if risk.size else 0.0,
            "quality_score": float(t.quality_score), "reward": float(t.reward),
        }
        state = t.state.detach().cpu().numpy().tolist()
        for i, name in enumerate(STATE_NAMES):
            row[f"state_{name}"] = float(state[i]) if i < len(state) else float("nan")
        rows.append(row)
    if rows:
        core.atomic_write_csv(path, list(rows[0].keys()), rows)


def _write_uniform_decisions(path: Path, ctrl: FixedRatioController, ratio_plan: Sequence[float], target_ratio: float) -> None:
    rows: List[Dict[str, Any]] = []
    remaining = float(target_ratio) * len(ratio_plan)
    for step, ratio in enumerate(ratio_plan):
        remaining -= float(ratio)
        item = ctrl.schedule[step]
        executed = sorted(int(v) for v in item.get("executed_blocks_0based", []))
        plan = _step_plan_payload(ctrl, step)
        rows.append({
            "step": step + 1, "schedule_mode": str(item.get("mode", "")),
            "step_compute_ratio": float(ratio),
            "actual_step_compute_fraction_proxy": _mean_branch_value(ctrl, step, "image_token_compute_fraction_proxy", float(ratio)),
            "remaining_full_step_equivalent_budget_after_step": float(remaining),
            "executed_block_count": len(executed),
            "skipped_block_count": int(ctrl.total_layers - len(executed)),
            "executed_blocks_0based": json.dumps(executed, separators=(",", ":")),
            "token_count_per_block": int(plan["token_count"]),
            "desired_image_token_block_units": int(plan["desired_total"]),
            "token_compute_counts_by_block": json.dumps(plan["counts"], sort_keys=True, separators=(",", ":")),
            "teacher_quality_score": _mean_branch_value(ctrl, step, "score", 0.0),
        })
    if rows:
        core.atomic_write_csv(path, list(rows[0].keys()), rows)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file(): return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_uniform_plan_artifact(output_dir: Path, base_mode: str, ratio_plan: Sequence[float], schedule, total_layers: int) -> None:
    rows=[]
    cumulative=0.0
    for step, ratio in enumerate(ratio_plan):
        cumulative += float(ratio)
        executed=len(schedule[step].get("executed_blocks_0based", []))
        rows.append({
            "step":step+1,"step_compute_ratio":float(ratio),"cumulative_full_step_equivalent":float(cumulative),
            "executed_block_count":int(executed),"executed_block_fraction":float(executed/max(1,total_layers)),
        })
    if rows:
        core.atomic_write_csv(output_dir/f"uniform_global25_plan_{base_mode}.csv",list(rows[0].keys()),rows)
        core.atomic_write_json(output_dir/f"uniform_global25_plan_{base_mode}.json",{
            "base_mode":base_mode,"global_compute_ratio":float(sum(ratio_plan)/max(1,len(ratio_plan))),"rows":rows,
        })
    plt=_plot_module()
    if plt is None or not rows:return
    diag=output_dir/"diagnostics";diag.mkdir(parents=True,exist_ok=True)
    x=np.arange(1,len(rows)+1)
    fig=plt.figure(figsize=(10,5))
    plt.plot(x,[r["step_compute_ratio"] for r in rows],marker="o",markersize=3)
    plt.xlabel("Denoising timestep");plt.ylabel("Full-equivalent compute ratio")
    plt.ylim(0.0,1.05);plt.title(f"Global-uniform compute plan - {base_mode}")
    plt.grid(True,alpha=0.25);plt.tight_layout()
    fig.savefig(diag/f"uniform_global25_plan_{base_mode}.png",dpi=180);plt.close(fig)


def _plot_action_diagnostics(output_dir: Path, eval_rows: Sequence[Dict[str, Any]], base_mode: str, static_actions: Sequence[int]) -> None:
    plt = _plot_module()
    if plt is None: return
    diag = output_dir / "diagnostics"; diag.mkdir(parents=True, exist_ok=True)
    all_rows: List[List[Dict[str, str]]] = []
    for row in eval_rows:
        sd = output_dir / "eval_samples" / f"sample_{int(row['sample_index']):05d}"
        data = _read_csv_rows(sd / f"decisions_{base_mode}_rl25.csv")
        if data: all_rows.append(data)
    if not all_rows: return
    num_steps = max(len(x) for x in all_rows)
    counts = np.zeros((NUM_ACTIONS, num_steps), dtype=np.float64)
    ratios = [[] for _ in range(num_steps)]
    remains = [[] for _ in range(num_steps)]
    blocks = [[] for _ in range(num_steps)]
    disagreements = np.zeros(num_steps, dtype=np.float64)
    totals = np.zeros(num_steps, dtype=np.float64)
    for rows in all_rows:
        for i, r in enumerate(rows):
            a = int(r["action"]); counts[a, i] += 1.0
            ratios[i].append(float(r["step_compute_ratio"]))
            remains[i].append(float(r["remaining_budget_fraction_after_step"]))
            blocks[i].append(float(r["executed_block_count"]))
            totals[i] += 1.0
            if i < len(static_actions) and a != int(static_actions[i]): disagreements[i] += 1.0
    probs = counts / np.maximum(1.0, counts.sum(axis=0, keepdims=True))
    x = np.arange(1, num_steps + 1)

    fig = plt.figure(figsize=(12, 5))
    plt.imshow(probs, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Action probability")
    plt.xlabel("Denoising timestep"); plt.ylabel("Action")
    plt.yticks(range(NUM_ACTIONS), [f"A{i}" for i in range(NUM_ACTIONS)])
    plt.title(f"Deterministic action probability - {base_mode}"); plt.tight_layout()
    fig.savefig(diag / f"action_probability_heatmap_{base_mode}.png", dpi=180); plt.close(fig)

    mean_ratio = np.asarray([np.mean(v) if v else np.nan for v in ratios], dtype=np.float64)
    std_ratio = np.asarray([np.std(v) if v else np.nan for v in ratios], dtype=np.float64)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, mean_ratio, marker="o", markersize=3, label="RL mean")
    plt.fill_between(x, mean_ratio - std_ratio, mean_ratio + std_ratio, alpha=0.2)
    if len(static_actions) == num_steps:
        plt.step(x, [float(ACTION_STEP_RATIOS[int(a)]) for a in static_actions], where="mid", linestyle="--", label="static learned")
    plt.xlabel("Denoising timestep"); plt.ylabel("Full-equivalent compute ratio")
    plt.ylim(0.0, 1.05); plt.title(f"Timestep compute allocation - {base_mode}")
    plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
    fig.savefig(diag / f"mean_step_compute_ratio_{base_mode}.png", dpi=180); plt.close(fig)

    entropy = -np.sum(np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-12)), 0.0), axis=0)
    dis = disagreements / np.maximum(1.0, totals)
    analysis_rows = []
    for i in range(num_steps):
        ar = {
            "step": i + 1,
            "mean_compute_ratio": float(mean_ratio[i]),
            "std_compute_ratio": float(std_ratio[i]),
            "action_entropy_nats": float(entropy[i]),
            "rl_static_disagreement_rate": float(dis[i]),
            "mean_remaining_budget_fraction": float(np.mean(remains[i])) if remains[i] else float("nan"),
            "mean_executed_block_count": float(np.mean(blocks[i])) if blocks[i] else float("nan"),
            "static_action": int(static_actions[i]) if i < len(static_actions) else -1,
        }
        for a in range(NUM_ACTIONS): ar[f"p_A{a}"] = float(probs[a, i])
        analysis_rows.append(ar)
    if analysis_rows:
        core.atomic_write_csv(output_dir / f"action_analysis_{base_mode}.csv", list(analysis_rows[0].keys()), analysis_rows)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, entropy, marker="o", markersize=3)
    plt.xlabel("Denoising timestep"); plt.ylabel("Action entropy (nats)")
    plt.title(f"Sample adaptivity / action entropy - {base_mode}")
    plt.grid(True, alpha=0.25); plt.tight_layout()
    fig.savefig(diag / f"action_entropy_{base_mode}.png", dpi=180); plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, dis, marker="o", markersize=3)
    plt.xlabel("Denoising timestep"); plt.ylabel("RL vs static disagreement rate")
    plt.ylim(0.0, 1.0); plt.title(f"Adaptive deviation from static schedule - {base_mode}")
    plt.grid(True, alpha=0.25); plt.tight_layout()
    fig.savefig(diag / f"rl_static_disagreement_{base_mode}.png", dpi=180); plt.close(fig)

    mean_remain = [np.mean(v) if v else np.nan for v in remains]
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, mean_remain, marker="o", markersize=3)
    plt.xlabel("Denoising timestep"); plt.ylabel("Remaining global budget fraction")
    plt.ylim(0.0, 1.05); plt.title(f"Budget consumption trajectory - {base_mode}")
    plt.grid(True, alpha=0.25); plt.tight_layout()
    fig.savefig(diag / f"remaining_budget_{base_mode}.png", dpi=180); plt.close(fig)

    if base_mode == "blueprint":
        mean_blocks = [np.mean(v) if v else np.nan for v in blocks]
        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, mean_blocks, marker="o", markersize=3)
        plt.xlabel("Denoising timestep"); plt.ylabel("Executed Block count")
        plt.title("Blueprint RL executed Blocks by timestep"); plt.grid(True, alpha=0.25); plt.tight_layout()
        fig.savefig(diag / "executed_block_count_blueprint_rl.png", dpi=180); plt.close(fig)


def _plot_summary_diagnostics(output_dir: Path, summary: Sequence[Dict[str, Any]], rows_out: Sequence[Dict[str, Any]]) -> None:
    plt = _plot_module()
    if plt is None: return
    diag = output_dir / "diagnostics"; diag.mkdir(parents=True, exist_ok=True)
    finite = [x for x in summary if math.isfinite(float(x["psnr_vs_full_dense"]))]
    if finite:
        fig = plt.figure(figsize=(8, 6))
        for x in finite:
            px=float(x["measured_speedup_vs_full_dense"]); py=float(x["psnr_vs_full_dense"])
            plt.scatter([px], [py], s=55); plt.annotate(str(x["method"]), (px, py), xytext=(4,4), textcoords="offset points", fontsize=8)
        plt.xlabel("Measured speedup vs Full Dense"); plt.ylabel("PSNR vs Full Dense")
        plt.title("Speed-quality tradeoff (PSNR)"); plt.grid(True, alpha=0.25); plt.tight_layout()
        fig.savefig(diag / "speed_psnr_pareto.png", dpi=180); plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        for x in summary:
            px=float(x["measured_speedup_vs_full_dense"]); py=float(x["ssim_vs_full_dense"])
            plt.scatter([px], [py], s=55); plt.annotate(str(x["method"]), (px, py), xytext=(4,4), textcoords="offset points", fontsize=8)
        plt.xlabel("Measured speedup vs Full Dense"); plt.ylabel("SSIM vs Full Dense")
        plt.title("Speed-quality tradeoff (SSIM)"); plt.grid(True, alpha=0.25); plt.tight_layout()
        fig.savefig(diag / "speed_ssim_pareto.png", dpi=180); plt.close(fig)

    comparisons = [
        ("full_rl25", "full_uniform25"), ("blueprint_rl25", "blueprint_uniform25"),
        ("full_rl25", "full_static25"), ("blueprint_rl25", "blueprint_static25"),
    ]
    paired_rows: List[Dict[str, Any]] = []
    for method, baseline in comparisons:
        gains=[]
        for r in rows_out:
            if f"{method}_psnr" in r and f"{baseline}_psnr" in r:
                gain=float(r[f"{method}_psnr"])-float(r[f"{baseline}_psnr"]); gains.append(gain)
                paired_rows.append({"sample_index":int(r["sample_index"]),"method":method,"baseline":baseline,"psnr_gain_db":gain})
        if gains:
            fig = plt.figure(figsize=(8, 5))
            plt.hist(gains, bins=min(20, max(5, len(gains))))
            plt.axvline(float(np.mean(gains)), linestyle="--", label=f"mean={np.mean(gains):.3f} dB")
            plt.xlabel("Paired PSNR gain (dB)"); plt.ylabel("Sample count")
            plt.title(f"{method} vs {baseline}"); plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
            fig.savefig(diag / f"paired_psnr_gain_{method}_vs_{baseline}.png", dpi=180); plt.close(fig)
    if paired_rows:
        core.atomic_write_csv(output_dir / "paired_psnr_improvements.csv", list(paired_rows[0].keys()), paired_rows)


def evaluate_eightway_router(
    *, pipe, eval_rows, train_rows, args, full_schedule, blueprint_schedule,
    full_policy_path: Path, blueprint_policy_path: Path, output_dir: Path,
) -> None:
    total_layers = len(pipe.transformer.transformer_blocks)
    variants = build_router_schedule_variants(full_schedule, blueprint_schedule, total_layers)
    full_fp = router_config_dict(args, "full", variants, train_rows)
    bp_fp = router_config_dict(args, "blueprint", variants, train_rows)
    if not router_policy_compatible(full_policy_path, full_fp) or not router_policy_compatible(blueprint_policy_path, bp_fp):
        raise FileNotFoundError("eval 需要当前 v10 Router 的 full/blueprint 两套 policy；请先完成 train。")
    full_q = load_router_policy(full_policy_path, full_fp, args.policy_device)
    bp_q = load_router_policy(blueprint_policy_path, bp_fp, args.policy_device)
    full_digest = core.fingerprint_digest(full_fp); bp_digest = core.fingerprint_digest(bp_fp)
    forwards_per_step = int(args.forwards_per_step) if args.forwards_per_step is not None else (2 if args.true_cfg_scale > 1.0 else 1)

    full_uniform_plan, full_uniform_eff = build_uniform_ratio_plan(
        schedule=full_schedule, total_layers=total_layers, num_steps=int(args.num_inference_steps),
        target_ratio=float(args.compute_ratio), min_compute_ratio=float(args.min_compute_ratio),
        cache_edge_blocks=bool(args.token_cache_edge_blocks),
    )
    bp_uniform_plan, bp_uniform_eff = build_uniform_ratio_plan(
        schedule=blueprint_schedule, total_layers=total_layers, num_steps=int(args.num_inference_steps),
        target_ratio=float(args.compute_ratio), min_compute_ratio=float(args.min_compute_ratio),
        cache_edge_blocks=bool(args.token_cache_edge_blocks),
    )
    full_static = _load_static_schedule(output_dir, "full", full_fp, int(args.num_inference_steps))
    bp_static = _load_static_schedule(output_dir, "blueprint", bp_fp, int(args.num_inference_steps))
    if core.is_main_process():
        _write_uniform_plan_artifact(output_dir, "full", full_uniform_plan, full_schedule, total_layers)
        _write_uniform_plan_artifact(output_dir, "blueprint", bp_uniform_plan, blueprint_schedule, total_layers)
    core.dist_barrier()

    methods = [
        "full_dense", "blueprint_only",
        "full_uniform25", "full_static25", "full_rl25",
        "blueprint_uniform25", "blueprint_static25", "blueprint_rl25",
    ]
    eval_root = output_dir / "eval_samples"; eval_root.mkdir(parents=True, exist_ok=True)

    pending=[]
    for i,row in enumerate(eval_rows):
        sd=eval_root/f"sample_{int(row['sample_index']):05d}"; rec=core._load_eval_record(sd)
        done=(
            rec.get("evaluation_algorithm_version")==ALGO_VERSION
            and rec.get("full_rl_fingerprint_sha256")==full_digest
            and rec.get("blueprint_rl_fingerprint_sha256")==bp_digest
            and all((sd/f"{m}.png").is_file() for m in methods)
            and all((m=="full_dense" or f"{m}_psnr" in rec) for m in methods)
        )
        if not done: pending.append((i,row))
    local=[x for j,x in enumerate(pending) if j%core.get_world_size()==core.get_rank()]
    if core.is_main_process(): print(f"[eval:v10] pending={len(pending)}/{len(eval_rows)}", flush=True)

    for eval_index,row in local:
        sample_index=int(row["sample_index"]); sample_args=core.make_sample_args(args,row)
        image=core.load_input_image(row["image_path"])
        sd=eval_root/f"sample_{sample_index:05d}"; sd.mkdir(parents=True,exist_ok=True)
        rec=core._load_eval_record(sd)
        if not (sd/"input.png").is_file(): core.save_image_atomic(image,sd/"input.png")

        with _quiet_context():
            generated_full,generated_full_time,teacher_refs=core.run_full_teacher(pipe,image,sample_args,forwards_per_step)
        if (sd/"full_dense.png").is_file(): full_img=core._load_png(sd/"full_dense.png")
        else: full_img=generated_full; core.save_image_atomic(full_img,sd/"full_dense.png")
        rec["full_dense_elapsed"]=float(generated_full_time)

        if not (sd/"blueprint_only.png").is_file() or "blueprint_only_psnr" not in rec:
            with _quiet_context(): img,tm,ctrl=core.run_blueprint_only(pipe,image,sample_args,forwards_per_step,blueprint_schedule,teacher_refs)
            core.save_image_atomic(img,sd/"blueprint_only.png"); m=core.metric_row(full_img,img)
            rec.update({"blueprint_only_elapsed":float(tm),"blueprint_only_psnr":float(m["psnr"]),"blueprint_only_ssim":float(m["ssim"])})
            core.release_controller_cuda_state(ctrl); del ctrl,img

        # 公平 global-uniform baselines：固定 Full/Normal Blueprint Block schedule，总 episode proxy 与 RL 一样。
        for base_mode,schedule,ratio_plan,eff in [
            ("full",full_schedule,full_uniform_plan,full_uniform_eff),
            ("blueprint",blueprint_schedule,bp_uniform_plan,bp_uniform_eff),
        ]:
            name=f"{base_mode}_uniform25"
            with _quiet_context():
                img,tm,ctrl=run_uniform_method(pipe=pipe,image=image,sample_args=sample_args,forwards_per_step=forwards_per_step,
                    schedule=schedule,teacher_refs=teacher_refs,step_ratio_plan=ratio_plan)
            core.save_image_atomic(img,sd/f"{name}.png"); m=core.metric_row(full_img,img)
            _write_uniform_decisions(sd/f"decisions_{name}.csv",ctrl,ratio_plan,float(args.compute_ratio))
            rec.update({f"{name}_elapsed":float(tm),f"{name}_psnr":float(m["psnr"]),f"{name}_ssim":float(m["ssim"]),
                        f"{name}_planned_global_compute_ratio":float(eff),
                        f"{name}_actual_global_compute_fraction_proxy":_controller_actual_global_proxy(ctrl)})
            core.release_controller_cuda_state(ctrl); del ctrl,img

        # static learned 与 adaptive RL：相同动作集合、相同硬预算，区别只有是否读取当前 sample 的动态 state。
        for base_mode,qnet,policy_fp,digest,static_payload in [
            ("full",full_q,full_fp,full_digest,full_static),
            ("blueprint",bp_q,bp_fp,bp_digest,bp_static),
        ]:
            feasible,reachable,target_units,eff=build_budget_reachability(
                base_mode=base_mode,variants=variants,total_layers=total_layers,num_steps=int(args.num_inference_steps),
                target_ratio=float(args.compute_ratio),min_compute_ratio=float(args.min_compute_ratio),
                cache_edge_blocks=bool(args.token_cache_edge_blocks))
            normal_schedule=blueprint_schedule if base_mode=="blueprint" else full_schedule

            static_runtime=StaticSequenceRuntime(actions=static_payload["actions"],feasible=feasible,reachable=reachable,target_units=target_units)
            img,tm,ctrl=run_router_method(pipe=pipe,image=image,sample_args=sample_args,forwards_per_step=forwards_per_step,
                normal_schedule=normal_schedule,variants=variants,teacher_refs=teacher_refs,base_mode=base_mode,runtime=static_runtime)
            name=f"{base_mode}_static25"; core.save_image_atomic(img,sd/f"{name}.png"); m=core.metric_row(full_img,img)
            _write_router_decisions(sd/f"decisions_{name}.csv",static_runtime.transitions,ctrl,target_units)
            rec.update({f"{name}_elapsed":float(tm),f"{name}_psnr":float(m["psnr"]),f"{name}_ssim":float(m["ssim"]),
                        f"{name}_planned_global_compute_ratio":float(eff),
                        f"{name}_actual_global_compute_fraction_proxy":_controller_actual_global_proxy(ctrl)})
            core.release_controller_cuda_state(ctrl); del ctrl,img,static_runtime

            runtime=RouterRuntime(qnet,args.policy_device,epsilon=0.0,seed=ROUTER_SEED+sample_index,
                feasible=feasible,reachable=reachable,target_units=target_units)
            img,tm,ctrl=run_router_method(pipe=pipe,image=image,sample_args=sample_args,forwards_per_step=forwards_per_step,
                normal_schedule=normal_schedule,variants=variants,teacher_refs=teacher_refs,base_mode=base_mode,runtime=runtime)
            name=f"{base_mode}_rl25"; core.save_image_atomic(img,sd/f"{name}.png"); m=core.metric_row(full_img,img)
            _write_router_decisions(sd/f"decisions_{name}.csv",runtime.transitions,ctrl,target_units)
            rec.update({f"{name}_elapsed":float(tm),f"{name}_psnr":float(m["psnr"]),f"{name}_ssim":float(m["ssim"]),
                        f"{name}_planned_global_compute_ratio":float(eff),
                        f"{name}_actual_global_compute_fraction_proxy":_controller_actual_global_proxy(ctrl)})
            core.release_controller_cuda_state(ctrl); del ctrl,img,runtime

        rec.update({
            "sample_index":sample_index,"prompt_id":str(row["prompt_id"]),"image_path":str(row["image_path"]),
            "generation_seed":int(row["generation_seed"]),"rl_algorithm_version":ALGO_VERSION,
            "evaluation_algorithm_version":ALGO_VERSION,
            "full_rl_fingerprint_sha256":full_digest,"blueprint_rl_fingerprint_sha256":bp_digest,
            "teacher_free_policy_observation":True,
        })
        core.atomic_write_json(sd/"record.json",rec); core.atomic_write_csv(sd/"metrics.csv",list(rec.keys()),[rec]); core.touch_done(sd/"DONE_V10")
        core.release_teacher_references(teacher_refs); del teacher_refs,image,generated_full,full_img
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"[eval:v10] {eval_index+1}/{len(eval_rows)} sample={sample_index} done",flush=True)
    core.dist_barrier()

    if core.is_main_process():
        rows_out=[]; sums={m:{"elapsed":0.0,"psnr":0.0,"ssim":0.0,"actual_proxy":0.0,"actual_proxy_count":0,"count":0} for m in methods}
        for row in eval_rows:
            sd=eval_root/f"sample_{int(row['sample_index']):05d}"; rec=core._load_eval_record(sd)
            if rec.get("evaluation_algorithm_version")!=ALGO_VERSION or rec.get("full_rl_fingerprint_sha256")!=full_digest or rec.get("blueprint_rl_fingerprint_sha256")!=bp_digest:
                raise RuntimeError(f"eval v10结果未完成：{sd}")
            rows_out.append(rec)
            for m in methods:
                sums[m]["elapsed"]+=float(rec[f"{m}_elapsed"]); sums[m]["count"]+=1
                if m!="full_dense":
                    sums[m]["psnr"]+=float(rec[f"{m}_psnr"]); sums[m]["ssim"]+=float(rec[f"{m}_ssim"])
                ap_key=f"{m}_actual_global_compute_fraction_proxy"
                if ap_key in rec and math.isfinite(float(rec[ap_key])):
                    sums[m]["actual_proxy"]+=float(rec[ap_key]); sums[m]["actual_proxy_count"]+=1
        full_time=sums["full_dense"]["elapsed"]/max(1,sums["full_dense"]["count"])
        bp_only_ratio=float(sum(len(blueprint_schedule[t].get("executed_blocks_0based",[])) for t in blueprint_schedule)/(max(1,len(blueprint_schedule))*max(1,total_layers)))
        summary=[]
        for m in methods:
            c=max(1,sums[m]["count"]); mt=sums[m]["elapsed"]/c
            ps=float("inf") if m=="full_dense" else sums[m]["psnr"]/c
            ss=1.0 if m=="full_dense" else sums[m]["ssim"]/c
            if m=="full_dense": planned=1.0
            elif m=="blueprint_only": planned=bp_only_ratio
            else: planned=float(args.compute_ratio)
            actual_proxy=(sums[m]["actual_proxy"]/sums[m]["actual_proxy_count"]) if sums[m]["actual_proxy_count"] else (1.0 if m=="full_dense" else float("nan"))
            summary.append({"method":m,"completed":sums[m]["count"],"mean_elapsed_seconds":mt,
                "measured_speedup_vs_full_dense":full_time/max(mt,1e-9),"psnr_vs_full_dense":ps,"ssim_vs_full_dense":ss,
                "planned_global_compute_ratio":planned,"mean_actual_global_compute_fraction_proxy":actual_proxy,
                "rl_algorithm_version":ALGO_VERSION if ("rl25" in m or "static25" in m) else ""})
        if rows_out:
            fields=[]
            for r in rows_out:
                for k in r:
                    if k not in fields: fields.append(k)
            core.atomic_write_csv(output_dir/"eightway_per_sample.csv",fields,rows_out)
        core.atomic_write_csv(output_dir/"eightway_summary.csv",list(summary[0].keys()),summary)
        core.atomic_write_json(output_dir/"eightway_summary.json",summary)

        _plot_action_diagnostics(output_dir,eval_rows,"full",full_static["actions"])
        _plot_action_diagnostics(output_dir,eval_rows,"blueprint",bp_static["actions"])
        _plot_summary_diagnostics(output_dir,summary,rows_out)

        print("\n========== v10 八组结果汇总 ==========",flush=True)
        for x in summary:
            print(f"{x['method']}: speedup={x['measured_speedup_vs_full_dense']:.4f} PSNR={x['psnr_vs_full_dense']} SSIM={x['ssim_vs_full_dense']}",flush=True)

# -----------------------------------------------------------------------------
# Monkey patch core main：其它 Blueprint/manifest/模型加载逻辑完全复用。
# -----------------------------------------------------------------------------
core.train_one_base = train_one_base_router
core.evaluate_sixway = evaluate_eightway_router
core.rl_fingerprint = router_fingerprint

if __name__ == "__main__":
    core.main()
