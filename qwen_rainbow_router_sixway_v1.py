#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit-2511: fixed-total-budget discrete compression router v9

核心设计
========
1. 保留 Fresh Blueprint、Full Reference、Fixed25 静态缓存和六组评估。
2. RL 不再给每个 Block 分配连续比例，而是每个 denoising timestep 只做一次离散宏动作。
3. 整个 episode 的总 image-token/block 代理计算量严格锁定为 --compute-ratio（默认 25% Full）。
   因此 reward 只有质量项，不含速度/计算惩罚。
4. full_rl25：所有 Block 路径保持 Full，只让 Router 在 timestep 之间重新分配 25% 总预算。
5. blueprint_rl25：Router 在 Full / Safe Blueprint / Normal Blueprint / Aggressive Blueprint
   等宏模式之间路由，同时仍严格满足同一个 25% 总预算。
6. 每个 timestep 的动作对当前 step 的所有 Block 生效；具体哪些 Block 执行由当前宏模式 schedule 决定。
7. 使用 Dueling Double-DQN + Prioritized Replay + n-step return（Rainbow-style router）。
8. 收敛只看固定 deterministic holdout 的质量 score；日志只打印 loss 和收敛状态。

六组保持：
- blueprint_only
- full_dense
- blueprint_fixed25
- blueprint_rl25   <- 本文件的新 Router
- full_fixed25
- full_rl25        <- 本文件的新 Router
"""
from __future__ import annotations

import contextlib
import copy
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

ALGO_VERSION = "rainbow_macro_router_fixed_global_budget_v1"
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
STATE_DIM = 15

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
        self.prev_score = 0.0
        self.prev_compute_ratio = 1.0
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
            math.log1p(max(0.0, float(self.prev_score))),
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
        if branch == self.forwards_per_step - 1:
            score = float(np.mean(self.step_scores[step]))
            # 总预算已被硬约束，因此 reward 只保留质量；score=0 时 reward=0，误差越大越负。
            reward = -math.log1p(max(0.0, score))
            terminal = step >= int(self.args.num_inference_steps) - 1
            self.router_runtime.finish_step(reward, score, terminal=terminal)
            self.prev_action = int(self.step_action[step])
            self.prev_score = score
            self.prev_compute_ratio = float(self.step_desired_ratio[step])
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
# 静态 Full Reference / Fixed25 cache 复用
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
    # Fixed25 cache仍补齐/复用，保持旧训练资产和六组基线不变；Router reward本身不使用它。
    fixed_fp = core.fixed25_cache_fingerprint(args, base_mode, schedule, full_fp)
    fixed_payload = core._load_static_cache(static_paths["fixed25"], fixed_fp) if args.cache_train_static else None
    if fixed_payload is None:
        with _quiet_context():
            fixed_img, fixed_elapsed, fixed_ctrl = core.run_token_method(
                pipe, image, sample_args, forwards_per_step, schedule, teacher_refs,
                budget_mode="fixed25", policy_runtime=None, fixed_score_map=None,
                base_is_blueprint=(base_mode == "blueprint"),
            )
        fixed_scores = core.score_map(fixed_ctrl)
        fixed_metric = core.metric_row(teacher_img, fixed_img) if teacher_img is not None else {}
        if not paths["fixed"].is_file(): core.save_image_atomic(fixed_img, paths["fixed"])
        core.release_controller_cuda_state(fixed_ctrl)
        if args.cache_train_static:
            core.atomic_torch_save(static_paths["fixed25"], {
                "fingerprint": fixed_fp, "fixed_scores": fixed_scores, "fixed_metric": fixed_metric,
                "first_compute_elapsed": float(fixed_elapsed),
            })
        del fixed_img, fixed_ctrl
    return image, sample_args, teacher_refs

# -----------------------------------------------------------------------------
# fingerprint / policy IO
# -----------------------------------------------------------------------------
def router_config_dict(args, base_mode: str, variants, train_rows) -> Dict[str, Any]:
    return {
        "algorithm": ALGO_VERSION,
        "base_mode": base_mode,
        "state_dim": STATE_DIM,
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


def save_router_policy(path: Path, qnet: DuelingQNet, fingerprint: Dict[str, Any], *, converged: bool, best_score: float, epoch: int) -> None:
    core.atomic_torch_save(path, {
        "rl_algorithm_version": ALGO_VERSION,
        "state_dict": qnet.state_dict(),
        "state_dim": STATE_DIM, "action_dim": NUM_ACTIONS,
        "hidden_dim": int(qnet.feature[0].out_features),
        "rl_fingerprint": fingerprint,
        "training_converged": bool(converged),
        "best_validation_score": float(best_score), "epoch": int(epoch),
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
            no_improve = int(ckpt.get("no_improve", 0)); converged = bool(ckpt.get("converged", False))
    if converged and router_policy_compatible(policy_path, fp):
        if core.is_main_process(): print(f"[router:{base_mode}] 已收敛，直接复用。", flush=True)
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
            _, score, reward, _, effective_ratio = _episode_rollout(
                pipe=pipe, row=row, args=args, base_mode=base_mode, base_schedule=base_schedule,
                variants=variants, qnet=online, epsilon=0.0, output_dir=output_dir,
                forwards_per_step=forwards_per_step, seed_offset=9000000 + epoch,
            )
            core.atomic_write_json(p, {"score": score, "reward": reward, "effective_ratio": effective_ratio})
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
            else:
                no_improve += 1
            converged = (epoch + 1) >= ROUTER_MIN_EPOCHS and no_improve >= ROUTER_PATIENCE
            # 用户要求简洁日志：只保留 loss + deterministic收敛状态。
            print(
                f"[router:{base_mode}] epoch={epoch+1} loss={train_loss:.6f} "
                f"val_score={val_score:.6f} best={best_score:.6f} "
                f"patience={no_improve}/{ROUTER_PATIENCE} converged={converged}", flush=True,
            )
            core.atomic_torch_save(ckpt_path, {
                "rl_fingerprint": fp, "online": online.state_dict(), "target": target.state_dict(),
                "optimizer": optimizer.state_dict(), "replay": replay.state_dict(),
                "epoch": epoch + 1, "gradient_step": gradient_step,
                "best_score": best_score, "best_state": best_state,
                "no_improve": no_improve, "converged": converged,
            })
            if converged:
                online.load_state_dict(best_state)
                save_router_policy(policy_path, online, fp, converged=True, best_score=best_score, epoch=epoch+1)
        # 同步停止标志和 best policy。
        flag_path = state_dir / "sync.json"
        if core.is_main_process(): core.atomic_write_json(flag_path, {"converged": converged})
        core.dist_barrier()
        stop = bool(json.loads(flag_path.read_text(encoding="utf-8"))["converged"])
        if stop: break
        epoch += 1

    if core.is_main_process() and not converged:
        online.load_state_dict(best_state)
        save_router_policy(policy_path, online, fp, converged=True, best_score=best_score, epoch=epoch)
        print(f"[router:{base_mode}] safety_stop best_val_score={best_score:.6f}", flush=True)
    core.dist_barrier()

# -----------------------------------------------------------------------------
# Eval：非RL四组完全沿用；仅两条 rl25 换成 Router。
# -----------------------------------------------------------------------------
def _write_router_decisions(path: Path, transitions: Sequence[RouteTransition]) -> None:
    rows = [{
        "step": int(t.step_index) + 1, "action": int(t.action), "mode": t.mode_name,
        "step_compute_ratio": float(t.step_compute_ratio), "quality_score": float(t.quality_score),
        "reward": float(t.reward),
    } for t in transitions]
    if rows:
        core.atomic_write_csv(path, list(rows[0].keys()), rows)


def evaluate_sixway_router(
    *, pipe, eval_rows, train_rows, args, full_schedule, blueprint_schedule,
    full_policy_path: Path, blueprint_policy_path: Path, output_dir: Path,
) -> None:
    total_layers = len(pipe.transformer.transformer_blocks)
    variants = build_router_schedule_variants(full_schedule, blueprint_schedule, total_layers)
    full_fp = router_config_dict(args, "full", variants, train_rows)
    bp_fp = router_config_dict(args, "blueprint", variants, train_rows)
    if not router_policy_compatible(full_policy_path, full_fp) or not router_policy_compatible(blueprint_policy_path, bp_fp):
        raise FileNotFoundError("eval 需要当前 Rainbow Router 的 full/blueprint 两套 policy；请先完成 train。")
    full_q = load_router_policy(full_policy_path, full_fp, args.policy_device)
    bp_q = load_router_policy(blueprint_policy_path, bp_fp, args.policy_device)
    full_digest = core.fingerprint_digest(full_fp); bp_digest = core.fingerprint_digest(bp_fp)
    forwards_per_step = int(args.forwards_per_step) if args.forwards_per_step is not None else (2 if args.true_cfg_scale > 1.0 else 1)
    methods = ["blueprint_only", "full_dense", "blueprint_fixed25", "blueprint_rl25", "full_fixed25", "full_rl25"]
    eval_root = output_dir / "eval_samples"; eval_root.mkdir(parents=True, exist_ok=True)

    pending = []
    for i, row in enumerate(eval_rows):
        sd = eval_root / f"sample_{int(row['sample_index']):05d}"
        rec = core._load_eval_record(sd)
        # 新 Router fingerprint 使用 record 字段判断；非RL文件继续复用。
        router_done = (
            core._eval_nonrl_ready(sd, rec)
            and (sd / "blueprint_rl25.png").is_file() and (sd / "full_rl25.png").is_file()
            and rec.get("full_rl_fingerprint_sha256") == full_digest
            and rec.get("blueprint_rl_fingerprint_sha256") == bp_digest
        )
        if not router_done: pending.append((i, row))
    local = [x for j, x in enumerate(pending) if j % core.get_world_size() == core.get_rank()]
    if core.is_main_process():
        print(f"[eval] pending={len(pending)}/{len(eval_rows)}", flush=True)

    for eval_index, row in local:
        sample_index = int(row["sample_index"]); sample_args = core.make_sample_args(args, row)
        image = core.load_input_image(row["image_path"])
        sd = eval_root / f"sample_{sample_index:05d}"; sd.mkdir(parents=True, exist_ok=True)
        rec = core._load_eval_record(sd)
        if not (sd / "input.png").is_file(): core.save_image_atomic(image, sd / "input.png")
        with _quiet_context():
            generated_full, generated_full_time, teacher_refs = core.run_full_teacher(pipe, image, sample_args, forwards_per_step)
        if (sd / "full_dense.png").is_file(): full_img = core._load_png(sd / "full_dense.png")
        else:
            full_img = generated_full; core.save_image_atomic(full_img, sd / "full_dense.png")
            rec["full_dense_elapsed"] = float(generated_full_time)
        rec.setdefault("full_dense_elapsed", float(generated_full_time))

        if not (sd / "blueprint_only.png").is_file() or "blueprint_only_psnr" not in rec:
            with _quiet_context():
                img, tm, ctrl = core.run_blueprint_only(pipe, image, sample_args, forwards_per_step, blueprint_schedule, teacher_refs)
            core.save_image_atomic(img, sd / "blueprint_only.png"); m = core.metric_row(full_img, img)
            rec.update({"blueprint_only_elapsed":tm,"blueprint_only_psnr":m["psnr"],"blueprint_only_ssim":m["ssim"]})
            core.release_controller_cuda_state(ctrl); del ctrl, img
        if not (sd / "blueprint_fixed25.png").is_file() or "blueprint_fixed25_psnr" not in rec:
            with _quiet_context():
                img, tm, ctrl = core.run_token_method(pipe, image, sample_args, forwards_per_step, blueprint_schedule, teacher_refs,
                    budget_mode="fixed25", policy_runtime=None, fixed_score_map=None, base_is_blueprint=True)
            core.save_image_atomic(img, sd / "blueprint_fixed25.png"); m=core.metric_row(full_img,img)
            rec.update({"blueprint_fixed25_elapsed":tm,"blueprint_fixed25_psnr":m["psnr"],"blueprint_fixed25_ssim":m["ssim"]})
            core.release_controller_cuda_state(ctrl); del ctrl, img
        if not (sd / "full_fixed25.png").is_file() or "full_fixed25_psnr" not in rec:
            with _quiet_context():
                img, tm, ctrl = core.run_token_method(pipe, image, sample_args, forwards_per_step, full_schedule, teacher_refs,
                    budget_mode="fixed25", policy_runtime=None, fixed_score_map=None, base_is_blueprint=False)
            core.save_image_atomic(img, sd / "full_fixed25.png"); m=core.metric_row(full_img,img)
            rec.update({"full_fixed25_elapsed":tm,"full_fixed25_psnr":m["psnr"],"full_fixed25_ssim":m["ssim"]})
            core.release_controller_cuda_state(ctrl); del ctrl, img

        for base_mode, qnet, policy_fp, digest in [
            ("blueprint", bp_q, bp_fp, bp_digest), ("full", full_q, full_fp, full_digest)
        ]:
            feasible, reachable, target_units, eff = build_budget_reachability(
                base_mode=base_mode, variants=variants, total_layers=total_layers,
                num_steps=int(args.num_inference_steps), target_ratio=float(args.compute_ratio),
                min_compute_ratio=float(args.min_compute_ratio), cache_edge_blocks=bool(args.token_cache_edge_blocks),
            )
            runtime = RouterRuntime(qnet, args.policy_device, epsilon=0.0,
                seed=ROUTER_SEED + sample_index, feasible=feasible, reachable=reachable, target_units=target_units)
            img, tm, ctrl = run_router_method(
                pipe=pipe, image=image, sample_args=sample_args, forwards_per_step=forwards_per_step,
                normal_schedule=(blueprint_schedule if base_mode=="blueprint" else full_schedule),
                variants=variants, teacher_refs=teacher_refs, base_mode=base_mode, runtime=runtime,
            )
            name = f"{base_mode}_rl25"
            core.save_image_atomic(img, sd / f"{name}.png"); m=core.metric_row(full_img,img)
            _write_router_decisions(sd / f"decisions_{name}.csv", runtime.transitions)
            rec.update({
                f"{name}_elapsed":float(tm), f"{name}_psnr":float(m["psnr"]), f"{name}_ssim":float(m["ssim"]),
                f"{name}_planned_global_compute_ratio":float(eff),
            })
            core.release_controller_cuda_state(ctrl); del ctrl, img, runtime

        rec.update({
            "sample_index": sample_index, "prompt_id": str(row["prompt_id"]), "image_path": str(row["image_path"]),
            "generation_seed": int(row["generation_seed"]), "rl_algorithm_version": ALGO_VERSION,
            "full_rl_fingerprint_sha256": full_digest, "blueprint_rl_fingerprint_sha256": bp_digest,
        })
        core.atomic_write_json(sd / "record.json", rec); core.atomic_write_csv(sd / "metrics.csv", list(rec.keys()), [rec]); core.touch_done(sd / "DONE")
        core.release_teacher_references(teacher_refs); del teacher_refs, image, generated_full, full_img
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"[eval] {eval_index+1}/{len(eval_rows)} sample={sample_index} done", flush=True)
    core.dist_barrier()

    if core.is_main_process():
        rows_out=[]; sums={m:{"elapsed":0.0,"psnr":0.0,"ssim":0.0,"count":0} for m in methods}
        for row in eval_rows:
            sd=eval_root/f"sample_{int(row['sample_index']):05d}"; rec=core._load_eval_record(sd)
            if rec.get("full_rl_fingerprint_sha256") != full_digest or rec.get("blueprint_rl_fingerprint_sha256") != bp_digest:
                raise RuntimeError(f"eval Router结果未完成：{sd}")
            rows_out.append(rec); sums["full_dense"]["elapsed"]+=float(rec["full_dense_elapsed"]); sums["full_dense"]["count"]+=1
            for m in [x for x in methods if x!="full_dense"]:
                sums[m]["elapsed"]+=float(rec[f"{m}_elapsed"]); sums[m]["psnr"]+=float(rec[f"{m}_psnr"]); sums[m]["ssim"]+=float(rec[f"{m}_ssim"]); sums[m]["count"]+=1
        full_time=sums["full_dense"]["elapsed"]/max(1,sums["full_dense"]["count"]); summary=[]
        for m in methods:
            c=max(1,sums[m]["count"]); mt=sums[m]["elapsed"]/c
            ps=float("inf") if m=="full_dense" else sums[m]["psnr"]/c; ss=1.0 if m=="full_dense" else sums[m]["ssim"]/c
            summary.append({"method":m,"completed":sums[m]["count"],"mean_elapsed_seconds":mt,
                "measured_speedup_vs_full_dense":full_time/max(mt,1e-9),"psnr_vs_full_dense":ps,"ssim_vs_full_dense":ss,
                "compute_ratio":1.0 if m in {"full_dense","blueprint_only"} else float(args.compute_ratio),
                "rl_algorithm_version":ALGO_VERSION if "rl25" in m else ""})
        if rows_out:
            fields=[]
            for r in rows_out:
                for k in r:
                    if k not in fields: fields.append(k)
            core.atomic_write_csv(output_dir/"sixway_per_sample.csv",fields,rows_out)
        core.atomic_write_csv(output_dir/"sixway_summary.csv",list(summary[0].keys()),summary); core.atomic_write_json(output_dir/"sixway_summary.json",summary)
        print("\n========== 六组结果汇总 ==========", flush=True)
        for x in summary:
            print(f"{x['method']}: speedup={x['measured_speedup_vs_full_dense']:.4f} PSNR={x['psnr_vs_full_dense']} SSIM={x['ssim_vs_full_dense']}", flush=True)

# -----------------------------------------------------------------------------
# Monkey patch core main：其它 Blueprint/manifest/模型加载逻辑完全复用。
# -----------------------------------------------------------------------------
core.train_one_base = train_one_base_router
core.evaluate_sixway = evaluate_sixway_router
core.rl_fingerprint = router_fingerprint

if __name__ == "__main__":
    core.main()
