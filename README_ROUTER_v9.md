# Qwen Rainbow-style Macro Router v9

## 目标

整条 denoising 轨迹的总 image-token/block 代理计算量固定为 `--compute-ratio`，默认 0.25。
RL 不再优化速度项，只在固定总预算下寻找质量最优的 timestep 压缩模式组合。

每个 timestep 只决策一次，当前 step 的所有 Block 共用这一个宏动作。

### blueprint_rl25 动作

- A0: Full，step 总计算 100%
- A1: Safe Blueprint，step 总计算 50%
- A2: Normal Blueprint，step 总计算 35%
- A3: Normal Blueprint，step 总计算 25%
- A4: Aggressive Blueprint，step 总计算 15%
- A5: Aggressive Blueprint，step 总计算 5%

注意：这里百分比是**当前整个 timestep 相对 Full 的总 image-token/block 代理计算比例**，不是“执行 Block 内 token 比例”。
Router 根据所选 Block schedule 自动反解执行 Block 内应该算多少 token。

### full_rl25

Block 始终为 Full 60 层，动作只改变当前 timestep 的 token 总预算。这作为不使用 Blueprint block routing 的 RL 对照。

## 总预算为何能精确 25%

动作成本均为 5% Full 的整数倍：`[100, 50, 35, 25, 15, 5]%`。
训练/推理前会做 backward reachability DP。当前动作只有在“选完它以后，剩余 step 仍能精确凑到总预算”时才允许选择。
因此 50 step、25% 目标最终一定闭合到 25% 计划预算。

## Reward

没有速度项：

`reward_t = -log(1 + teacher_quality_score_t)`

其中 `teacher_quality_score` 仍是已有 Full Reference 的 noise/image-token/text-token 加权误差。

## 算法

Dueling Double-DQN + Prioritized Replay + n-step return（Rainbow-style）。

## 六组

1. blueprint_only
2. full_dense
3. blueprint_fixed25
4. blueprint_rl25（新 Router）
5. full_fixed25
6. full_rl25（新 Router）

## 日志

训练阶段默认隐藏逐 step controller 日志，只显示：

`[router:full] epoch=... loss=... val_score=... best=... patience=... converged=...`

以及 blueprint 对应行。

## 运行

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MODE=all \
bash run_qwen_rainbow_router_sixway_v1.sh
```

只训练：`MODE=train`；只六组评估：`MODE=eval`。

默认会复用已有 manifest、Fresh Blueprint、Full Reference、Fixed25 静态缓存。算法指纹变化只会让旧 RL policy/state 失效。
