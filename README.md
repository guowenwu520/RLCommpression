# Qwen Rainbow Macro Router v10

这一版只围绕 v9 已经暴露出来的三个问题做修改，不改 Fresh Blueprint 本身的 Safe / Normal / Aggressive 定义，也不改 Rainbow 的主要训练算法。

## 1. 去掉 Teacher 信息进入 policy state

v9 的 state 包含 `prev_score`，它是当前样本压缩结果和 Full Teacher 的误差，因此真实部署时不可获得。

v10 删除 `prev_score`，替换为两个 **Teacher-free、当前压缩模型自己可观测** 的动态量：

- `prev_token_change_mean_log1p`
- `prev_token_change_max_log1p`

它们来自现有 token selector 已经计算的 `selection_score_all_mean`：当前 image hidden 与该 token 最近一次真实计算来源 hidden 的归一化变化。Full Teacher 仍只用于训练 reward 和离线评估，不再进入 policy observation。

因此 state 现在是 16 维：timestep、剩余预算、上一步动作/预算、Blueprint Block/risk、Block cache age，以及上述两个 teacher-free token drift 特征。

## 2. 修复 Blueprint Fixed25 不同预算的问题

旧 `blueprint_fixed25` 的 25% 是“Blueprint 已执行 Block 内部各算 25% token”，所以相对 Full episode 的总代理可能远低于 25%。它不能与 `blueprint_rl25` 的 global 25% 做公平比较。

v10 主结果中不再使用旧 `blueprint_fixed25`，改成：

- `full_uniform25`
- `blueprint_uniform25`

两者都把 **整条 episode 相对 Full 的总代理严格设为 `--compute-ratio`（默认 25%）**。

`blueprint_uniform25` 始终使用原始 Normal Blueprint schedule，不动态改变 Block 模式。由于 step0 必须 Full，程序用 water-filling / box projection 自动求其它 step 的静态比例；如果某个 Blueprint step 能承载的最大预算较低，会在该 step 截断，并把剩余预算平均补到其它 step，最终全局仍严格为 25%。

## 3. 区分“固定 timestep 规律”与“sample-adaptive routing”

每个 epoch 的 deterministic validation 现在额外保存每个 timestep 的 action。

训练结束后，程序读取 **best validation epoch** 的 action 分布，在完全相同的 action feasibility 和 25% 硬预算下做一次 DP，得到一条所有样本共享的：

- `full_static25`
- `blueprint_static25`

Static baseline 不是简单逐 step 取众数，而是最大化各 step empirical action probability，同时保证整条轨迹预算能够精确闭合。

最终可比较：

- `uniform25`：没有学习，尽量均匀分预算；
- `static25`：学习到通用 timestep schedule，但所有图片相同；
- `rl25`：真正读取当前样本 teacher-free token drift / cache state 动态决策。

如果 `static25 ≈ rl25`，说明主要收益来自普适 timestep 重要性；如果 `rl25` 明显更好，才说明 sample-adaptive routing 有额外价值。

## 八组主评估

1. `full_dense`
2. `blueprint_only`
3. `full_uniform25`
4. `full_static25`
5. `full_rl25`
6. `blueprint_uniform25`
7. `blueprint_static25`
8. `blueprint_rl25`

其中后六个压缩方法（除 blueprint_only）用于 global 25% 的公平分析。

## Reward

仍然保持 v9 的设计：总预算由硬约束负责，reward 只评价质量。

`reward_t = -log(1 + teacher_quality_score_t)`

Teacher score 仍由 noise / image-token / text-token 误差组合得到。Teacher 只参与训练 reward 和离线指标，不再参与 policy state。

## 自动输出的数据和图

训练：

- `_router_state_full/training_history.csv/json`
- `_router_state_blueprint/training_history.csv/json`
- `diagnostics/convergence_val_score_full.png`
- `diagnostics/convergence_val_score_blueprint.png`
- `diagnostics/training_loss_full.png`
- `diagnostics/training_loss_blueprint.png`
- `diagnostics/epsilon_full.png`
- `diagnostics/epsilon_blueprint.png`

Static schedule：

- `static_learned_schedule_full.csv/json`
- `static_learned_schedule_blueprint.csv/json`
- `diagnostics/static_schedule_compute_ratio_*.png`
- `diagnostics/static_schedule_actions_*.png`

评估：

- `eightway_summary.csv/json`
- `eightway_per_sample.csv`
- `paired_psnr_improvements.csv`
- 每个样本的 `decisions_full_rl25.csv` / `decisions_blueprint_rl25.csv`
- 每个样本的 `decisions_full_static25.csv` / `decisions_blueprint_static25.csv`
- 每个样本的 `decisions_full_uniform25.csv` / `decisions_blueprint_uniform25.csv`
- `action_analysis_full.csv`
- `action_analysis_blueprint.csv`

主要图：

- action probability heatmap
- mean timestep compute ratio + static schedule 对照
- action entropy（衡量 sample adaptivity）
- RL vs static disagreement rate
- remaining budget trajectory
- Blueprint RL executed Block count
- speed-PSNR / speed-SSIM Pareto 图
- RL vs uniform / static 的 paired PSNR gain 分布

Decision CSV 也比 v9 更完整，包含具体执行 Block、每 Block token count、实际 proxy ratio、risk、remaining budget、quality score、reward 以及全部命名 state 特征。

## 运行

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MODE=all \
bash run_qwen_rainbow_router_sixway_v1.sh
```

只训练：`MODE=train`；只评估：`MODE=eval`。

默认继续使用 `/data4/guowenwu/RLCompression`。已有 manifest、Fresh Blueprint 和 Full Reference cache 会继续复用。算法 fingerprint 已更新，所以旧 v9 policy / Router state 不会被误用；旧 fixed25 文件不会被删除。
