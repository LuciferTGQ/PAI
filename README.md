# PAI Industrial World Model

本项目基于 NeoRL Industrial Benchmark（IB）的历史工业轨迹学习状态转移规律，并在冻结的世界模型中优化连续控制策略，最后部署到 NeoRL simulator 评估长期累计奖励。完整流程为：

**历史轨迹 → 世界模型 → 未来状态预测 → 策略优化 → NeoRL simulator 验证**

## 主要成果

- 完成 MLP、GRU、LSTM、Transformer-2L、Transformer-4L 在 M100、M1000、M10000 上的统一 `5 × 3` 比较。
- 世界模型使用 validation-only protocol 选择，不使用 simulator reward 反向挑选模型。
- 比较 CEM、iCEM、MPPI 与 MB-PPO；规划长度消融在固定 M1000 GRU 和 CEM 参数下比较 `H=5/10/20`，最终采用 `H=10`。
- 最终 fresh seeds 100–109 的 1000-step NeoRL simulator 结果如下（奖励越高越好）：

| 数据规模 | 最终系统 | Episode return（mean ± std） | 相对 BC | Win rate |
|---|---|---:|---:|---:|
| M100 | GRU + CEM | -274,181 ± 396 | +14,229 | 100% |
| M1000 | Transformer-2L + MPPI | -220,129 ± 1,699 | +68,282 | 100% |
| M10000 | Transformer-2L + MPPI | -223,876 ± 1,033 | +64,535 | 100% |

上述数值来自 [`outputs/metrics/final_selected_systems_fresh_seeds_summary.csv`](outputs/metrics/final_selected_systems_fresh_seeds_summary.csv)。

## 数据规模

| 名称 | 轨迹数 | 状态转移数 | 每条轨迹长度 |
|---|---:|---:|---:|
| M100 | 100 | 100,000 | 1000 steps |
| M1000 | 1,000 | 1,000,000 | 1000 steps |
| M10000 | 10,000 | 10,000,000 | 1000 steps |

原始数据不提交到 GitHub。文件命名、放置目录和 M10000 memory-map 准备方法见 [`data/README.md`](data/README.md)。

固定 NeoRL 源码位于 `external/NeoRL` 后，可分别下载并导出三种规模的数据：

```powershell
python scripts/download_neorl_data.py --scale 100
python scripts/download_neorl_data.py --scale 1000
python scripts/download_neorl_data.py --scale 10000
python scripts/prepare_npz_memmap.py data/raw/ib-medium-10000-train.npz data/cache/ib-medium-10000-train
```

每条下载命令同时生成对应的 10% validation 数据；`--scale 10000` 需要较大的磁盘和内存空间。

## 世界模型

输入为 `30 × 6` 历史工业状态与 3 维连续动作，输出下一帧 6 维状态：

```text
history [B, 30, 6] + action [B, 3] -> next_frame [B, 6]
```

六个状态变量为 setpoint、velocity、gain、shift、fatigue、consumption。模型评价同时包含 one-step NRMSE 和 H5/H10/H20/H50 递归预测误差。`Hk` 指第 k 个递归预测时刻的误差，并非前 k 步误差的算术和。

模型选择规则为：先保留 `one-step NRMSE <= 1.10 × 最优 one-step NRMSE` 的候选，再在候选中最小化 `mean(H5,H10,H20)`；H50仅用于长期稳定性诊断。

## 策略优化

- **CEM**：用精英样本迭代更新动作序列分布。
- **iCEM**：加入 colored noise、精英复用和上一时刻解平移等采样改进。
- **MPPI**：通过路径积分权重更新控制序列。正式入口为 [`src/strategy/reference_mppi.py`](src/strategy/reference_mppi.py)，核心优化使用 `pytorch-mppi`。
- **MB-PPO**：在冻结世界模型中生成 imagined rollout，并用行为 KL 约束策略不要过度偏离离线行为分布。

Horizon 消融使用 CEM 作为代表性规划器，固定 M1000 GRU、population=64、elites=8、iterations=2，仅改变预测长度；5 个独立 simulator 初始条件的结果为：

| Horizon | 1000-step return（mean ± std） | median |
|---:|---:|---:|
| 5 | -280,284 ± 1,052 | -280,283 |
| **10** | **-269,374 ± 487** | **-269,072** |
| 20 | -272,618 ± 490 | -272,625 |

H=10 的平均回报最高，且是 one-standard-error rule 下唯一候选，因此用于 CEM、iCEM、MPPI；MB-PPO 同样使用 10-step imagined rollout，但其含义是训练阶段的模型内轨迹，而不是滚动时域规划。

## 项目结构

```text
src/data/          NeoRL IB 数据加载、轨迹切分与 normalization
src/world_model/   五种世界模型、训练器和冻结模型接口
src/strategy/      BC、CEM、iCEM、MPPI、MB-PPO
src/evaluation/    预测、模型选择、策略矩阵与最终仿真评价
configs/           正式训练和评价配置
scripts/           数据准备、训练与正式实验入口
outputs/metrics/   正式 CSV/JSON 结果与逐步日志
report/            LaTeX 正文、图表生成脚本和 source data
output/pdf/        编译后的最终报告
tests/             轻量单元测试与训练 smoke test
```

## 关键代码位置

| 功能 | 代码位置 |
|---|---|
| 数据加载与轨迹处理 | [`src/data/ib_dataset.py`](src/data/ib_dataset.py) |
| MLP / GRU / LSTM / Transformer | [`src/world_model/model.py`](src/world_model/model.py) |
| World Model 训练与 resume | [`src/world_model/trainer.py`](src/world_model/trainer.py) |
| 冻结 World Model 接口 | [`src/world_model/interface.py`](src/world_model/interface.py) |
| 单步与多步递归评价 | [`src/evaluation/checkpoint_selection.py`](src/evaluation/checkpoint_selection.py) |
| Validation-only checkpoint 选择 | [`src/evaluation/checkpoint_selection.py`](src/evaluation/checkpoint_selection.py) |
| CEM | [`src/strategy/cem_mpc.py`](src/strategy/cem_mpc.py) |
| iCEM | [`src/strategy/icem_mpc.py`](src/strategy/icem_mpc.py) |
| MPPI | [`src/strategy/reference_mppi.py`](src/strategy/reference_mppi.py) |
| MB-PPO + behavior KL | [`src/strategy/mbppo.py`](src/strategy/mbppo.py) |
| Horizon 消融 | [`src/evaluation/cem_horizon.py`](src/evaluation/cem_horizon.py) |
| `3 × 4` Strategy Matrix | [`src/evaluation/main_system_matrix.py`](src/evaluation/main_system_matrix.py) |
| World Model × Strategy | [`src/evaluation/world_model_control_cross.py`](src/evaluation/world_model_control_cross.py) |
| Final NeoRL 评价 | [`src/evaluation/final_selected_systems.py`](src/evaluation/final_selected_systems.py) |

## 实验入口与复现条件

实验环境使用 Python、PyTorch、NeoRL、Gym、NumPy、Pandas、PyYAML、Matplotlib、`colorednoise` 和 `pytorch-mppi`，具体版本范围见 [`requirements.txt`](requirements.txt)。NeoRL simulator 从 `external/NeoRL` 加载；本项目固定 NeoRL commit `717c9a92d5253876f8cb28318ef72e3d5ab05968`，其 OfflineRL 代码固定为 `807933a87f77529f17bd81ac64d717aad89f5cdf`。

各实验组与入口的对应关系如下：

| 实验组 | 入口 |
|---|---|
| NeoRL 数据下载与导出 | `scripts/download_neorl_data.py` |
| 单个 World Model 训练 | `scripts/train_world_model.py` |
| `5 × 3` World Model 训练 | `scripts/run_world_model_matrix.py` |
| 统一 common-validation 评价 | `scripts/evaluate_world_model_5x3.py` |
| Horizon 消融 | `scripts/evaluate_cem_horizons.py` |
| `3 × 4` Strategy Matrix | `scripts/run_main_system_matrix.py` |
| World Model × Strategy | `scripts/run_world_model_control_cross.py` |
| Final fresh-seed evaluation | `scripts/run_final_selected_systems.py` |

所有入口均从 `configs/` 读取参数；训练支持 checkpoint resume。原始数据和模型 checkpoint 不上传，已有正式 CSV/JSON 与最终报告可以直接阅读，不依赖重新运行实验。

## 正式结果与实验报告

- 世界模型完整指标：`outputs/metrics/world_model_5x3_common_validation*`
- Strategy Matrix：`outputs/metrics/main_system_matrix_development*`
- World Model × Strategy：`outputs/metrics/world_model_control_cross_m1000*`
- Final fresh-seed evaluation：`outputs/metrics/final_selected_systems_fresh_seeds*`
- 最终 PDF：[`output/pdf/pai_industrial_world_model_report.pdf`](output/pdf/pai_industrial_world_model_report.pdf)
- LaTeX 与统一图表源：[`report/`](report/)

GitHub 保存正式 CSV/JSON、逐步日志、最终图表和 PDF。原始 NeoRL 数据、缓存及 checkpoint 因体积较大不上传；按配置重新训练后会写入 `outputs/checkpoints/`，正式结果表无需重新训练即可直接查阅和生成报告图表。
