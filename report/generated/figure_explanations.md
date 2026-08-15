# Figure interpretation drafts

## Figure 1 — Overall Framework

Question: 历史工业轨迹如何经过World Model、策略优化和simulator评价形成闭环？

Observation: World Model selection与simulator reward之间存在明确隔离；模型冻结后才进入CEM、iCEM、MPPI或MB-PPO。

Interpretation: 该隔离避免用控制回报反向挑选dynamics模型，保留了模型比较的因果清晰度。

Limitation: 流程图描述实验协议，不提供任何性能证据。

## Figure 2 — Architecture × Dataset Scale

Question: 架构和数据规模如何共同影响单步与多步预测？

Observation: M100由GRU取得最低合规综合rollout误差；M1000和M10000由Transformer-2L胜出。M10000的Transformer-2L mean(H5,H10,H20)为0.283，但不同架构并未随数据量单调改善。

Interpretation: 架构归纳偏置与数据覆盖发生交互；单步与递归rollout关注的误差模式不同。

Limitation: 每格是单训练seed下的validation结果，不能替代多训练seed鲁棒性研究。

## Figure 3 — Multi-step Error Accumulation

Question: one-step NRMSE为何不足以描述递归World Model？

Observation: M1000中五个模型的one-step都约为0.17--0.19，但H20和H50明显分离；Transformer-2L在H5--H20综合最低，而Transformer-4L在H50更低。

Interpretation: 小的单步偏差在闭环递归更新中会以变量相关方式传播，短中期稳定性需要直接评价。

Limitation: 连接线不表示未测horizon处的插值性能。

## Figure 4 — Representative H50 Rollout

Question: 代表性连续轨迹上的预测漂移是什么形态？

Observation: 所示轨迹在101个确定性候选起点中最接近中位H50误差，综合NRMSE为0.123；velocity、fatigue和consumption呈现不同的漂移速度。

Interpretation: 同一个整体NRMSE背后可能包含非常不同的变量级误差，工业控制应监控关键变量而非只看总分。

Limitation: 一条代表轨迹不能替代全体rollout起点的统计结果。

## Figure 5 — Strategy Comparison

Question: 每个数据规模固定其dynamics-selected World Model后，哪种策略更强？

Observation: M100的CEM最高（-274,238）；M1000和M10000的MPPI最高。M100 iCEM约为-2,646,910，出现严重失效。

Interpretation: 规划器能力与模型质量共同决定真实控制效果；优化器可能利用低质量World Model的错误高收益区域。

Limitation: iCEM的toy objective已通过，因此该图不能推出iCEM一般不适合工业过程。

## Figure 6 — World Model × Downstream Strategy

Question: dynamics prediction ranking是否完全决定downstream control ranking？

Observation: M1000中MPPI在Transformer-4L上最高（-216,331），MB-PPO在LSTM上最高（-277,127），而dynamics validation选择Transformer-2L。

Interpretation: planning与learned policy对不同误差结构的敏感性不同，通用预测误差与决策效用并非完全等价。

Limitation: 这是当前五模型、两策略和一个数据规模上的观察，不是普遍定律，也不用于反向修改World Model选择。

## Figure 7 — Final NeoRL Simulator Result

Question: 最终World-Model-based systems能否提高1000步累计奖励？

Observation: Original Behavior数据轨迹均值为-282,885；三个最终系统相对BC的改善分别为14,229、68,282和64,535，且十个最终seed的win rate均为100%。

Interpretation: 经过validation-only模型选择和冻结策略选择后，基于World Model的优化在未使用simulator seeds上保持了长期回报提升。

Limitation: Original Behavior来自数据集已有轨迹，不是与在线系统同seed部署的配对基线；结果也只覆盖NeoRL Industrial Benchmark。

## Figure 8 — MB-PPO KL Ablation

Question: behavior KL是否抑制MB-PPO利用World Model误差？

Observation: 不加KL的平均return约为-883,709，加KL后为-284,715；dense training history显示无KL策略与行为分布的偏离显著扩大。

Interpretation: 在该M1000 GRU消融中，behavior constraint显著降低了model exploitation风险。

Limitation: 消融只覆盖一个World Model和一个训练seed，不能量化最佳KL系数或保证所有场景都有效。
