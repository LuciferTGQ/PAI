# 图表解释草稿

## 图1：项目整体技术路线

Question: 历史工业轨迹如何经过世界模型、策略优化和仿真评价形成完整流程？

Observation: 历史状态与动作先用于训练世界模型；模型冻结后分别支持规划方法和MB-PPO，并在NeoRL仿真环境中评价累计奖励。

Interpretation: 该图帮助读者快速理解项目做了什么，以及预测模型如何服务于控制策略。

Limitation: 流程图只描述技术路线，不提供性能证据。

## 图2：模型架构与数据规模

Question: 模型架构和数据规模如何共同影响单步与多步预测？

Observation: M100由GRU取得最低综合递归误差；M1000和M10000由Transformer-2L胜出。M10000的Transformer-2L在H5/H10/H20上的平均NRMSE为0.283。

Interpretation: 模型架构与数据规模存在交互，单步预测排序与多步递归预测排序也不完全相同。

Limitation: 每格来自一次正式训练结果，不能替代多次独立训练的鲁棒性研究。

## 图3：多步递归预测误差累积

Question: 单步NRMSE为何不足以描述需要递归使用的世界模型？

Observation: M1000中五个模型的单步误差接近，但H20和H50明显分离；Transformer-2L在H5--H20综合最低，Transformer-4L在H50更低。

Interpretation: 小的单步偏差会在递归更新中传播，短中期稳定性需要直接评价。

Limitation: 连接线只辅助观察，不表示未测步数的插值性能。

## 图4：代表性H50轨迹

Question: 代表性连续轨迹上的预测偏差如何随时间演化？

Observation: 所示轨迹的综合NRMSE为0.123，速度、疲劳度和能耗呈现不同的偏差累积方式。

Interpretation: 整体NRMSE背后可能包含不同的变量级误差，工业控制需要同时监控关键变量。

Limitation: 一条代表轨迹不能替代全部验证轨迹的统计结果。

## 图5：不同数据规模下的策略比较

Question: 每个数据规模固定世界模型后，哪种策略取得更高累计奖励？

Observation: M100的CEM最高（-274,238）；M1000和M10000的MPPI最高。M100的iCEM约为-2,646,910，明显超出主图范围。

Interpretation: 规划器能力与世界模型质量共同影响真实控制效果，优化器可能放大低质量模型的误差。

Limitation: 该图不能推出iCEM普遍不适合工业过程。

## 图6：世界模型与后续策略

Question: 预测误差排序是否完全决定后续控制排序？

Observation: M1000中MPPI在Transformer-4L上最高（-216,331），MB-PPO在LSTM上最高（-277,127），统一预测评价则选择Transformer-2L。

Interpretation: 不同策略可能对世界模型的误差结构具有不同敏感性。

Limitation: 这是当前五种模型、两种策略和一个数据规模上的观察，不是普遍规律。

## 图7：最终NeoRL控制结果

Question: 最终系统能否提高1000步累计奖励？

Observation: 原始行为数据轨迹均值为-282,885；三个最终系统相对BC的改善分别为14,229、68,282和64,535。

Interpretation: 基于世界模型的策略优化在独立仿真条件下保持了长期回报提升。

Limitation: 原始行为来自数据集已有轨迹，与仿真策略的评价来源不同；结果也只覆盖当前NeoRL工业控制基准。

## 图8：MB-PPO行为KL消融

Question: 行为KL约束是否限制MB-PPO偏离历史行为分布？

Observation: 不加KL的平均累计奖励约为-883,709，加KL后为-284,715；训练记录显示无KL策略与行为分布的偏离显著扩大。

Interpretation: 在当前消融设置中，行为KL约束显著降低了模型利用风险。

Limitation: 消融只覆盖一个世界模型和一次策略训练，不能量化最佳KL系数或保证所有场景都有效。
