# PAI Industrial World Model — Codex Master Prompt

你现在是本项目的首席实现 Agent。

当前首轮预计使用：

**GPT-5.6 Sol / High reasoning**

项目时间、计算资源和 Agent 额度有限。必须同时保证：

- 正确；
- 来源可靠；
- 模块清晰；
- 完整跑通；
- 实验可复现；
- 尽量节省 Token；
- 必要时可以被其他 Agent 接手。

核心原则：

> **先确认题目、论文和原作者官方代码已经提供了什么，再实现缺失部分。**
>
> **先完成题目要求的完整基础闭环，再进行模型比较、优化和创新。**
>
> **正确性优先于节省 Token，但禁止重复分析、碎片化测试和过度工程。**

---

# 1. 固定项目位置

唯一正式工作目录：

`G:\PAI`

远程 Git 仓库：

`https://github.com/LuciferTGQ/PAI.git`

用户会提前创建本地 Git 仓库并配置 remote。

开始后首先确认当前目录和 Git：

```bash
git status
git remote -v
git branch --show-current
git log --oneline -8
```

预期 remote：

`https://github.com/LuciferTGQ/PAI.git`

如果 remote 正确，不修改。

如果缺失或不符，不擅自覆盖用户已有配置，记录问题并继续可以安全完成的本地任务。

不要在其他位置创建第二份 PAI 项目。

---

# 2. 已存在的本地权威资料

以下两个文件已经存在：

`G:\PAI\PAI世界模型测试题目.docx`

`G:\PAI\2102.00714v2.pdf`

分别对应：

- 老师提供的项目题目；
- NeoRL 原论文。

**不要移动、重命名或重复复制这两个文件。**

遇到题目要求模糊：

直接重新阅读：

`G:\PAI\PAI世界模型测试题目.docx`

遇到 NeoRL 数据、采集过程、behavior policy、benchmark 定义等问题：

优先阅读：

`G:\PAI\2102.00714v2.pdf`

不要先依赖模型记忆或二手博客。

---

# 3. NeoRL 官方代码当前不在本地

当前不要假设存在：

`G:\PAI\external\NeoRL`

NeoRL 官方源码需要你主动访问官方 GitHub：

`https://github.com/Polixir/NeoRL`

首先阅读官方仓库：

- README；
- benchmark；
- environment；
- dataset相关代码；
- OfflineRL相关代码；
- IB实现；
- reward实现；
- BC实现与配置。

如果后续开发、运行、源码审计需要完整本地代码：

创建：

`G:\PAI\external\`

然后将 **官方 NeoRL repository** clone 到：

`G:\PAI\external\NeoRL`

必须确认 clone 来源确实为：

`https://github.com/Polixir/NeoRL`

记录 exact commit hash。

第三方仓库不是我们的原创源码，不要把其内容混入 `src/`。

---

# 4. 权威资料优先级

遇到任何关于以下内容的不确定问题：

- 任务到底要求什么；
- NeoRL数据是什么；
- IB observation/action含义；
- 数据如何产生；
- trajectory结构；
- train/validation方式；
- behavior policy；
- Low/Medium/High；
- 100/1000/10000；
- reward；
- simulator；
- BC；
- 原始策略；
- benchmark方法；

必须优先检查：

1. `G:\PAI\PAI世界模型测试题目.docx`
2. `G:\PAI\2102.00714v2.pdf`
3. NeoRL 官方 GitHub  
   `https://github.com/Polixir/NeoRL`
4. NeoRL 官方源码 / benchmark
5. benchmark 携带或引用的 OfflineRL / d3pe 等官方实现
6. 如果需要进一步理解 Industrial Benchmark 内部动力学，再查其原始论文和官方资料

禁止仅凭经验猜测。

---

# 5. Source-First / Anti-Hallucination

重要结论应区分来源：

- `[TASK]`
- `[PAPER]`
- `[OFFICIAL CODE]`
- `[VERIFIED LOCALLY]`
- `[ENGINEERING CHOICE]`
- `[UNVERIFIED]`

禁止凭空假设存在：

- behavior policy checkpoint；
- 某个 NeoRL API；
- 某种 dataset shape；
- 某个 reward function；
- 某个 split；
- 某个模型文件；
- 某个官方下载地址。

如果无法确认：

先查本地题目、论文和官方 GitHub。

仍不能确认的关键问题可以记录到：

`docs/OPEN_QUESTIONS.md`

但普通小问题不要为了写文档打断开发。

如果题目、论文、README、官方源码和实际运行出现冲突：

1. 考核要求以题目为准；
2. 当前 API / shape 以当前官方源码 + 本机实际运行为准；
3. 论文背景和定义以论文为准；
4. 我们自己的方法明确作为 Engineering Choice。

重大冲突记录到 `docs/DECISIONS.md`。

---

# 6. 首先完成 Source Audit

正式大量开发前，对题目、NeoRL论文和官方代码进行一次集中审计。

建立：

`docs/SOURCE_AUDIT.md`

至少确认：

## NeoRL

- NeoRL解决什么问题；
- IB为何作为工业环境；
- IB数据如何产生；
- behavior policy如何训练；
- Low / Medium / High是什么；
- 100 / 1000 / 10000是什么；
- train和validation如何生成；
- IB的数据采集与其他环境有什么特殊之处；
- 原始 behavior policy checkpoint 是否公开。

## Industrial Benchmark

确认：

- observation dimension；
- action dimension；
- 单帧维度；
- 180维 observation真实组成；
- history length；
- frame ordering；
- 六个变量含义；
- history window update；
- trajectory length；
- action bounds；
- reward definition。

## 官方 Baseline

检查官方是否已经提供：

- BC；
- CQL；
- MOPO；
- BREMEN；
- MB-PPO；
- 其他 relevant algorithms。

Source Audit完成一次后，后续优先读取该文件，避免重复阅读整篇论文。

新的关键歧义仍需回到一手资料验证。

---

# 7. 官方代码优先，尤其是 BC

如果 NeoRL 原作者已经提供：

- dataset downloader；
- environment；
- simulator；
- reward；
- BC；
- benchmark evaluation；
- model-free/model-based baseline；

优先研究和复用官方实现。

BC必须优先在 NeoRL 官方 benchmark 中寻找并读取：

`benchmark/OfflineRL/offlinerl/algo/modelfree/bc.py`

以及：

`benchmark/OfflineRL/offlinerl/config/algo/bc_config.py`

实际记录：

- actor；
- architecture；
- hidden size；
- number of layers；
- loss；
- optimizer；
- learning rate；
- batch size；
- epochs / steps；
- validation metric；
- action processing。

禁止先自行写一个普通 MLP+MSE，然后称为 NeoRL BC。

---

# 8. 官方代码兼容策略

第一选择：

**官方原代码直接运行。**

如果因为旧 Python / Gym / dependency 导致失败：

先定位真正问题，只做最小 compatibility patch。

重大 patch 才记录到：

`docs/OFFICIAL_CODE_PATCHES.md`

如果官方 BC framework确实无法在当前环境合理运行，可以建立：

`src/strategy/official_bc_compat/`

做 faithful compatible reproduction。

尽量保持官方：

- actor；
- architecture；
- loss；
- optimizer；
- hyperparameters；
- action处理；
- evaluation protocol。

如果发生改变，必须明确说明，不能继续称为“完全原始官方实现”。

---

# 9. 三模块架构

严格按照老师强调的三个模块组织。

## Module A — World Model Construction

只负责：

- 数据；
- 状态转移模型；
- Temporal Transformer；
- 训练；
- one-step prediction；
- multi-step rollout；
- error accumulation。

不负责策略优化。

---

## Module B — Strategy Optimization

接受：

**Frozen World Model**

负责：

- Basic Policy；
- candidate action generation；
- planning；
- strategy optimization。

不能偷偷重训 World Model。

---

## Module C — Simulator Evaluation

负责：

- 加载策略；
- NeoRL IB simulator；
- episode reward；
- 多 seed；
- Original / Basic / World Model Policy comparison；
- 指标与图表。

三个模块必须解耦。

不要把全部功能揉进一个大 `train.py`。

---

# 10. 推荐项目结构

```text
G:\PAI
│
├─ MASTER_PROMPT.md
├─ AGENTS.md
├─ README.md
├─ PAI世界模型测试题目.docx
├─ 2102.00714v2.pdf
│
├─ docs/
│  ├─ SOURCE_AUDIT.md
│  ├─ DATA_AUDIT.md
│  ├─ ENVIRONMENT.md
│  ├─ ARCHITECTURE.md
│  ├─ STATE.md
│  ├─ TODO.md
│  ├─ DECISIONS.md
│  ├─ EXPERIMENTS.md
│  ├─ OPEN_QUESTIONS.md
│  ├─ OFFICIAL_CODE_PATCHES.md
│  ├─ HANDOFF.md
│  └─ FUTURE_WORK.md
│
├─ configs/
├─ src/
│  ├─ data/
│  ├─ world_model/
│  ├─ strategy/
│  ├─ evaluation/
│  └─ utils/
├─ scripts/
├─ tests/
├─ external/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ outputs/
│  ├─ checkpoints/
│  ├─ metrics/
│  ├─ figures/
│  └─ logs/
└─ report/
```

目录只在真正需要时创建。

不要为了满足结构图一次创建大量空文件和无意义抽象。

---

# 11. 环境策略：先检查已有 Conda + CUDA

本机已经存在至少一个以前成功使用 GPU/CUDA/PyTorch 的 Conda 环境。

因此：

> **禁止默认创建新环境。**
>
> **禁止默认重新下载 CUDA。**

首先运行：

```bash
conda env list
```

找到最可能已有可用 GPU/PyTorch 配置的环境。

对候选环境实际确认：

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

同时检查：

- Python；
- numpy；
- scipy；
- pandas；
- gym/gymnasium；
- ray；
- attrdict；
- NeoRL相关依赖。

只集中做一次 Environment Audit。

---

# 12. CUDA判断规则

必须区分：

### NVIDIA Driver

通过：

`nvidia-smi`

### 系统 CUDA Toolkit

如果已有：

`nvcc --version`

### PyTorch CUDA Runtime

通过：

`torch.version.cuda`

和：

`torch.cuda.is_available()`

不能只根据 `nvcc` 判断 PyTorch 是否能使用 GPU。

---

# 13. 环境复用顺序

依次考虑：

## A. 直接复用已有环境

如果：

- GPU正常；
- PyTorch正常；
- Python兼容NeoRL；
- NeoRL依赖不会明显破坏环境；

优先直接复用。

## B. Clone已有环境

如果已有GPU环境很重要，不适合污染：

优先考虑：

```bash
conda create --name <new_env> --clone <working_env>
```

然后在clone中适配NeoRL。

## C. 新建环境

只有当：

- Python版本不兼容；
- Gym / NeoRL依赖严重冲突；
- 安装会破坏用户已有项目；

才创建新环境。

如果新建：

应尽量参考这台机器已经成功运行过的：

- GPU；
- driver；
- PyTorch；
- CUDA runtime组合。

不要盲目追最新版本。

---

# 14. 禁止盲目重新下载完整 CUDA Toolkit

PyTorch预编译包通常携带运行所需 CUDA runtime。

除非确认某个依赖必须：

- 使用 nvcc；
- 编译 CUDA extension；
- source build；

否则不要下载安装新的完整 CUDA Toolkit。

如果必须新增 CUDA 组件：

优先使用与本机已有成功环境兼容的版本。

---

# 15. 环境完成标准

不能因为：

`pip install`

没有报错就认为成功。

必须真实验证：

- PyTorch GPU；
- `import neorl`；
- 创建 IB environment；
- `reset()`；
- `step()`；
- `get_dataset()`。

必要环境信息和最终激活方法记录到：

`docs/ENVIRONMENT.md`

这个文件不需要频繁维护。

环境真正稳定后更新一次即可。

---

# 16. Data Audit

成功运行 NeoRL 后集中完成一次：

`docs/DATA_AUDIT.md`

实际记录：

- obs.shape；
- next_obs.shape；
- action.shape；
- reward.shape；
- done.shape；
- index；
- dtype；
- action bounds；
- transition数量；
- trajectory数量；
- trajectory长度；
- train size；
- validation size。

---

# 17. 30×6结构必须验证

当前存在强先验：

`180 = 30 historical frames × 6 variables`

但禁止仅根据 Prompt 硬编码。

必须通过：

**官方 NeoRL源码 + 实际下载的数据**

双重确认：

- 180真实结构；
- frame ordering；
- latest frame位置；
- 六变量顺序；
- history window update。

重点验证：

`next_obs_t`

是否与：

`obs_t`

共享29帧：

- 删除最老帧；
- 插入1个新帧。

---

# 18. 数据下载

优先使用：

**IB Medium**

计划下载：

- IB-M-100；
- IB-M-1000；
- IB-M-10000。

先：

IB-M-100。

确认正确后：

IB-M-1000。

然后检查：

- G盘剩余空间；
- 官方实际下载文件大小；
- 解压/加载后的实际占用；

空间合理再下载10000。

不要根据估算盲目下载。

数据统一放：

`G:\PAI\data\`

具体目录结构和文件名以 NeoRL 官方 downloader/API 为准。

数据不得提交 Git。

---

# 19. 数据使用策略

开发和完整流水线 smoke：

**IB-M-100**

正式基础实验：

**IB-M-1000**

IB-M-10000：

可以提前缓存，但基础 pipeline 未完成前不要用于正式训练。

---

# 20. World Model V1

第一版只做：

**Temporal Transformer**

目标：

> 先完成题目要求的完整成品。

基础成品前暂不做：

- residual；
- ensemble；
- uncertainty；
- probabilistic dynamics；
- MLP comparison；
- LSTM comparison；
- GRU comparison；
- physics-informed；
- architecture search；
- multi-step training loss。

---

# 21. Temporal Transformer

Temporal Transformer不是另一个全新Transformer算法。

第一版直接使用 PyTorch标准：

`nn.TransformerEncoder`

如果 Data Audit确认：

`obs = 30 frames × 6 features`

则：

每个6维时间帧作为1个 token。

基本结构：

```text
history [B,30,6]
        ↓
Linear frame embedding
        ↓
Temporal / positional encoding
        ↓
Transformer Encoder
        ↓
Temporal representation

current action [B,3]
        ↓
small action embedding

Temporal representation
+
Action representation
        ↓
simple fusion / concat
        ↓
prediction head
        ↓
next frame [B,6]
```

建议第一版小模型：

- d_model = 64或128；
- nhead = 4；
- layers = 2，必要时再增加；
- FFN = 128/256。

所有参数配置化。

不要手写Q/K/V。

当前输入只有过去，因此第一版无需强行做GPT式 causal mask。

---

# 22. World Model预测目标

如果确认 sliding-window：

输入：

`history [30,6] + action [3]`

预测：

`next_frame [6]`

然后使用确定性 history-window update 重构：

`next_obs`

不要让网络重新学习复制其余29个历史frame。

第一版使用：

**Direct next-frame prediction**

暂不做 residual。

允许简单画六变量时序图，但不要基础阶段花大量Token分析残差建模。

---

# 23. Normalization

只根据 training dataset 计算：

- frame/state statistics；
- action statistics；
- target statistics。

Validation不得参与。

保存 normalization statistics，并随 checkpoint 使用。

---

# 24. World Model训练

开发阶段：

IB-M-100。

用较小 epoch / batch 做真正可运行的训练。

目标：

- shape正确；
- loss下降；
- CUDA正常；
- checkpoint save/load正常。

不要第一轮就追求最优指标。

正式实验时：

通过 config 切换到 IB-M-1000。

不要为换数据规模修改核心代码。

---

# 25. One-Step Evaluation

题目要求单步预测精度。

至少输出：

- MSE；
- MAE；
- normalized RMSE或合理尺度无关指标；
- per-variable MSE；
- per-variable MAE。

如果官方reward确认与某些变量直接相关：

重点展示这些变量，例如 fatigue / consumption。

保存：

CSV/JSON + figures。

---

# 26. Multi-Step Evaluation

题目明确要求：

多步预测误差累积。

至少评价 horizon：

- 1；
- 5；
- 10；
- 20；
- 50。

Rollout必须：

1. 真实 history作为起点；
2. World Model预测 next frame；
3. 将预测frame插入下一history；
4. 后续继续使用模型预测state；
5. action使用validation真实action sequence；
6. 不得每步重新注入真实state；
7. 不得跨episode boundary。

输出：

- horizon-error curve；
- selected-variable rollout。

---

# 27. World Model模块完成条件

World Model至少能独立执行：

- train；
- one-step evaluation；
- multi-step evaluation。

并输出：

- checkpoint；
- config；
- normalization；
- metrics；
- figures。

完成后：

**冻结 World Model。**

Strategy模块通过稳定接口调用，例如：

```python
predict_next_frame(history,action)
rollout(history,action_sequence)
```

---

# 28. Basic Policy

基础策略优先：

**NeoRL 官方 BC**

优先原始官方代码。

如确实需要兼容移植：

忠实复现官方方案。

不要自行替换为普通监督 MLP。

---

# 29. Original Behavior Policy

必须调查：

产生 IB-M 数据的原始 SAC Behavior Policy checkpoint 是否公开。

检查：

- NeoRL官方GitHub；
- 官方数据生成代码；
- benchmark；
- 论文；
- 官方引用的资源。

禁止伪造：

loader

或：

checkpoint。

如果没有找到：

明确记录。

评价采用以下优先级：

### Level A

官方 Behavior Policy checkpoint  
→ NeoRL simulator online evaluation。

### Level B

如果checkpoint不可得：

使用dataset完整trajectory的episode return作为：

`Original Behavior Policy — empirical dataset performance`

必须明确：

这不是重新online replay policy得到的结果。

### Level C

NeoRL论文中的reported behavior return

只作为literature reference。

不能冒充我们运行出的实验。

---

# 30. Strategy Optimization

只有：

World Model训练和基础评价完成，

并冻结之后才开始。

先查看 NeoRL 官方是否已有相关 model-based methods：

例如：

- MOPO；
- BREMEN；
- MB-PPO；
- 其他。

理解它们是否：

- 能直接使用我们训练好的 Transformer World Model；
- 要求替换 dynamics；
- 破坏三模块结构；
- 实现成本过高。

不要因为官方有某算法就强行使用。

---

# 31. Strategy V1

如果没有官方方案可以干净调用：

Frozen Transformer World Model，

默认采用经典：

**MPC + CEM**

作为第一版 World-Model-Based Strategy。

它是基础方案，不作为第一版创新点吹嘘。

---

# 32. MPC/CEM流程

当前真实 history：

1. 生成 candidate future action sequences；
2. action遵守 IB官方bounds；
3. Frozen World Model进行multi-step rollout；
4. 根据预测state调用官方reward逻辑；
5. 计算predicted cumulative return；
6. CEM选择elite；
7. 更新action distribution；
8. 重复若干轮；
9. 执行最佳sequence的第一个action；
10. simulator获得真实next state；
11. 下一时刻重新规划。

参数必须配置化：

- horizon；
- population；
- elite ratio；
- iterations；
- discount；
- seed。

开发先使用较小参数保证完整episode能够运行。

---

# 33. Reward

必须先查 NeoRL 官方 reward源码。

如果 reward可以由 predicted next state/frame直接计算：

复用官方 reward逻辑。

第一版不要额外训练 Reward Model。

如果官方源码显示不是这样：

再重新判断。

---

# 34. 最终三策略比较

严格对应题目：

## 1. Original Behavior Policy

NeoRL数据采集策略或其可靠reference。

## 2. Basic Policy

NeoRL官方BC或忠实兼容复现。

## 3. World-Model-Optimized Policy

Temporal Transformer World Model

+

Strategy Optimization。

最终核心评价：

**NeoRL IB simulator episode reward**

正式评价使用多个固定seed。

至少输出：

- episode returns；
- mean；
- std；
- median（方便时）。

如果Original Behavior不是相同形式的online evaluation：

必须明确标注来源差异。

---

# 35. 第一版核心实验图

至少生成：

1. World Model train/validation curve；
2. one-step prediction result；
3. multi-step horizon error curve；
4. selected variable rollout；
5. three-policy episode reward comparison。

---

# 36. 第一版 Definition of Done

只有以下全部完成，才叫基础成品：

### Environment
NeoRL IB simulator真实可运行。

### Data
IB-M-100真实读取并完成Data Audit。

### World Model
Transformer能训练、保存、加载。

### Prediction
one-step + multi-step评价完成。

### Basic Policy
官方BC或faithful compatible reproduction可运行。

### Strategy
World Model Strategy能够使用Frozen World Model。

### Simulator
BC和World Model Strategy能够完成真实episode。

### Comparison
获得：

- Original Behavior reference；
- BC performance；
- World Model Strategy performance；
- 最终reward comparison。

---

# 37. 基础成品前禁止扩张

基础闭环完成前暂不做：

- MLP vs LSTM vs GRU vs Transformer；
- residual；
- ensemble；
- uncertainty；
- probabilistic World Model；
- behavior-constrained planner；
- physics-informed；
- Low/Medium/High系统横向比较；
- extensive hyperparameter tuning。

这些只作为：

`docs/FUTURE_WORK.md`

中的候选方向。

基础项目完成后再优先考虑：

1. MLP / GRU/LSTM / Transformer；
2. Direct vs Residual；
3. planner horizon；
4. ensemble / uncertainty；
5. conservative / behavior-constrained planning。

---

# 38. 测试策略：模块级测试

不要每写一个小函数就跑测试。

采用：

**Boundary-Based Testing**

### Data模块较完整
→ 一次Data Audit。

### Dataset + Model + Trainer较完整
→ 一次tiny training smoke。

### One-step + Multi-step完成
→ 一次integration evaluation。

### Planner主体完成
→ 一次短episode。

### 全部模块连接
→ 一次end-to-end smoke。

如果测试失败：

再缩小范围做针对性调试。

不能为了节省Token跳过关键正确性验证。

---

# 39. Git版本控制

Git用于：

**稳定恢复 + 大模块checkpoint。**

`.gitignore` 排除：

- datasets；
- checkpoints；
- environment；
- cache；
- large logs；
- credentials；
- secrets；
- 第三方仓库内部 `.git`。

建议大致形成：

1. scaffold + Source/Environment/Data Audit；
2. Transformer trainable；
3. one-step + multi-step；
4. official BC integrated；
5. World Model Strategy；
6. full pipeline；
7. formal experiment/report。

复杂阶段允许增加少量中间稳定checkpoint。

不要制造大量：

`fix`
`fix2`
`tmp`
`test`

commit。

---

# 40. Remote Push

Remote：

`https://github.com/LuciferTGQ/PAI.git`

不要每个commit都push。

建议主要在：

1. Environment/Data稳定；
2. World Model完成；
3. Strategy完成；
4. Full Pipeline完成；
5. Final版本；

进行push。

总体约4~6个主要remote checkpoint。

高风险重构、dependency migration或官方源码patch前：

先形成稳定local commit；

必要时提前push。

Push前检查：

```bash
git status
git diff --stat
```

确认没有数据集、checkpoint、credential、环境、cache和大日志。

Remote失败不阻塞本地研究。

---

# 41. 项目状态文档采用低频维护

本项目预计主要由同一个 Agent 连续完成。

因此：

> **不要为了未来可能发生的 Agent 切换而频繁维护 Markdown 状态文件。**

日常项目真实状态主要由：

**代码 + Git commit + config + 实验产物**

体现。

---

# 42. AGENTS.md

首次建立后保持稳定。

只记录：

- Source First；
- Official Code First；
- 三模块原则；
- 环境复用；
- 防幻觉；
- 基础成品优先等长期规则。

长期规则未改变就不要更新。

---

# 43. TODO.md

只维护：

**Phase / 大模块级别**

例如：

```text
Source/Data        DONE
World Model        DOING
Prediction Eval    TODO
Official BC        TODO
Strategy           TODO
Final Evaluation   TODO
```

不要记录每个小函数、小修复、小测试。

不需要每完成一两个功能就更新。

---

# 44. STATE.md

平时不强制更新。

只在以下情况下更新：

- 一个较大Phase完成；
- 用户明确要求整理当前进度；
- 准备切换模型/Agent；
- 当前Agent额度即将耗尽；
- 当前阶段复杂，需要保存恢复点；
- 高风险重大修改前后。

---

# 45. HANDOFF.md

**平时不要维护。**

只有真正准备让另一个 Agent 接手时：

才创建或集中更新。

交接时至少记录：

- Current Phase；
- 已完成模块；
- Conda环境；
- Python/PyTorch/CUDA；
- 已验证命令；
- 当前dataset；
- 最新稳定Git commit；
- 当前主要metrics；
- known issues；
- next priority task。

---

# 46. DECISIONS.md

只有发生重大技术决定时更新，例如：

- World Model input/output定义；
- BC采用方式；
- Original Behavior Policy评价方式；
- Strategy算法选择；
- 官方代码重大兼容修改。

普通实现细节不记录。

---

# 47. EXPERIMENTS.md

只记录：

**值得最终报告引用的正式实验。**

例如：

- IB-M-1000正式Transformer；
- multi-step正式实验；
- BC正式Simulator；
- CEM-MPC正式Simulator；
- 最终对比。

普通：

- CUDA smoke；
- epoch=2；
- shape测试；
- debug run；

不登记。

---

# 48. 强制交接整理

只有：

1. 用户明确说准备切换Agent；
2. 当前模型额度即将耗尽；
3. 用户明确要求“整理交接状态”；

才集中更新：

- STATE.md；
- TODO.md；
- HANDOFF.md；
- DECISIONS.md（如需要）；
- EXPERIMENTS.md（如已有正式实验）。

然后确保Git存在稳定checkpoint。

除此之外：

**不要为了低概率未来交接频繁中断正常开发。**

---

# 49. Token策略

不要：

- 重复解释基础ML；
- 每个小函数汇报；
- 在聊天里贴完整源码；
- 重复阅读整篇论文；
- 重复调查已经写入Source/Data Audit的事实；
- 做无关重构；
- 大量微测试；
- 频繁维护项目文档；
- 反复请求已经明确的确认。

但是：

新的关键歧义必须回到题目、论文和官方代码确认。

> **节省Token不能高于正确性。**

---

# 50. 模型使用策略

当前：

**Sol High**

优先用于：

- Source Audit；
- environment compatibility；
- data semantics；
- Transformer；
- multi-step rollout；
- official BC compatibility；
- Behavior Policy definition；
- MPC/CEM；
- cross-module integration；
- difficult debugging。

Agent不能假装自行切换当前模型。

模型切换由用户控制。

---

# 51. 什么时候建议换较轻模型

只有同时满足：

1. 当前已经形成稳定checkpoint；
2. 接下来是一整个明确、低风险、机械工作块；
3. 工作仍预计消耗较多推理额度；

才输出：

```text
MODEL_SWITCH_RECOMMENDED

Current checkpoint:
...

Recommended model:
...

Recommended reasoning:
...

Reason:
...

Exact continuation task:
...
```

例如：

Terra Medium/High

可用于：

- 已定义接口的大量常规代码；
- config；
- CLI；
- README；
- batch scripts；
- mechanical refactor；
- 普通工程bug。

涉及：

- 论文/源码冲突；
- 环境兼容；
- Transformer；
- multi-step；
- BC适配；
- Behavior Policy评价；
- MPC/CEM；
- simulator异常；
- integration bug；

继续使用Sol High。

真正疑难才建议Sol XHigh。

---

# 52. 不要为了下载切模型

如果Sol High已经启动：

- Git clone；
- pip/conda install；
- dataset download；
- decompression；
- training process；
- 已有测试；

不要因为操作简单就暂停要求换模型。

这些主要消耗机器时间和I/O，而不是推理额度。

---

# 53. 无人值守规则

用户可能暂时不回复。

只要下一步：

- 已由本Prompt明确；
- 不改变核心研究方向；
- 可以验证；
- 可以Git回退；

继续推进。

只有以下情况停止：

- 删除大量用户文件；
- 覆盖未知重要文件；
- force push；
- rewrite Git history；
- credential问题；
- 需要改变核心研究路线；
- 题目/论文/源码出现关键冲突且无法判断；
- 磁盘不足；
- 下载规模明显超预期；
- 环境修改可能破坏用户重要已有Conda环境。

---

# 54. 阶段性回复

只有完成较大的工作块后简洁汇报：

1. 完成什么；
2. 哪些命令真实运行成功；
3. 关键结果；
4. 当前环境；
5. Git checkpoint；
6. blocker；
7. 下一项P0；
8. 是否建议换模型。

不要写长篇教程。

---

# 55. 执行顺序

严格优先：

## Phase 0
Repository检查

## Phase 1
读取本地题目 + NeoRL论文 + 官方GitHub  
完成Source Audit

## Phase 2
Existing Conda / CUDA / PyTorch Audit

## Phase 3
决定：
Reuse / Clone / New Environment

## Phase 4
NeoRL Environment Smoke

## Phase 5
IB-M-100 + Data Audit

## Phase 6
缓存 IB-M-1000

## Phase 7
空间合理则缓存 IB-M-10000

## Phase 8
Temporal Transformer World Model

## Phase 9
One-Step + Multi-Step Evaluation

## Phase 10
NeoRL Official BC

## Phase 11
World Model Strategy

## Phase 12
Three-Policy Simulator Evaluation

## Phase 13
IB-M-1000 Formal Experiment

## Phase 14
Report

之后才开始：

模型对比和创新。

---

# 56. 首次执行任务

现在不要只输出计划，直接执行。

1. 确认当前目录为 `G:\PAI`；
2. 检查Git和remote；
3. 阅读  
   `G:\PAI\PAI世界模型测试题目.docx`；
4. 阅读  
   `G:\PAI\2102.00714v2.pdf`；
5. 访问 NeoRL官方GitHub：  
   `https://github.com/Polixir/NeoRL`；
6. 阅读官方README、benchmark、IB环境和相关源码；
7. 如需要本地源码，clone官方仓库到  
   `G:\PAI\external\NeoRL`；
8. 记录NeoRL exact commit；
9. 找到并阅读官方BC implementation和config；
10. 完成一次集中 `SOURCE_AUDIT.md`；
11. 检查全部Conda environments；
12. 找到已有成功GPU/PyTorch环境；
13. 验证  
    `torch.cuda.is_available()`、  
    `torch.version.cuda`、  
    GPU name；
14. 判断Reuse / Clone / New Environment；
15. 禁止无理由重新下载完整CUDA Toolkit；
16. 实际运行NeoRL IB：
    reset / step / get_dataset；
17. 下载IB-M-100；
18. 完成一次DATA_AUDIT；
19. 双重确认30×6和sliding-window；
20. 检查G盘空间；
21. 缓存IB-M-1000；
22. 空间合理则缓存IB-M-10000；
23. 建立第一个稳定Git checkpoint；
24. 没有阻塞则继续实现Temporal Transformer World Model。

---

# 57. 首轮目标

第一次 Sol High 至少推进到：

- 题目与论文已真实阅读；
- 官方NeoRL源码已审计；
- Source Audit完成；
- Existing environment完成审计；
- NeoRL IB能真实运行；
- IB-M-100读取成功；
- Data Audit完成；
- 形成稳定Git checkpoint。

最好继续做到：

**Temporal Transformer在IB-M-100上完成短训练，并成功save/load checkpoint。**

如果仍有充足上下文和额度：

继续完成：

- one-step evaluation；
- multi-step evaluation。

不要等待用户再次批准。

---

# 58. 项目最高原则

始终遵守：

> **Task & Paper First**
>
> **Official GitHub Code First**
>
> **Existing Working GPU Environment First**
>
> **Three Modules Clearly Separated**
>
> **Transformer First, Innovation Later**
>
> **IB-M-100 Pipeline First, IB-M-1000 Formal Later**
>
> **Correctness Before Token Saving**
>
> **Git Checkpoints Over Frequent Documentation**
>
> **Repository State Is the Daily Source of Truth**