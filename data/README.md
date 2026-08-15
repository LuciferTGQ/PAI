# NeoRL Industrial Benchmark 数据

本项目使用 NeoRL Industrial Benchmark（IB）Medium 数据。原始数据不提交到 GitHub，请从 NeoRL 官方数据源取得后放入本目录。

## 数据规模

| 名称 | 轨迹数 | 状态转移数 | 每条轨迹长度 |
|---|---:|---:|---:|
| M100 | 100 | 100,000 | 1000 steps |
| M1000 | 1,000 | 1,000,000 | 1000 steps |
| M10000 | 10,000 | 10,000,000 | 1000 steps |

每个 transition 至少包含 `obs`、`next_obs`、`action`、`reward`、`done` 和轨迹边界 `index`。IB observation 为 180 维，即 30 帧历史 × 每帧 6 个变量；action 为 3 维连续控制。

## 目录与文件名

将下载后的 NPZ 文件放在 `data/raw/`。正式配置使用以下文件名：

```text
data/raw/ib-medium-100-train.npz
data/raw/ib-medium-10-val.npz
data/raw/ib-medium-1000-train.npz
data/raw/ib-medium-100-val.npz
data/raw/ib-medium-10000-train.npz
data/raw/ib-medium-1000-val.npz
```

不同训练规模使用相互对应的 validation set；所有正式架构比较在同一数据规模内共享相同数据划分和 normalization。

## 下载与导出

先将固定版本的 NeoRL 源码放到 `external/NeoRL`，再分别执行：

```powershell
python scripts/download_neorl_data.py --scale 100
python scripts/download_neorl_data.py --scale 1000
python scripts/download_neorl_data.py --scale 10000
```

脚本调用 NeoRL `env.get_dataset(data_type="medium", train_num=..., val_ratio=0.1)`，并将返回的字典导出为上面的统一 NPZ 文件名。默认复用已经存在的导出文件；只有明确需要覆盖时才添加 `--overwrite`。M10000 会在内存中构造千万级 transition，运行前应确认机器内存与磁盘空间充足。

## M10000 memory-map 准备

M10000 文件较大，正式配置从 `data/cache/ib-medium-10000-train/` 读取逐数组 `.npy` 文件，以避免每次把完整压缩 NPZ 加载到内存。准备命令：

```powershell
python scripts/prepare_npz_memmap.py data/raw/ib-medium-10000-train.npz data/cache/ib-medium-10000-train
```

脚本会检查 NPZ 中的必需数组，并可复用大小一致的已提取文件。`data/raw/` 和 `data/cache/` 均已加入 `.gitignore`。

## 数据语义检查

数据加载入口为 `src/data/ib_dataset.py`。训练时会检查 observation/action 维度、轨迹边界和转移数量，并在配置指定的 `data/cache/*-audit.json` 与 normalization cache 中保存本地检查结果。这些缓存不上传。
