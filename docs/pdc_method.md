# PDC analysis method contract

本文档冻结修正后 PDC 管线的计算定义。它是可重算、可回归的主管线；论文已发表数值由独立的历史适配器处理。

## 分析单位与分段

- 15 名受试，visual = `st_no`，tactile = `st_tf2`。
- `qh` 的 tactile event code 为 8，其余为 9；visual 均为 1。
- 排除 trial 1，trial 2–5 取 10–60 s，切为 20 个不重叠的 10 s segment。
- EEG 与 EMG 必须在 segment 数、采样率、样本数和分段 provenance 上完全一致。
- 输入为 1000 Hz；分段后使用 polyphase anti-aliasing 重采样至 160 Hz，每段 1600 点。

## 信号与 ROI

- `.set/.fdt` 输入应已完成 1–58 Hz、平均参考、坏道插值、ICA 和 ±100 µV rejection；读取后应用 CSD。
- 对齐后 EMG 不再滤波，不整流。
- ROI: `CP5, CP3, CP1, CPz, C5, C3, C1, Cz, FC5, FC3, FC1, FCz`。
- 每个 ROI EEG 通道与肱二头肌 EMG 单独构建一个双变量模型。

## VAR 与 PDC

- 每个 epoch/channel 沿时间独立 z-standardize。
- 默认使用显式固定 25 阶 OLS-VAR，不从 Notebook 执行状态继承阶数。
- 通道顺序为 `[EEG, EMG]`；descending = EEG→EMG = PDC `[target=1, source=0]`；ascending = EMG→EEG = `[target=0, source=1]`。
- PDC 按 Baccalá & Sameshima 定义从频域 VAR 系数直接计算。内部实现已与本机 SCoT 0.3.dev0 在随机输入上逐点对比，两个方向最大绝对差均为 0。
- 不依赖 PyPI SCoT 0.2.1，因其使用了现代 SciPy 已删除的 API。

## 频率、零假设与指标

- 801 个频点严格对应 0–80 Hz，步长 0.1 Hz；正式汇总使用 2–58 Hz。
- 用与实际分析相同的 `20 epochs × 2 channels × 1600 samples`、25 阶 VAR 拟合 1000 组独立高斯噪声。
- 固定 `RandomState(0)`，每个方向、每个频点取第 95 百分位；阈值是频率分辨的数组，不是一个标量。
- 稳定指标为 `max_normalized_suprathreshold_excess`：对每通道计算 `max(PDC - threshold, 0)` 的带内积分，除以频带宽度，再取 ROI 最大值。
- alpha 8–13 Hz，beta 13–30 Hz，gamma 30–58 Hz，边界与历史代码一样为 inclusive。

## 统计与相关

- 差值定义为 `st_no - st_tf2`。
- 每个方向内分别对三频带做双侧配对 Wilcoxon，并在该方向内做 Holm 校正。
- 效应量 `r_z = z / sqrt(n_pairs)`，差值为 0 的 pair 不进入效应量分母。
- 修正后 Figure 8 使用本仓库行为学输出的 `force_cv`，在每个条件内计算 descending gamma PDC 的 Spearman 相关。

## 真实数据回归

`configs/pdc_corrected_expected.json` 冻结 2026-08-24 的 30 套真实输入聚合结果，不含受试级数据。正式运行必须在 `1e-12` 容差内通过六个 direction × band 的全字段回归。
