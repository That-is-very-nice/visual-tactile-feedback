# CMC analysis method contract

本文档冻结 CMC 稳定管线的计算定义。修改以下任一参数时，必须重新运行真实数据回归并检查 provenance。

## 分析单位

- 正式分析包含 15 名受试。
- visual feedback = `st_no`；tactile feedback = `st_tf2`。
- `qh` 的 tactile event code 为 8；其余受试为 9。visual event code 均为 1。
- 每个条件包含 5 个 60 s trial；trial 1 为熟悉试次并排除。
- trial 2–5 只使用 10–60 s，切成五个不重叠的 10 s 段；每条件共 20 段。

## 信号与频谱

- EEG 与 EMG 的 segment 数、采样率和样本数必须一致。
- 输入 `.set` 必须已完成 1–58 Hz、平均参考、坏道插值、ICA 和 ±100 µV rejection；本管线在读取后应用 CSD。
- 对齐后的 EMG 不再滤波，也不整流。
- 使用 multitaper magnitude-squared coherence（`coh`）。
- 频谱范围为 2–58 Hz；multitaper bandwidth 为 2 Hz，adaptive weighting 开启。
- 置信阈值为 `1 - 0.05 ** (1 / (L - 1))`，`L=20`。

## ROI 与稳定指标

ROI 包含 `CP5, CP3, CP1, CPz, C5, C3, C1, Cz, FC5, FC3, FC1, FCz`。

稳定指标固定为 `max_mean_suprathreshold_excess`：先对每个 ROI 通道保留高于置信阈值的正 excess，再在频带内求正 excess 的均值，最后取 ROI 通道最大值。它对应历史 Notebook 的 `cmc5`。

频带使用历史代码的 inclusive mask：alpha 8–13 Hz、beta 13–30 Hz、gamma 30–58 Hz。

## 统计

- 差值定义为 `st_no - st_tf2`。
- 每频带使用双侧配对 Wilcoxon signed-rank test。
- 零差值不进入 `n_pairs` 和效应量分母；`subject_count` 仍记录完整 15 名受试。
- 效应量为 `r_z = z / sqrt(n_pairs)`，不使用 continuity correction。
- 三频带 p 值使用 Holm 校正。

## 回归与输出

`configs/cmc_corrected_expected.json` 保存 2026-08-24 由 30 套真实输入重算得到的聚合基线，不包含受试级数据。正式运行必须通过该回归检查。

当前 QC 可以验证文件、事件、维度、采样率、EEG/EMG 对齐和 ROI 通道，但不能从 `.set` header 自动证明上游滤波、坏道处理或 ICA 决策；这些步骤仍依赖预处理数据的人工 provenance。

默认输出的 `figure_5_cmc_corrected.*` 是修正后单次方法结果，不冒充论文原始 Fig. 5。论文图表的历史复现由独立 adapter 和独立 baseline 处理。
