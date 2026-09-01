# EEG brain-network method contract

本文档冻结论文 EEG 脑网络的计算和统计方法。

## 分析单位与分段

- 15 名受试者；visual 条件为 `st_no`，tactile 条件为 `st_tf2`。
- `qh` 的 tactile event code 为 8，其余受试者为 9；visual event code 均为 1。
- 输入 `.set/.fdt` 已完成 1–58 Hz 滤波、平均参考、坏道插值、ICA 和 ±100 µV 伪迹排除；读取后应用 CSD。
- 排除 trial 1。trial 2–5 取 10–60 s，每个 trial 切为 5 个不重叠的 10 s segment，共 20 个 segment。
- 输入数据为 61 个 EEG 通道、5 个 epochs、每个 epoch 60,000 个采样点，采样率为 1000 Hz。

## ROI

10 个 ROI 覆盖 52 个非中线通道：

| ROI | Channels |
|---|---|
| Left_Frontal | Fp1, AF3, AF7, F1, F3, F5, F7 |
| Right_Frontal | Fp2, AF4, AF8, F2, F4, F6, F8 |
| Left_Central | FC1, FC3, FC5, C1, C3, C5, CP1, CP3, CP5 |
| Right_Central | FC2, FC4, FC6, C2, C4, C6, CP2, CP4, CP6 |
| Left_Temporal | FT7, T7, TP7 |
| Right_Temporal | FT8, T8, TP8 |
| Left_Parietal | P1, P3, P5, P7 |
| Right_Parietal | P2, P4, P6, P8 |
| Left_Occipital | PO3, PO7, O1 |
| Right_Occipital | PO4, PO8, O2 |

`Fpz, AFz, Fz, FCz, Cz, CPz, Pz, POz, Oz` 不进入 ROI 分析。

## 连接估计与聚合

- 使用 `mne-connectivity` 的 multitaper `imcoh`，bandwidth 为 2 Hz，启用 adaptive weighting。
- 对每条电极连接先在频带内平均 imaginary coherence，再取绝对值，得到 band-averaged `|ImCoh|`。
- 频段为 delta 2–4 Hz、theta 4–8 Hz、alpha 8–13 Hz、beta 13–30 Hz 和 gamma 30–58 Hz；频带边界按闭区间处理。
- ROI 连接值为相应全部电极对 `|ImCoh|` 的算术平均。
- 每个 subject-condition-band 生成 55 个唯一 ROI 连接：45 个脑区间连接和 10 个脑区内部连接。

## 统计分析

视觉与触觉的差值定义为 `st_no − st_tf2`。

每个频带包含 10 × 10 = 100 个有序 ROI 组合，其中包括 90 个脑区间组合和 10 个脑区内部组合。对每个组合的 15 名受试者条件差值进行双侧配对 Wilcoxon 符号秩检验，参数固定为：

- `zero_method="pratt"`
- `correction=True`
- `alternative="two-sided"`
- `method="auto"`（旧版 SciPy 中对应 `mode="auto"`）

随后在每个频带的 100 个原始 p 值内执行 Holm step-down 校正，显著性阈值为 0.05。不同频带分别构成不同的多重比较家族。

Holm 校正完成后再合并数值相同的正反方向连接，得到每个频带 55 个唯一 ROI 连接。论文结果只报告 45 个脑区间连接，不报告 10 个 ROI 内部连接。

## 固定结果

最终方法得到 11 条有向显著记录：5 对脑区间连接各出现两个方向，另有 1 条 ROI 内部连接。合并方向并排除 ROI 内部连接后，论文报告 5 条连接。具体数值见 `docs/brain_network_results.md`，机器可读基线见 `configs/brain_network_expected.json`。
