# EEG brain-network method contract

本文档冻结传感器空间脑网络主管线。它用于可重算分析；论文原始数值由独立的历史适配器验证。

## 分析单位与分段

- 15 名受试，visual = `st_no`，tactile = `st_tf2`。
- `qh` 的 tactile event code 为 8，其余为 9；visual 均为 1。
- 输入 `.set/.fdt` 已完成 1–58 Hz、平均参考、坏道插值、ICA 和 ±100 µV rejection；读取后重新应用 CSD。
- 排除 trial 1；trial 2–5 取 10–60 s，每个 trial 切为 5 个不重叠的 10 s segment，共 20 段。
- 实际输入为 61 个 EEG 通道、5 epochs、60,000 samples/epoch、1000 Hz。论文正文写“64 channels”，与归档数据不一致。

## ROI

稳定 ROI 映射覆盖 52 个非中线通道：

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

`Fpz, AFz, Fz, FCz, Cz, CPz, Pz, POz, Oz` 不进入 10 个 ROI。旧 Notebook 曾存在把 CP 通道归入 parietal 的另一版映射；稳定管线禁止运行时静默替换映射。

## 连接估计与聚合

- 使用 `mne-connectivity` multitaper `imcoh`，bandwidth = 2 Hz，adaptive weighting 开启。
- 每个频带先平均 signed ImCoh，再取绝对值，得到 band-averaged `|ImCoh|`，消除有向符号解释；这是历史 channel CSV 和 max‑T 链实际使用的顺序。
- 频段为 delta 2–4、theta 4–8、alpha 8–13、beta 13–30、gamma 30–58 Hz；共享边界按历史代码的 inclusive 规则进入相邻两带。
- 只计算唯一无向电极对。52 个通道共有 1326 对，其中 1188 对跨 ROI、138 对位于 ROI 内。
- 每个 ROI pair 的值为其全部电极对 `|ImCoh|` 的等权平均。正式脑区间表每个 subject-condition 为 `45 pairs × 5 bands`；区内 10 对只为历史 max‑T correction family 保留。

## 两套统计口径

差值均定义为 `st_no - st_tf2`。

1. `declared_wilcoxon_holm_global_225`：对 45 个跨区连接 × 5 个频段分别做双侧配对 Wilcoxon，再对全部 225 项做一次 Holm step-down。这对应论文方法文字。
2. `published_style_exact_max_t_per_band_55`：每个频段用 45 个跨区 + 10 个区内连接构成 family，对 15 人执行全部 `2^15 = 32768` 种同步符号翻转，使用 studentized mean difference 的双侧 max‑T。这对应历史实际统计代码的结构。

效应方向始终由 visual − tactile 的均值差解释。两套结果必须分文件保存，不允许把 max‑T p 值标成 Holm p 值。

## 修正后真实数据回归

2026-08-24 的 30 套真实输入结果冻结在 `configs/brain_network_corrected_expected.json`，不包含受试级数据。

- declared global Holm：2 条显著连接，均在 alpha（Left_Frontal–Right_Parietal；Right_Frontal–Right_Parietal）。
- published-style exact max‑T：3 条显著连接（alpha 的 Right_Frontal–Right_Parietal、Right_Temporal–Right_Parietal；theta 的 Left_Central–Left_Temporal）。

正式运行必须在 `1e-12` 容差内复现完整显著边集合及其均值差和校正 p 值。
