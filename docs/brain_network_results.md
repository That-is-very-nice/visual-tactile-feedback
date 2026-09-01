# EEG brain-network results

## 结果定义

视觉与触觉条件差值为 `st_no − st_tf2`。每个频带的 100 个有序 ROI 组合分别进行双侧配对 Wilcoxon 符号秩检验，并在该频带内进行 Holm 校正。校正后合并正反方向记录，仅报告脑区间连接。

## 显著脑区间连接

| Band | ROI pair | visual − tactile | Holm-adjusted p |
|---|---|---:|---:|
| alpha | Left_Frontal–Right_Parietal | -0.035143 | 0.017944 |
| alpha | Right_Frontal–Right_Parietal | -0.051572 | 0.006104 |
| alpha | Right_Temporal–Right_Parietal | -0.058882 | 0.029297 |
| theta | Left_Central–Left_Temporal | -0.023809 | 0.041443 |
| theta | Left_Frontal–Left_Central | -0.021708 | 0.030518 |

5 条连接的均值差均为负，表示 tactile 条件下的绝对虚部相干高于 visual 条件。

## 完整性检查

- 完整统计表：500 行，即 5 个频带 × 每频带 100 个有序 ROI 组合。
- Holm 校正后显著：11 条有向记录，其中 10 条对应 5 对脑区间连接，1 条为 theta 频带的 Left_Central ROI 内部连接。
- 合并正反方向并排除 ROI 内部连接后：5 条论文报告连接。
- 完整统计表 SHA-256：`6013b7df56c3a12f2a33e950ec13ba352df61adc29b2cd3f8da1af72c4f12545`。
- 显著结果表 SHA-256：`06bf5e51c3db3beabceb1011ac15b76e5fe7e3e6631b6cbe80e04541aede0030`。

机器可读的结果基线保存在 `configs/brain_network_expected.json`。
