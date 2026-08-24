# EEG brain-network provenance

## 结论

论文 Figure 10 的 5 条显著连接可以从历史 max‑T CSV 精确复现，但论文正文把这些 p 值写成了 Holm 校正。修正后的主管线同时保留“文字声明口径”和“历史实际口径”，且明确标注两者。

## 已核实的旧代码问题

- 主 Notebook `data_process_CSD_brainNetwork.ipynb` 和 graph-theory Notebook 中存在多版同名函数、ROI 字典和运行参数，结果依赖 cell 执行顺序。
- 分段有 5 s 与 10 s 两版；分析窗口也出现过 0–5、5–10、10–60 等探索设置。
- 部分 subject cell 先生成 CSD 数据，后续又用未 CSD 副本覆盖同一变量。
- delta 下界出现 1 Hz 与 2 Hz 两版；正式 connectivity 计算下界为 2 Hz，因此主管线冻结 2–4 Hz。
- 最终 ROI 版将 CP 通道放在 central；另一探索版将其放在 parietal 并增加 midline。论文 Figure 10 的连接命名对应前者。
- 历史 99 MB channel CSV 含正反两个方向；取绝对值后两行数值相同。新管线只保留唯一无向电极对。
- `save_roi_fc_means` 的函数名和注释声称取 mean，但最终执行行实际调用 `groupby(...).median()`；因此归档 `__ROI.csv` 是通道边中位数，而论文文字和 max‑T 代码使用通道边均值。
- 论文写 64 通道，真实 `.set` 头信息为 61 通道；10 个 ROI 实际覆盖 52 个非中线通道。
- source-space connectivity 与 graph-theory degree/efficiency/clustering/small-worldness 均未进入论文主要方法和结果，本次不纳入稳定主管线。

## Holm 与 max‑T 的证据

历史文件 `coherence_maxT_all_[10,60]_imcoh.csv` 中论文 5 个 p 值为 `2^15` 分母的精确符号翻转结果：

| Band | ROI pair | visual − tactile | max‑T p |
|---|---|---:|---:|
| alpha | Left_Frontal–Right_Parietal | -0.035143 | 0.017944 |
| alpha | Right_Frontal–Right_Parietal | -0.051572 | 0.006104 |
| alpha | Right_Temporal–Right_Parietal | -0.058882 | 0.029297 |
| theta | Left_Central–Left_Temporal | -0.023809 | 0.041443 |
| theta | Left_Frontal–Left_Central | -0.021708 | 0.030518 |

历史适配器会折叠方向重复，再以 `1e-12` 容差核对这 5 条。归档来源 SHA-256：

- ROI table: `691b2c86b85800e9d295861e759499a23c78abf3b630b5cb1d7019731fa1a1ae`
- max‑T table: `6013b7df56c3a12f2a33e950ec13ba352df61adc29b2cd3f8da1af72c4f12545`

对同一历史 ROI table 严格执行论文文字声明的 `45 × 5 = 225` 项全局 Holm 后，显著连接为 0 条。因此 5 个历史 p 值不能再称作 Holm-adjusted p。

## 修正后可重算链

主管线从 30 套 `.set/.fdt` 重新应用 CSD、分段和 multitaper `|ImCoh|`。QC 为 30/30 通过；`zys/st_tf2` 的 `.set` 内部 FDT 名少一个下划线，但现有 sibling 文件可正常读取，记录为 warning。

修正后结果与历史均值非常接近，但显著集合并不完全相同：

- global Holm：2 条；
- exact max‑T：3 条；
- historical published max‑T：5 条。

这符合旧 Notebook 存在处理版本混用的证据。修正结果用于未来重算；历史结果用于解释和展示论文当时实际产生的数字，两者互不冒充。
