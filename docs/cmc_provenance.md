# CMC published-output provenance

## 结论

论文 Fig. 5 和 Table 1 的数值可以复现，但它们不是一套可唯一识别的 CMC 处理运行的直接输出。

历史文件 `all_subjects_cmc_zcmc_coh_fb_CSD_[10,60].csv` 对每个正式 `subject × condition × band` 键保存了两行。配对行数值互相冲突，说明这是两次不同参数运行追加到同一 CSV 的结果。旧统计 helper 在 pivot 前执行 `mean()`，因此无提示地平均了两批结果。

对两批结果显式平均并选择历史 `cmc5` 后，可精确复现论文：

| Band | Visual mean | Tactile mean | W | Holm p | r_z |
|---|---:|---:|---:|---:|---:|
| alpha | 0.0140263166 | 0.0142383548 | 54 | 1.0 | -0.0879882690 |
| beta | 0.0288263068 | 0.0336838926 | 45 | 1.0 | -0.2199706725 |
| gamma | 0.0273562583 | 0.0298845890 | 52 | 1.0 | -0.1173176920 |

## 仓库采用的发布政策

默认 CMC 结果采用参数显式的 10 秒分段单次重算，不再平均冲突运行。2026-08-24 对 15 名正式受试重算得到：

| Band | Visual mean | Tactile mean | W | Holm p | r_z |
|---|---:|---:|---:|---:|---:|
| alpha | 0.0043348572 | 0.0016347558 | 11 | 1.0 | 0.1916629695 |
| beta | 0.0153437291 | 0.0225426793 | 20 | 0.8349609375 | -0.3484991691 |
| gamma | 0.0167479442 | 0.0179050570 | 43 | 1.0 | -0.0484569837 |

两条证据链的结论均为三频带条件差异不显著，但数值和来源不可互换。

## 代码护栏

- 稳定批处理在 pivot 和统计前拒绝重复分析键。
- `legacy_cmc.reproduce_published_cmc_aggregate` 是唯一允许平均两批冲突行的函数；它要求每键恰好两行，并验证每组数值确实冲突。
- `cmc_corrected_expected.json` 冻结修正后方法；`cmc_published_expected.json` 只记录论文报告；`cmc_published_legacy_expected.json` 冻结历史 CSV 的精确复现和 SHA-256。
- `tools/reproduce_published_cmc.py` 会同时验证源文件 SHA-256、重复结构和全部冻结数值。

此设计保留论文可追溯性，同时避免把已知 provenance 缺陷继续当作默认科学计算。
