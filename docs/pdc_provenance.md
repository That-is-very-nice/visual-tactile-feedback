# PDC published-output provenance

## 结论

论文 Table 2 和 Figure 8 可以从归档 CSV 精确复现，但其实际计算链与方法文字不完全一致，也不能从旧 Notebook 恢复成唯一的可重算参数集。

## 已证实的历史处理

- 论文方法文字对应旧代码 `pdc_down4/pdc_up4`：ROI 最大的归一化阈上面积。
- Table 2 与 Figure 8 实际选择 `pdc_down5/pdc_up5`：ROI 最大的平均阈上 excess。
- 旧 Monte Carlo helper 虽先产生频率分辨曲线，后续却对整条曲线求均值，最终用一个方向标量阈值。
- SCoT 频谱本应对应 0–Nyquist，旧 Notebook 却用 `linspace(2, 80, n_freq)` 赋予频率标签。
- PDC helper 默认只 crop 每个 10 s segment 的前 5 s；论文写的是与 CMC 相同分段，未说明这一再截断。

## 模型阶数无法完整恢复

旧 `order_dict` 保存了 14 个名字，其中包含非正式受试 `zza`，但缺少正式受试 `ice` 和 `pepper`。可识别的正式记录为：

| Subject | Order | Subject | Order |
|---|---:|---|---:|
| qh | 25 | wxl | 33 |
| zys | 24 | lzy | 30 |
| xl | 26 | zhb | 31 |
| phoom | 23 | prae | 29 |
| maple | 27 | regina | 28 |
| ljj | 29 | pathe | 25 |
| pun | 29 | | |

更关键的是，历史批处理 cell 只在进入多受试连续执行前赋值了一次 `order`，后续受试 cell 并未重新从字典取值。因此归档 CSV 可能使用了一个残留常量阶数，但 Notebook 执行状态不能证明该常量是多少。

旧阶数搜索实验可见 2–79 阶扫描，并将 VAR 参数 coherence 与 multitaper coherence 的均方差最小化；但代码实际使用未加权 MSE，论文写的是加权 MSE，且没有保存所有正式受试的选阶产物。

## 两条证据链

### 历史发表链

`legacy_pdc.reproduce_published_pdc_statistics` 只接受唯一的 `subject × condition × band` 键，显式选择 PDC5，并复现 Table 2。归档 CSV SHA-256 和所有数值冻结在 `pdc_published_legacy_expected.json`。

| Direction | Band | W | Holm p | r_z |
|---|---|---:|---:|---:|
| descending | alpha | 45 | 0.421204 | -0.219971 |
| descending | beta | 32 | 0.241089 | -0.410612 |
| descending | gamma | 17 | 0.037354 | -0.630583 |
| ascending | alpha | 44 | 0.916512 | 0.029074 |
| ascending | beta | 36 | 0.562866 | 0.351953 |
| ascending | gamma | 43 | 0.718262 | 0.249300 |

Figure 8 归档 merged-pair 表使用历史 `fluct_across_trials`，它与现在重建行为管线的 `force_cv` 接近但并非逐值相同。历史 Spearman 结果为 visual `ρ=-0.286, p=0.302`，tactile `ρ=-0.546, p=0.0351`。

### 修正后可重算链

主管线显式使用 25 阶、完整 10 s segment、正确 0–Nyquist 频率轴、频率分辨的 95% 阈值和 PDC4 语义指标。这是一条透明的修正方法，不声称能唯一重建原始执行状态。

| Direction | Band | W | Holm p | r_z |
|---|---|---:|---:|---:|
| descending | alpha | 51 | 0.638672 | -0.131982 |
| descending | beta | 30 | 0.283813 | -0.439941 |
| descending | gamma | 35 | 0.337646 | -0.366618 |
| ascending | alpha | 45 | 0.842407 | 0.219971 |
| ascending | beta | 40 | 0.830566 | 0.293294 |
| ascending | gamma | 54 | 0.842407 | 0.087988 |

对当前重建的 `force_cv`，修正后 descending gamma PDC 相关为 visual `ρ=-0.496, p=0.0598`，tactile `ρ=-0.661, p=0.00733`。这些数值应标注为 corrected analysis，不应替换或冒充论文原 Figure 8。
