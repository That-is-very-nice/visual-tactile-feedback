# Brain-network data layout

脑网络计算只读取预处理 EEG，不读取 force 或 EMG。包含本机路径的 `configs/brain_network.local.toml` 不进入 Git。

## 输入

```text
eeglab_set_dir/
  qh1_[0 60]_EEG.set
  qh1_[0 60]_EEG.fdt
  qh8_[0 60]_EEG.set
  qh8_[0 60]_EEG.fdt
  ...
```

文件名由 `subject + event_code + "_[0 60]_EEG"` 构成。QC 检查 `.set/.fdt` 是否成对，并核对 5 个 epochs、60,000 个采样点、1000 Hz、61 个通道以及 52 个 ROI 通道。

结果验证使用完整统计表 `coherence_maxT_all_[10,60]_imcoh.csv`。该表对每个频带保存 100 个有序 ROI 组合；其中 `p_maxT` 列由 Wilcoxon–Holm 分析函数写入，内容是 Holm 调整后的 p 值。其路径配置在 `paths.brain_network_statistics_csv`。

## 完整重算输出

```text
results/runs/brain_network/
  brain_network_input_quality.json
  brain_network_input_registry.csv
  brain_network_checkpoint.json
  brain_network_subject_summary.csv
  brain_network_statistics_all.csv
  brain_network_statistics_canonical.csv
  brain_network_significant_interregional.csv
  brain_network_statistics.json
  brain_network_validation.json
  brain_network_manifest.json
  figure_9_brain_network.{png,pdf}
  figure_10_brain_network.{png,pdf}
```

`brain_network_subject_summary.csv` 为 8,250 行：`15 subjects × 2 conditions × 55 ROI pairs × 5 bands`。统计文件分别保存 500 条有向记录、275 条合并方向后的记录，以及最终显著脑区间连接。

## 固定结果验证输出

```text
results/runs/brain_network_verification/
  brain_network_statistics_all.csv
  brain_network_statistics_canonical.csv
  brain_network_significant_interregional.csv
  brain_network_regression.json
  figure_10_brain_network.png
```

`results/runs/**` 已被 `.gitignore` 排除；Git 只保存代码、方法文档和不含受试者级数据的聚合结果基线。

## 运行命令

```bash
vtf-network qc --config configs/brain_network.local.toml \
  --output-dir results/runs/brain_network_qc

vtf-network run --config configs/brain_network.local.toml \
  --output-dir results/runs/brain_network --resume

vtf-network verify --config configs/brain_network.local.toml \
  --output-dir results/runs/brain_network_verification
```

完整重算每完成一个 subject-condition 就原子写入 checkpoint；只有配置 hash 相同的运行可以续跑。`verify` 读取完整统计表，合并对称方向，并在 `1e-12` 容差内核对最终 5 条脑区间连接。
