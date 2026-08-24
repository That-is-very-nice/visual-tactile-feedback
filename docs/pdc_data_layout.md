# PDC local data layout

PDC 与 CMC 共用 EEG/EMG 输入注册表。Git 仓库只保存配置模板、聚合回归值和代码，不保存人体原始数据、受试级 PDC 输出或本机绝对路径。

## 需要的本地输入

```text
eeglab_set_dir/
  qh1_[0 60]_EEG.set
  qh1_[0 60]_EEG.fdt
  qh1_[0 60]_EMG.set
  qh1_[0 60]_EMG.fdt
  qh8_[0 60]_EEG.set
  ...
annotation_dir/
  eeg_qh1_61_5_annotations.txt
  eeg_qh8_61_5_annotations.txt
  ...
behavior run directory/               # 仅 Figure 8 需要
  behavior_subject_summary.csv
```

实际文件名由 `configs/pdc.local.toml` 的 subject/event code 映射解析。每个数据集必须同时存在 EEG `.set/.fdt`、EMG `.set/.fdt` 和 annotation，共 15 subjects × 2 conditions = 30 套。

## 配置

```bash
cp configs/pdc.example.toml configs/pdc.local.toml
```

只修改 `[paths]` 中的本地路径。`configs/pdc.local.toml` 已被 `.gitignore` 排除。

## 输出

```text
results/runs/pdc_corrected/
  pdc_input_quality.json
  pdc_header_quality.json
  pdc_input_registry.csv
  pdc_null_threshold.npz
  pdc_null_threshold.json
  pdc_checkpoint.json
  pdc_subject_summary.csv
  pdc_statistics.csv
  pdc_statistics.json
  pdc_method_regression.json
  pdc_behavior_correlations.csv
  pdc_behavior_correlations.json
  figure_7_pdc_corrected.{png,pdf}
  figure_8_pdc_behavior_corrected.{png,pdf}
  run_manifest.json
```

`results/runs/**` 已被 Git 排除。`--resume` 要求 checkpoint 中的计算配置 hash 与当前运行一致；Monte Carlo 阈值也会按其独立设置 hash 缓存。
