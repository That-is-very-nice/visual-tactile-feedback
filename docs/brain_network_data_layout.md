# Brain-network data layout

脑网络主管线只读取预处理 EEG，不读取 force 或 EMG 信号。私有本机配置不进入 Git。

## 输入

```text
eeglab_set_dir/
  qh1_[0 60]_EEG.set
  qh1_[0 60]_EEG.fdt
  qh8_[0 60]_EEG.set
  qh8_[0 60]_EEG.fdt
  ...
```

命名由 `subject + event_code + "_[0 60]_EEG"` 构成。QC 检查 `.set/.fdt` 是否成对、5 epochs、60,000 samples、1000 Hz、61 channels，以及 52 个 ROI 通道是否齐全。

历史发表链另外读取两份归档 CSV：

- `network_imcoh_fb_[10,60]__ROI.csv`：15 人 × 6 条件 × 5 频段 × 45 跨区连接；
- `coherence_maxT_all_[10,60]_imcoh.csv`：论文 Figure 10 使用的 max‑T 输出。

这两个路径只应放在 `configs/brain_network.local.toml`。

## 输出

```text
results/runs/brain_network_corrected/
  brain_network_input_quality.json
  brain_network_input_registry.csv
  brain_network_checkpoint.json
  brain_network_subject_summary.csv
  brain_network_statistics_declared_holm.csv
  brain_network_statistics_published_style_max_t.csv
  brain_network_statistics.json
  brain_network_method_regression.json
  brain_network_manifest.json
  figure_9_brain_network_corrected.{png,pdf}
  figure_10_brain_network_corrected.{png,pdf}
```

subject summary 为 8250 行：`15 subjects × 2 conditions × 55 ROI pairs × 5 bands`。`results/runs/**` 已被 `.gitignore` 排除；Git 只保存代码、无人体数据的聚合回归基线和方法文档。

## 推荐命令

```bash
vtf-network qc --config configs/brain_network.local.toml \
  --output-dir results/runs/brain_network_qc

vtf-network run --config configs/brain_network.local.toml \
  --output-dir results/runs/brain_network_corrected --resume
```

每完成一个 subject-condition 就原子写入 checkpoint；同一配置 hash 可安全续跑，配置变化时拒绝误用旧断点。
