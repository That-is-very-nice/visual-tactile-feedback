# Local CMC data layout

真实 EEG、EMG 和 annotation 保存在 Git 仓库之外。本地路径通过 `configs/cmc.local.toml` 指定。

## EEGLAB 文件

`paths.eeglab_set_dir` 中每个 subject/event 包含配对 EEG 和 EMG：

```text
<set_dir>/
  qh1_[0 60]_EEG.set
  qh1_[0 60]_EEG.fdt
  qh1_[0 60]_EMG.set
  qh1_[0 60]_EMG.fdt
  qh8_[0 60]_EEG.set
  ...
```

文件 stem 为 `{subject}{event_code}_[0 60]`。正式配置中的 subject-specific event map 是唯一映射来源，不要在代码中拼接固定 tactile event。

EEG `.set` 是已经完成滤波、平均参考、坏道插值、ICA 和 artifact rejection 的分析输入；本仓库的 CMC 命令负责后续 CSD、分段、coherence、统计和回归，不重新执行上游人工 ICA 决策。

## Annotation 文件

`paths.annotation_dir` 中使用：

```text
<annotation_dir>/
  eeg_qh1_61_5_annotations.txt
  eeg_qh8_61_5_annotations.txt
  ...
```

CSV 必须包含 `Onset` 和 `Annotation` 列。质量检查将 `DC trigger 13` 与其后的 `DC trigger 14` 配对，并要求每套数据得到五个约 60 s trial。

## 不进入 Git 的内容

- `.set`、`.fdt` 和 annotation 原始文件；
- subject-level CMC 输出；
- `results/runs/` 下的 checkpoint、图和 manifest；
- 含个人绝对路径的 `configs/cmc.local.toml`。
