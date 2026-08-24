# Local force-data layout

真实受试数据保存在 Git 仓库之外。本地路径通过 `configs/behavior.local.toml` 的 `paths.force_data_dir` 指定。

## 默认文件布局

```text
<force_data_dir>/
  qh_st_no.csv
  qh_st_tf2.csv
  wxl_st_no.csv
  wxl_st_tf2.csv
  ...
```

默认文件名模板是 `{subject}_{condition}.csv`。如果本地数据使用其他布局，修改配置中的 `file_template`，而不要在 Python 源码中写绝对路径。

## CSV 列

CSV 无 header，前四列依次为：

1. `event`：`start`、`end` 或普通样本标记；
2. `time_ms`：毫秒时间戳；
3. `target_force`：目标力；
4. `measured_force`：实测力。

正式运行前，质量检查会验证文件完整性、start/end 数量、trial 数量、时间单调性、分析窗和采样率。
