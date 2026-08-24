# Visual–tactile feedback research analysis

这是一个用于整理“视觉与振动触觉反馈下等长力控制”研究分析流程的私人仓库。目标是让分析逻辑、论文参数和结果来源长期可追溯，便于后续写作、修改和展示。

第一个稳定版本只包含行为学分析。CMC、PDC 和 EEG 脑网络将在后续版本中逐步加入。原始人体数据、个人路径、历史 Notebook 和生成结果不进入 Git。

## 当前稳定范围

行为学流程从修复后的 force-sensor CSV 生成：

- trial-level 平均力、标准差和变异系数（CV）；
- subject × condition 汇总表；
- visual 与 tactile 条件的配对 Wilcoxon 统计；
- 与论文行为学结果的数值回归检查；
- Figure 3 的 PNG/PDF 图。

已使用 15 名正式受试的 repaired force 数据从原始 CSV 重算并验证：

- mean force: `W=39`, `p=0.25238037109375`, `r_z=-0.3079589415`；
- force CV: `W=0`, `p=0.00006103515625`, `r_z=-0.8798826901`。

## 仓库结构

```text
configs/
  behavior.example.toml          # 行为学论文参数与本地路径模板
  paper_behavior_expected.json   # 论文聚合结果回归基线
docs/
  behavior_method.md             # 行为学方法契约
  data_layout.md                 # 本地 CSV 布局和列定义
src/visual_tactile_force/
  behavior.py                    # 读取、分 trial、滤波和行为指标
  behavior_cli.py                # 行为学命令行流程
  quality.py                     # 数据质量检查
  statistics.py                  # 配对 Wilcoxon 和效应量
  regression.py                  # 论文数值回归
  figures.py                     # Figure 3
tests/                            # 行为学单元测试
```

## 运行

Python 3.10 或更高版本：

```bash
python -m pip install -e .
cp configs/behavior.example.toml configs/behavior.local.toml
```

修改 `configs/behavior.local.toml` 中的 `force_data_dir`，然后运行：

```bash
vtf-behavior run \
  --config configs/behavior.local.toml \
  --output-dir results/runs/paper_behavior
```

输出包括数据质量报告、trial 表、subject 汇总表、统计结果、论文回归报告、Figure 3 和运行 manifest。详细定义见 [行为学方法契约](docs/behavior_method.md)。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

GitHub Actions 会在每次 push 后运行同样的基础测试。测试不使用或上传真实受试数据。
