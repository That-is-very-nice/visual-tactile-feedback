# Visual–tactile feedback research analysis

这是用于整理“视觉与振动触觉反馈下等长力控制”研究分析流程的私人仓库。目标是让论文参数、代码逻辑和结果来源长期可追溯，便于后续写作、修改和展示。

当前稳定版本包含行为学与 CMC。PDC 和 EEG 脑网络将在后续版本中逐步加入。原始人体数据、个人路径、历史 Notebook 和生成结果不进入 Git。

## 已稳定的分析

### 行为学

- trial 级平均力、标准差和变异系数；
- subject × condition 汇总；
- visual 与 tactile 条件的配对 Wilcoxon 统计；
- 论文行为学结果回归检查和 Figure 3。

真实 repaired force 数据已复现论文统计：平均力 `W=39, p=0.25238037109375`；force CV `W=0, p=0.00006103515625`。

### CMC

- 30 套 EEG/EMG/annotation 输入注册与质量检查；
- trial 2–5 的 10–60 s 稳态区间，切分为 20 个十秒段；
- CSD 后 EEG 与未整流 EMG 的 multitaper magnitude-squared coherence；
- 左侧感觉运动 ROI 的单一 CMC 指标；
- 三频带配对 Wilcoxon、Holm 校正、回归检查和修正后 Figure 5。

CMC 有两条明确分开的证据链：默认管线是参数显式的单次重算；论文 Fig. 5/Table 1 只能通过历史 CSV 中两批冲突输出的显式平均复现。详细说明见 [CMC provenance](docs/cmc_provenance.md)。

## 仓库结构

```text
configs/
  behavior.example.toml
  cmc.example.toml
  paper_behavior_expected.json
  cmc_corrected_expected.json
  cmc_published_expected.json
  cmc_published_legacy_expected.json
docs/
  behavior_method.md
  cmc_method.md
  cmc_provenance.md
  data_layout.md
  cmc_data_layout.md
src/visual_tactile_force/
  behavior*.py
  cmc*.py
  neuro_registry.py
  statistics.py
tests/
```

## 安装

Python 3.10 或更高版本：

```bash
python -m pip install -e .
python -m pip install -e ".[cmc]"
```

第一行安装行为学依赖；第二行增加 CMC 所需的 MNE 和 mne-connectivity。

## 运行行为学

```bash
cp configs/behavior.example.toml configs/behavior.local.toml
vtf-behavior run \
  --config configs/behavior.local.toml \
  --output-dir results/runs/paper_behavior
```

## 运行 CMC

```bash
cp configs/cmc.example.toml configs/cmc.local.toml
vtf-cmc qc \
  --config configs/cmc.local.toml \
  --output-dir results/runs/cmc_qc

vtf-cmc run \
  --config configs/cmc.local.toml \
  --output-dir results/runs/cmc_corrected \
  --resume
```

CMC 输出包括输入 QC、subject 汇总、统计、数值回归、修正后 Figure 5、checkpoint 和运行 manifest。真实数据布局见 [CMC data layout](docs/cmc_data_layout.md)。

## 测试

```bash
python -m unittest discover -s tests -v
```

GitHub Actions 会分别运行基础测试和包含 MNE 的 CMC 测试；测试不使用或上传真实受试数据。
