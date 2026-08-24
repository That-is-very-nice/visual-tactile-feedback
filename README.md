# Visual–tactile feedback research analysis

这是用于整理“视觉与振动触觉反馈下等长力控制”研究分析流程的私人仓库。目标是让论文参数、代码逻辑和结果来源长期可追溯，便于后续写作、修改和展示。

当前稳定分析包含行为学、CMC、PDC 与 EEG 脑网络。原始人体数据、个人路径、历史 Notebook 和生成结果不进入 Git。

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

### PDC

- 与 CMC 共用的 30 套输入 QC、CSD 和 20 个十秒段；
- 160 Hz 降采样、显式 25 阶双变量 OLS-VAR 和 EEG↔EMG PDC；
- 1000 次固定种子、频率分辨的 Monte Carlo 阈值；
- 双方向三频带统计、行为相关、回归检查、Figure 7 和 Figure 8。

PDC 同样分为历史发表链和修正后可重算链。论文 Table 2/Fig. 8 实际使用 PDC5，而方法文字对应 PDC4；旧 Notebook 的阶数和频率轴也存在不可忽略的 provenance 问题。详细说明见 [PDC provenance](docs/pdc_provenance.md)。

### EEG 脑网络

- 30 套 EEG 输入的独立 QC、CSD 和 20 个十秒段；
- 52 个非中线电极、10 个左右半球脑区、5 个频段的绝对虚部相干；
- 唯一无向电极对聚合，不再保存历史表中的正反方向重复；
- 论文文字声明的全局 Wilcoxon+Holm 与历史实际使用的 exact max‑T 两套统计；
- 数值回归、Figure 9、Figure 10、checkpoint 和运行 manifest。

真实重算得到 8,250 行 subject-level tidy 表。修正后全局 Holm 有 2 条显著连接，published-style exact max‑T 有 3 条；历史论文的 5 条结果可以从归档 max‑T 表精确复现，但不能标成 Holm。详细说明见 [brain-network provenance](docs/brain_network_provenance.md)。

## 仓库结构

```text
configs/
  behavior.example.toml
  cmc.example.toml
  pdc.example.toml
  brain_network.example.toml
  paper_behavior_expected.json
  cmc_corrected_expected.json
  cmc_published_expected.json
  cmc_published_legacy_expected.json
  pdc_corrected_expected.json
  pdc_published_legacy_expected.json
  brain_network_corrected_expected.json
  brain_network_published_legacy_expected.json
docs/
  behavior_method.md
  cmc_method.md
  cmc_provenance.md
  pdc_method.md
  pdc_provenance.md
  brain_network_method.md
  brain_network_provenance.md
  data_layout.md
  cmc_data_layout.md
  pdc_data_layout.md
  brain_network_data_layout.md
src/visual_tactile_force/
  behavior*.py
  cmc*.py
  pdc*.py
  brain_network*.py
  legacy_brain_network.py
  legacy_pdc.py
  neuro_registry.py
  statistics.py
tests/
```

## 安装

Python 3.10 或更高版本：

```bash
python -m pip install -e .
python -m pip install -e ".[cmc]"
python -m pip install -e ".[pdc]"
python -m pip install -e ".[network]"
```

第一行安装基础依赖；其余各行分别增加 CMC、PDC 和脑网络的神经信号读取/计算依赖。

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

## 运行 PDC

```bash
cp configs/pdc.example.toml configs/pdc.local.toml
vtf-pdc qc \
  --config configs/pdc.local.toml \
  --output-dir results/runs/pdc_qc

vtf-pdc run \
  --config configs/pdc.local.toml \
  --output-dir results/runs/pdc_corrected \
  --resume
```

PDC 首次运行会生成 Monte Carlo 阈值，之后按设置 hash 缓存。每个 subject-condition 完成后立即写入 checkpoint。真实数据布局见 [PDC data layout](docs/pdc_data_layout.md)。

## 运行 EEG 脑网络

```bash
cp configs/brain_network.example.toml configs/brain_network.local.toml
vtf-network qc \
  --config configs/brain_network.local.toml \
  --output-dir results/runs/brain_network_qc

vtf-network run \
  --config configs/brain_network.local.toml \
  --output-dir results/runs/brain_network_corrected \
  --resume

vtf-network legacy-regression \
  --config configs/brain_network.local.toml \
  --output-dir results/runs/brain_network_legacy
```

`run` 是可重算主管线；`legacy-regression` 只读取两份指定的历史 CSV，用于验证论文发表的 5 条 max‑T 连接。真实数据布局见 [brain-network data layout](docs/brain_network_data_layout.md)。

## 测试

```bash
python -m unittest discover -s tests -v
```

GitHub Actions 会分别运行基础、CMC、PDC 和脑网络测试；测试不使用或上传真实受试数据。
