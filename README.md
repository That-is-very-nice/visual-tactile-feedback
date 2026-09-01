# Visual–tactile feedback research analysis

这是用于整理“视觉与振动触觉反馈下等长力控制”研究分析流程的私人仓库。目标是让论文参数、代码逻辑和结果来源长期可追溯，便于后续写作、修改和展示。

当前稳定分析包含行为学、CMC、PDC 与 EEG 脑网络。原始人体数据、个人路径、历史 Notebook 和生成结果不进入 Git。

## 已稳定的分析

### 行为学

- trial 级平均力、标准差和变异系数；
- subject × condition 汇总；
- visual 与 tactile 条件的配对 Wilcoxon 统计；

### CMC

- 30 套 EEG/EMG/annotation 输入注册与质量检查；
- trial 2–5 的 10–60 s 稳态区间，切分为 20 个十秒段；
- CSD 后 EEG 与未整流 EMG 的 multitaper magnitude-squared coherence；
- 左侧感觉运动 ROI 的单一 CMC 指标；
- 三频带配对 Wilcoxon、Holm 校正。

### PDC

- 与 CMC 共用的 30 套输入 QC、CSD 和 20 个十秒段；
- 160 Hz 降采样、显式 25 阶双变量 OLS-VAR 和 EEG↔EMG PDC；
- 1000 次固定种子、频率分辨的 Monte Carlo 阈值；
- 双方向三频带统计、行为相关、回归检查、Figure 7 和 Figure 8。

### EEG 脑网络

- 30 套 EEG 输入的质量检查、CSD 和稳态分段；
- 52 个非中线电极，划分为 10 个左右半球 ROI；
- 计算 delta、theta、alpha、beta 和 gamma 五个频带的绝对虚部相干；
- 对每个频带的 100 个有序 ROI 组合进行双侧配对 Wilcoxon 符号秩检验；
- 在每个频带内采用 Holm step-down 方法控制家族错误率；
- 合并正反方向重复后，输出独立脑区间显著连接及 Figure 9 和 Figure 10。

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
  brain_network_expected.json
docs/
  behavior_method.md
  cmc_method.md
  cmc_provenance.md
  pdc_method.md
  pdc_provenance.md
  brain_network_method.md
  brain_network_results.md
  data_layout.md
  cmc_data_layout.md
  pdc_data_layout.md
  brain_network_data_layout.md
src/visual_tactile_force/
  behavior*.py
  cmc*.py
  pdc*.py
  brain_network*.py
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

## 运行 EEG 脑网络

```bash
cp configs/brain_network.example.toml configs/brain_network.local.toml
vtf-network qc \
  --config configs/brain_network.local.toml \
  --output-dir results/runs/brain_network_qc

vtf-network run \
  --config configs/brain_network.local.toml \
  --output-dir results/runs/brain_network \
  --resume

vtf-network verify \
  --config configs/brain_network.local.toml \
  --output-dir results/runs/brain_network_verification
```

## 测试

```bash
python -m unittest discover -s tests -v
```
