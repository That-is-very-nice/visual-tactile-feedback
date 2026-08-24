# Behavior analysis method contract

本文档冻结论文行为学分析中已验证的定义。修改下列参数时，必须同时评估论文回归结果。

## 研究设计

- 正式分析包含 15 名受试。
- visual feedback = `st_no`。
- tactile feedback = `st_tf2`。
- 每名受试、每个条件包含 5 个 60 s trial。
- trial 1 是熟悉试次，不进入分析。
- 分析 trial 2–5 的 10–60 s 稳态阶段。

## 信号处理

- CSV 时间戳单位是毫秒。
- 从相邻时间戳中估计采样率；正式数据期望为 500 Hz。
- 对 measured force 应用 4 阶、5 Hz Butterworth 零相位低通滤波。
- 每个 trial 使用 target force 的中位数作为归一化尺度。

## 行为指标

对每个稳态 trial 计算：

- normalized mean force；
- 样本标准差，`ddof=1`；
- coefficient of variation: `standard deviation / mean force`。

再对每名受试、每个条件的 4 个 trial 取平均。

## 统计契约

- 差值定义为 `st_no - st_tf2`。
- 使用双侧配对 Wilcoxon signed-rank test。
- NaN pair 和零差值在效应量计算中排除。
- 效应量为 `r_z = z / sqrt(n)`，其中 `z` 根据 positive-rank sum 计算，不使用 continuity correction。

## 回归基线

`configs/paper_behavior_expected.json` 保存聚合统计基线，不包含受试级原始数据。正式数据运行必须通过该回归检查后才生成最终图和 manifest。
