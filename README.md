# Call Volume & Tickets Forecasting

本项目用于预测 `call_volume` 与 `tickets_received`。当前两条序列走的是同一套完整流程：

1. `Chronos/Chronos2` 作为时序基线模型。
2. `direct multi-step supervised` 作为多步监督分支。
3. 对 Chronos 与 direct 结果做自动权重搜索和融合。
4. 基于回测残差做状态分段纠偏。
5. 可选做 lead-wise 分层纠偏。
6. 可选做区间后校准。
7. 针对节假日做零值锚定后处理。
8. 导出标准化 CSV、特征工程 CSV、模型摘要 CSV，并绘图。

其中业务特征已经包含：

- 月末结算/账期特征：`is_month_end_settlement`、`is_billing_cycle_day`、`days_to_month_end`
- 连休前后特征：`is_pre_holiday_1d/2d`、`is_post_holiday_workday_1/2/3`
- 节假日/补班与运营状态特征：`is_holiday`、`is_makeup_workday`、`operational_state_code`

残差纠偏已升级为“每个状态一个小模型（Ridge）+ 状态偏差回退”混合方案，并对未来法定节假日加入零值锚定优化。

## 目录结构

```text
.
├── config.yaml          # 全局配置文件
├── requirements.txt     # 依赖清单
├── src/
│   ├── __init__.py
│   ├── forecast.py          # 主编排流程（Chronos + 纠偏 + 校准）
│   ├── feature_engineering.py  # 特征工程与监督模型
│   ├── output_manager.py    # 输出目录与CSV导出
│   └── visualization.py     # 绘图模块
├── main.py              # 主入口（读取 config + CLI 覆盖）
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt
```

使用配置文件直接运行：

```bash
python main.py
```

每次启动会提示选择运行档位：`quick` 或 `formal`。

- `quick`: 使用加速参数（更快出结果）
- `formal`: 使用配置文件原参数（正式评估）

如果需要非交互执行（如脚本/CI），可显式指定：

```bash
python main.py --profile quick
python main.py --profile formal
```

覆盖目标日期运行：

```bash
python main.py --target_date 2026-04-30
```

覆盖任意参数（会覆盖 config.yaml）：

```bash
python main.py --target_date 2026-04-30 --context_ensemble_topk 5 --interval_coverage 0.85
```

关闭 lead-wise 或区间校准：

```bash
python main.py --target_date 2026-04-30 --disable_leadwise_correction
python main.py --target_date 2026-04-30 --disable_interval_calibration
```

## 完整运转流程

### 1. 入口与参数整合

- 统一入口是 `main.py`
- `main.py` 先读取 `config.yaml`
- CLI 参数会覆盖配置文件中的同名字段
- 然后把参数转发给 `src/forecast.py`

### 2. 数据读取与预处理

`src/forecast.py` 会读取 `data.csv`，并完成以下处理：

- 校验 `date`、`call_volume`、`tickets_received` 是否存在
- 清理 `Unnamed:*` 这类表格导出残留列
- 将日期转换为按天频率的规则时间序列
- 对预测列缺失值补 0
- 校验目标日期必须晚于历史最后一天

### 3. 日历能力初始化

- 如果环境中安装了 `chinese-calendar`，则启用法定节假日与补班日识别
- 如果未安装，则仅区分工作日/周末，不识别法定节假日与补班日

### 4. Chronos 基线回测与未来预测

对 `call_volume` 和 `tickets_received` 分别执行：

- 根据预测目标天数自动对齐回测窗口长度（或使用固定值）
- 搜索最佳 `context_length`，或者按固定 `context_length` 运行
- 当 `context_ensemble_topk > 1` 时，对 Top-K context 做加权融合
- 生成 Chronos 的回测结果与未来分位数预测 `p10/p50/p90`

### 5. Direct 多步监督分支

对两条序列分别构造 direct 特征并训练逐 lead 小模型：

- `call_volume` 以 `tickets_received` 作为参考序列
- `tickets_received` 以 `call_volume` 作为参考序列
- 逐预测步长训练监督模型，候选模型包括 `Ridge` 与 `HistGradientBoostingRegressor`
- 样本过少或特征不可用时自动回退到规则预测
- 导出完整特征工程明细与每个 lead 的模型摘要

### 6. Chronos + Direct 自动融合

- 对两条序列都基于滚动回测搜索最佳融合权重
- 融合回测结果用于后续残差学习
- 融合未来结果作为后续纠偏的输入基线

### 7. 状态分段残差纠偏

融合后的结果继续进入残差模块：

- 按 `workday / weekend / holiday_or_makeup` 分段
- 学习全局偏差、状态偏差、状态-周几偏差、状态-月内阶段偏差
- 在状态样本充足时，额外训练每状态一个 `Ridge` 小模型
- 通过权重搜索选择较优残差组合强度

### 8. Lead-wise 分层纠偏（可选）

- 在残差纠偏之后，再按预测步长学习 lead-wise bias
- 通过搜索 `leadwise_weight` 决定强度
- 可通过 `disable_leadwise_correction` 关闭

### 9. 区间后校准（可选）

- 基于调整后的回测残差，计算目标覆盖率下的经验半径
- 扩张未来 `p10/p90` 区间，提升经验覆盖率稳定性
- 可通过 `disable_interval_calibration` 关闭

### 10. 节假日零值锚定

- 对未来预测中的法定节假日日期做额外后处理
- 结合历史节假日零值占比、holiday median 和 p90 进行收缩
- 主要用于改善春节等长假期间的高估问题

### 11. 导出与绘图

主流程最后导出：

- 标准预测结果 CSV
- call+tickets 合并特征工程 CSV
- call+tickets 合并 direct 模型摘要 CSV
- 回测图、未来预测图、仅基于导出 CSV 的结果图

## 代码模块职责

- `main.py`：读取 `config.yaml`，合并 CLI 参数，并转发到预测主流程
- `src/forecast.py`：主编排流程，负责 Chronos、融合、残差、lead-wise、区间校准、节假日零值锚定、导出和绘图
- `src/feature_engineering.py`：构造 direct 分支特征、训练逐 lead 监督模型、输出特征工程与模型摘要
- `src/output_manager.py`：创建输出目录并导出 CSV
- `src/visualization.py`：生成评估图和预测图

## 配置说明

主要配置在 `config.yaml`：

- `target_date`: 预测截止日期（必填）
- `model_id`: Chronos 模型 ID 或本地模型目录
- `local_files_only`: 是否离线加载模型
- `context_candidates`: `auto` 或自定义候选列表
- `context_search_points`: 自动搜索候选数量（建议 50）
- `context_ensemble_topk`: Top-K context 融合数量
- `backtest_horizon`: 回测窗口天数（若启用自动对齐，会根据预测天数调整）
- `auto_backtest_horizon`: 是否自动对齐回测窗口
- `residual_weight_search_points`: 状态分段残差权重搜索粒度
- `direct_weight_search_points`: `call_volume` 与 `tickets_received` 的 Chronos/direct 融合权重搜索粒度
- `leadwise_weight_search_points`: Lead-wise 纠偏权重搜索粒度
- `leadwise_weight_cap`: Lead-wise 强度上限（守护阈值，建议线上可设 0.8）
- `interval_coverage`: 区间后校准目标覆盖率
- `monitor_recent_days`: 分桶监控报告中“recent”窗口天数（默认 84）
- `monitor_low_sample_threshold`: 报告低样本阈值（默认 12，低于阈值会标记为低置信度）
- `disable_interval_calibration`: 是否关闭区间后校准
- `disable_leadwise_correction`: 是否关闭 lead-wise 分层纠偏

## 输出文件

运行完成后会生成：

- `outputs/csv/forecast_export.csv`: 标准化导出（`date,target_name,p10,p50,p90`）
- `outputs/csv/feature_engineering_merged.csv`: call+tickets 合并后的完整特征工程明细
- `outputs/csv/direct_model_summary_merged.csv`: call+tickets 合并后的各预测步长模型摘要
- `outputs/csv/monitor_bucket_report.csv`: 分桶监控报告（all/recent，含 holiday、post_holiday_workday_1_3、workday_normal 等桶）
- `outputs/csv/monitor_bucket_history_profile.csv`: 历史分桶画像（history_all/history_recent，含样本数、均值、标准差、零值占比）
- `outputs/png/evaluation_results.png`: 回测对比图
- `outputs/png/future_forecast.png`: 历史 + 未来预测图
- `outputs/png/forecast_export_plot.png`: 仅基于导出 CSV 的预测图
- `outputs/png/monitor_bucket_sample_scope.png`: 样本口径对照图（最近窗口双柱 + 全历史折线双轴）

## 业务阅读指南

建议按下面顺序阅读生成的 Markdown 报告：

1. 先看“一页结论”与“风险提示”，快速判断近期是否存在明显风险桶。
2. 再看“样本口径说明”，确认表格中的样本数是 recent 还是 all。
3. 然后看“分场景表现”，对普通工作日、节假日、节后1-3工作日分别解读。
4. 最后看“未来预测摘要”和配图，判断未来高峰与区间宽度。

### 样本口径避免误读

- `回测样本数`：表示最近监控窗口内，真正参与误差评估的样本数。
- `历史样本数(最近窗口)`：表示最近监控窗口内，该场景在真实历史里出现了多少次。
- `历史样本数(全历史)`：表示从数据起始日到当前，全历史里该场景一共出现了多少次。
- `法定节假日` 仅统计法定节假日，不包含普通周末；普通周末单独计入 `周末` 场景。

例如：如果报告中显示“普通工作日 回测样本数 22，历史样本数(最近窗口) 20，历史样本数(全历史) 302”，意思是：

- 最近窗口里普通工作日样本大约 20 多天；
- 全历史里普通工作日远不止 20 天；
- 不能把 recent 样本误解为全历史总量。

### 图表中文化说明

- 当前图表标题、图例、坐标轴均已统一为中文。
- 绘图时会优先尝试加载 Windows 常见中文字体（如 `Microsoft YaHei`、`SimHei`），并自动回退到默认字体，降低中文乱码风险。

## 当前代码流程核对结论

当前代码与上面的流程基本一致，已经核对到以下事实：

1. `main.py` 确实负责读取 `config.yaml` 并转发到 `src/forecast.py`
2. `call_volume` 与 `tickets_received` 都已经接入 Chronos + direct 自动融合
3. 两条序列都已经接入状态分段残差纠偏
4. 两条序列都支持 lead-wise 分层纠偏和区间后校准
5. 两条序列都在最终阶段执行节假日零值锚定
6. 特征工程输出和 direct 模型摘要输出已经是 call+tickets 合并版

另外，本次已补齐一处文档和配置链路的一致性问题：

- `disable_leadwise_correction` 现在可以像其他开关一样，从 `config.yaml` 直接传递到主预测流程

## 兼容说明

`src/forecast.py` 是核心实现文件。推荐使用 `main.py` 作为统一入口，以便配置管理和参数覆盖。
