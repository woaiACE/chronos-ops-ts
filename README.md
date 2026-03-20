# Call Volume & Tickets Received Forecasting System

这是一个专门为客服团队构建的“每日电话量 (call_volume)”和“工单量 (tickets_received)”预测系统。
考虑到运行环境为无独立显卡的 Intel CPU 机器（16GB 内存），该系统基于 **Chronos Forecasting** 推理管道进行优化，支持 Chronos/Chronos2 模型自动分发：
1. **纯 CPU 推理支持**: 严格依赖 CPU 优化。
2. **内存极简策略**: 采取对 `call_volume` 和 `tickets_received` 依次执行的串行预测逻辑，并在每次切换时强制进行垃圾回收。
3. **可配置上下文窗口**: 支持通过 `--context_length` 固定窗口，或通过 `--context_candidates` 自动搜索最优窗口，不再固定为 512 天。
4. **保留业务特征**: 尊重法定节假日 0 值不进线特征，绝不进行均值或插值填充，确保模型学到最真实的业务脉搏。

---

## 依赖安装与准备
请确保您的环境中有 Python 3.9+，并在仓库根目录运行以下命令安装核心依赖：
```bash
pip install -r requirements.txt
```

> **注意：**
> 1. 您需要确保仓库根目录下存在 `data.csv`，格式请参考：
>    ```
>    date,call_volume,tickets_received,tickets_resolved
>    2023/1/1,72,38,123
>    ...
>    ```
> 2. 当前仓库中附带的 `data.csv` 是通过内置模拟脚本生成的假数据（包含了周末下降和长假 0 值特征），用于端到端跑通与测试。
> 3. 您可以随时用您真实的 `data.csv` 文件将其覆盖，脚本无需任何改动。

---

## 使用说明
核心执行脚本为 `forecast.py`。您只需要传入一个未来需要预测的目标日期 `--target_date`（格式 `YYYY-MM-DD`）即可启动整个端到端的**回测验证 + 未来预测**流程。

**基础示例命令：**
```bash
python forecast.py --target_date 2026-04-30
```
*提示：指定的 `--target_date` 必须晚于 `data.csv` 中的最后一天记录，否则脚本会抛出友好提示。*

### 1. 网络受限时，使用镜像地址
如果您的网络无法直接访问 Hugging Face，可以通过 `--hf_endpoint` 指定镜像地址：

```bash
python forecast.py --target_date 2026-04-30 --hf_endpoint https://hf-mirror.com
```

### 2. 模型已在缓存或本地文件夹中，强制离线加载
如果模型已经存在于本机缓存中，或者已提前下载到本地，可通过 `--local_files_only` 禁止联网请求：

```bash
python forecast.py --target_date 2026-04-30 --model_id amazon/chronos-t5-mini --local_files_only
```

### 3. 通过 `--model_id` 指定本地模型目录
如果您已经将模型下载到仓库下的本地目录，可以把目录路径直接传给 `--model_id`：

```bash
python forecast.py --target_date 2026-04-30 --model_id local_chronos_model_mini --local_files_only
```

```bash
python forecast.py --target_date 2026-04-30 --model_id local_chronos2_model --local_files_only
```

### 4. 通过滚动回测自动搜索更优的 `context_length`
如果您希望针对当前业务数据自动搜索更优的历史窗口长度，可以使用以下参数：

- `--backtest_horizon`: 回测预测天数。
- `--rolling_windows`: 滚动回测窗口数量。
- `--context_candidates`: 待搜索的历史窗口长度列表。
- `--context_length`: 如果您已经知道最优窗口长度，可直接固定，不再自动搜索。

**自动搜索示例：**
```bash
python forecast.py --target_date 2026-04-30 --model_id local_chronos2_model --local_files_only --backtest_horizon 14 --rolling_windows 4 --context_candidates 90,180,365,512,730
```

**固定窗口长度示例：**
```bash
python forecast.py --target_date 2026-04-30 --model_id local_chronos2_model --local_files_only --context_length 365 --backtest_horizon 14 --rolling_windows 4
```

如果您使用的是 Windows PowerShell，也可以先设置镜像环境变量，再运行脚本：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python forecast.py --target_date 2026-04-30
```

---

## 指标与可视化说明
执行结束后，除了终端打印日志，脚本将在当前目录生成两张专业的业务图表，辅助排班与决策：

### 1. 评测对比图 (`evaluation_results.png`)
该图展示了在进行未来预测前的回测效果。当前脚本默认采用**滚动回测**（可通过参数调整）：
- **回测跨度**: 每个窗口预测长度由 `--backtest_horizon` 决定（默认 14 天）。
- **窗口数量**: 由 `--rolling_windows` 决定（默认 4 个窗口）。
- **图中曲线**: 展示最近一个回测窗口的 Actual vs Predicted(p50)。
- **终端指标**: 打印的是所有回测窗口的平均 `sMAPE` 与平均 `RMSE`。
- **纵向排布子图**: 上图为电话量，下图为工单量。
- **业务意义**: 通过实际曲线（Actual）与预测中位数曲线（Predicted p50）的贴合度，让业务人员直观信任模型能力。
- **关联指标**: 终端会打印 sMAPE (对称平均绝对百分比误差) 和 RMSE (均方根误差)。
  - **sMAPE**: 反映预测偏离真实值的相对百分比大小。相较于传统 MAPE，当历史真实数据存在 0 值（如节假日）时，sMAPE 不会产生无穷大的计算溢出，表现更稳定。越小越好。
  - **RMSE**: 反映预测偏离真实值的绝对规模波动。越小越好。

### 2. 未来预测图 (`future_forecast.png`)
基于全量历史数据，模型自动预测从最后一天起，直到您传入的 `--target_date` 期间的走势。预测时使用的历史窗口长度来自：
- 固定 `--context_length`，或
- 自动搜索得到的最优 `context_length`（由 `--context_candidates` + 滚动回测决定）。
- **历史衔接**: 包含过去 90 天的真实数据走向（黑线）提供近因上下文。
- **点预测线**: 未来走势的中位数预期线（p50）。
- **阴影冗余区域**: 提供了 `p10` 至 `p90` 的 80% 预测置信区间（半透明填充色）。
- **排班业务价值**: 注释中标明的 `p90` 曲线代表了极端情况下的预估上限，是客服业务线进行**人力排班冗余基线**的关键参考点。
