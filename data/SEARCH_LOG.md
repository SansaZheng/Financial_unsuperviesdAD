# 检索记录（2026-11-06）

目的：寻找可用于无监督金融交易异常检测的公开数据，尽量覆盖主要任务家族；同时为当前Kronos预测误差方法核实是否存在真实OHLCV输入。

范围：公开网页搜索，论文和原作者GitHub、Kaggle原始提供者、Zenodo、Figshare、美国Federal Reserve、SNAP及原数据提供方。先用综述/清单发现候选，再尽量核对原始发布。没有付费购买、登录接受协议、联系作者或提交计算任务。

主要检索组：

- financial fraud anomaly detection public datasets credit card PaySim BankSim AMLSim Elliptic download
- financial market anomaly detection dataset market manipulation pump dump spoofing labeled dataset public
- financial time series anomaly detection dataset OHLCV GitHub labeled anomalies benchmark
- pump dump dataset github market manipulation dataset download
- spoofing market manipulation dataset public order book dataset github
- financial fraud datasets official AMLworld SynthAML Fraud Detection Handbook dataset download
- SynthAML dataset original figshare / AMLNet Zenodo / Elliptic2 official download
- S-FFSD / MS-FFSD / DGraphFin / T-Finance / FiGraph original repositories
- BAF Feedzai dataset license / FiFAR original dataset
- Berka PKDD 1999 financial dataset original download
- cryptocurrency pump and dump OHLCV dataset Zenodo

纳入：有原始发布入口、包含金融支付/银行转账/加密资产交易/市场异常标注或可用于无标签探索的市场数据。保留真实/合成、标签层级、时间维度、是否派生等区别。相邻领域保留在目录的 `related_*` 分类，不视作支付交易。

排除或不重复计算：

- Amazon/Yelp虚假评论、Twitter机器人、恶意URL、工业传感器、NAB社交提及量等不是金融交易异常数据。
- German Credit、贷款违约、营销响应不是交易欺诈；如果作为相邻任务出现，不把目标重新命名为交易异常。
- PSM虽来自金融机构，内容为服务监控指标，不是金融交易流水。
- ADBench等基准中的creditcard派生/采样版本不算新的独立金融数据源。
- 匿名个人仓库中缺来源的“financial anomaly.csv”、重新平衡生成的“2023 credit card”或把模型输出当真值的数据不纳入主基准。
- 不将论文提到但未发布的私有数据视为可下载；不从非授权镜像补齐受限、无许可或竞赛数据。

校验范围：已下载发布件全部核对SHA256；ZIP通过CRC；主要CSV实测行数与列名；数值NPY检查形状/有限值；object NPY不执行反序列化。未下载来源仅核对公开页面或元数据，不声称做过文件级核验。

已发现的差异：SD-FMM发布数组为337个加密货币窗口和27个股票窗口，与论文说明中的338和25不同。FiGraph README与LICENSE的商业使用条件不同；BAF Kaggle与作者数据说明许可不同；MS-FFSD与原S-FFSD的仓库许可关系待厘清；统一OHLCV P&D Zenodo记录公开但文件为Restricted。

本清单属于有记录的公开资源检索，不是证明无遗漏的系统文献综述。服务访问限制、退市资产、许可证变更以及持续增长的市场历史档案意味着无法保证“所有数据已经下载”。逐项状态以CATALOG.md、catalog.json和实际文件manifest为准。

## 补充检索（2026-11-10）：OHLCVA K线异常数据

目的：寻找以 OHLCVA（open/high/low/close/volume/amount）K线为输入、带异常/操纵标注的公开数据，供 Kronos 类方法直接使用。

检索词：OHLCVA K-line anomaly detection dataset；K-line candlestick stock anomaly labeled github；Kronos K-line OHLCVA；stock market manipulation OHLCV labeled China A-share；cryptocurrency pump and dump candlestick OHLCV labeled dataset github；A股 K线 异常检测 数据集 成交额 标注；github pump and dump binance klines labeled。

结果：

- 新增下载：Xu & Livshits（USENIX Security 2019）小时 OHLCVA + PumpOlymp 信号；B0R0koko Binance 事件标签（MIT）；Hu et al. 事件日志（原为仅入口）。
- Zenodo 21377192（统一 P&D OHLCV 2017–2025）：复核仍为文件受限。
- Bristol「Pump, Dump, and then What?」（GitLab，9,191 事件，CCXT 采集 OHLCV）：页面未见许可证，运行环境网络无法访问 GitLab，仅记录入口。
- La Morgia 等后续工作（arXiv 2510.00836）复用 Sapienza 数据，未发布新数据。
- erha-erha/quant-kline-data、Hugging Face A股1分钟OHLCV 等为**无标签**行情，且无独立成交额列或许可未明，不作为异常数据纳入。
- 运行环境只能访问 GitHub：Zenodo、Hugging Face、GitLab、data.binance.vision 均被网络策略拒绝，因此 Binance 事件对应的K线未下载。

许可说明：Xu & Livshits 与 Hu et al. 上游没有许可证文件。此前对无许可来源只记录入口；本次应用户明确要求下载入库，并在 catalog/CATALOG/README 中标注“NO LICENSE FILE”，再分发前需核实。

