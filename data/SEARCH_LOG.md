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
