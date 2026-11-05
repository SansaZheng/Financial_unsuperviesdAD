# 完整金融异常检测数据源清单

核对日期：2026-11-06；共35项，含相邻任务，不表示独立可下载基准的数量。

| ID / 来源 | 类型 | 当前状态 | 许可声明 | Kronos适配 |
| --- | --- | --- | --- | --- |
| `sapienza_pump_dump` · [Sapienza cryptocurrency pump-and-dump (ICCCN 2020)](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset) | market_manipulation | 已下载所列完整发布件 | MIT (upstream repository) | needs_original_trades_or_ohlcv |
| `s_ffsd` · [S-FFSD](https://github.com/AI4Risk/antifraud) | payment_fraud | 已下载所列完整发布件 | GPL-3.0 (repository-wide license; no separate dataset license found) | not_ohlcv |
| `ms_ffsd` · [MS-FFSD](https://github.com/AI4Risk/MS-FFSD) | payment_fraud | 仅入口：许可待核实 | Apache-2.0 (repository); derived from S-FFSD, whose repository is GPL-3.0 | not_ohlcv |
| `sd_fmm` · [SD-FMM released financial manipulation arrays](https://github.com/ys-Dai/SD-FMM) | market_manipulation | 已下载所列完整发布件 | MIT (upstream project) | not_direct_ohlcv |
| `ulb_creditcard` · [ULB/Worldline credit card fraud (2013)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | payment_fraud | 已下载所列完整发布件 | ODbL-1.0 database / DbCL-1.0 contents | not_ohlcv |
| `banksim` · [BankSim](https://www.kaggle.com/datasets/ealaxi/banksim1) | payment_fraud | 已下载所列完整发布件 | CC-BY-NC-SA-4.0 | not_ohlcv |
| `paysim` · [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) | payment_fraud | 仅入口：大文件 | CC-BY-SA-4.0 | not_ohlcv |
| `ieee_cis` · [IEEE-CIS / Vesta Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) | payment_fraud | 需登录接受竞赛条款 | Kaggle competition rules | not_ohlcv |
| `sparkov` · [Sparkov / Kartik Shenoy synthetic card fraud](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | payment_fraud | 仅入口：大文件 | CC0-1.0 (Kaggle listing) | not_ohlcv |
| `fraud_handbook` · [Fraud Detection Handbook simulated transactions](https://github.com/Fraud-Detection-Handbook/simulated-data-raw) | payment_fraud | 仅入口：许可待核实 | Data repository has no license file in inspected snapshot; handbook code GPL-3.0 is not assumed to license separate data | not_ohlcv |
| `tabformer` · [IBM TabFormer synthetic credit card transactions](https://github.com/IBM/TabFormer) | payment_fraud | 仅入口：大文件 | Apache-2.0 (repository; verify data-specific terms at Box) | not_ohlcv |
| `amlworld` · [IBM AMLworld / AML-Data](https://github.com/IBM/AML-Data) | anti_money_laundering | 仅入口：大文件 | CDLA-Sharing-1.0 data; Apache-2.0 repository code | not_ohlcv |
| `amlsim` · [IBM AMLSim simulator](https://github.com/IBM/AMLSim) | anti_money_laundering | 仅模拟器入口 | Apache-2.0 code | not_ohlcv |
| `synthaml` · [SynthAML](https://doi.org/10.6084/m9.figshare.c.6504421.v1) | anti_money_laundering | 部分下载 | CC0-1.0 (both original Figshare file records) | not_ohlcv |
| `amlnet` · [AMLNet](https://zenodo.org/records/16736515) | anti_money_laundering | 仅入口：大文件 | CC-BY-NC-4.0 | not_ohlcv |
| `elliptic` · [Elliptic Bitcoin transaction graph](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) | blockchain_fraud | 仅入口：大文件 | CC-BY-NC-ND-4.0 | not_ohlcv |
| `elliptic_plus` · [Elliptic++](https://github.com/git-disl/EllipticPlusPlus) | blockchain_fraud | 仅入口：许可待核实 | No standalone license in inspected repository; original Elliptic has CC-BY-NC-ND-4.0 | not_ohlcv |
| `hu_pump_dump` · [Hu et al. cryptocurrency pump-and-dump](https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency) | market_manipulation | 仅入口：许可待核实 | No license file found in inspected repository | requires_market_data |
| `adma21` · [ADMA 2021 labeled stock manipulation intervals](https://github.com/zhuye88/ADMA21) | market_manipulation | 已下载标注；未下载行情 | GPL-3.0 repository; Yahoo market prices have separate provider terms | labels_only_need_ohlcv |
| `midsummer_memes` · [A Midsummer Meme’s Dream (USENIX Security 2026)](https://zenodo.org/records/17830944) | market_manipulation | 仅入口：许可待核实 | No explicit license shown in Zenodo/GitHub inspected metadata | ohlcv_available_but_license_unresolved |
| `cardsim` · [Federal Reserve CardSim](https://www.federalreserve.gov/econres/feds/cardsim-a-bayesian-simulator-for-payment-card-fraud-detection-research.htm) | payment_fraud | 已下载代码及公共调查输入；未生成欺诈交易 | Public domain US Federal Reserve Board materials, unless otherwise indicated; see package third-party notices | not_ohlcv |
| `baf` · [Feedzai Bank Account Fraud suite](https://github.com/feedzai/bank-account-fraud) | related_application_fraud | 仅入口：许可待核实 | CONFLICT: Kaggle CC-BY-NC-SA-4.0 vs original datasheet CC-BY-NC-ND-4.0 | not_ohlcv |
| `dgraphfin` · [DGraphFin](https://github.com/DGraphXinye/DGraphFin_baseline) | related_financial_social_graph | 需注册获取 | Provider registration/terms | not_ohlcv |
| `t_finance` · [T-Finance (BWGNN)](https://github.com/squareRoot3/Rethinking-Anomaly-Detection) | related_financial_user_graph | 原提供方入口 | Verify original dataset terms; repository license does not automatically cover external download | not_ohlcv |
| `figraph` · [FiGraph](https://github.com/XNetLab/FiGraph) | related_company_fraud | 仅入口：许可待核实 | CONFLICT: README says CC-BY-NC-4.0 while LICENSE says CC-BY-4.0 | not_ohlcv |
| `bitcoin_alpha` · [Bitcoin Alpha trust network](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) | related_trust_ratings | 参考来源 | SNAP source page; verify reuse terms | not_ohlcv |
| `bitcoin_otc` · [Bitcoin OTC trust network](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) | related_trust_ratings | 参考来源 | SNAP source page; verify reuse terms | not_ohlcv |
| `fi2010` · [FI-2010 limit order book benchmark](https://arxiv.org/abs/1705.03233) | related_unlabeled_market | 原提供方入口 | Original provider research terms | needs_ohlcv_aggregation |
| `lobster_sample` · [LOBSTER public sample data](https://lobsterdata.com/info/DataSamples.php) | related_unlabeled_market | 原提供方入口 | LOBSTER sample terms; no blanket redistribution inferred | needs_trade_bar_aggregation |
| `binance_public` · [Binance Public Data archive](https://github.com/binance/binance-public-data) | related_unlabeled_market | 持续增长的市场数据档案 | Market data provider terms; repository code MIT is not assumed to license raw market data | ohlcv_with_quote_volume |
| `ecommerce_fraud` · [E-commerce Fraud_Data benchmark](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce) | payment_fraud | 仅入口：许可待核实 | No license listed per Amazon FDB; verify original provider | not_ohlcv |
| `elliptic2` · [Elliptic2](https://github.com/MITIBMxGraph/Elliptic2) | blockchain_fraud | 原提供方入口 | Apache-2.0 code; data download governed by Elliptic provider terms | not_ohlcv |
| `berka` · [PKDD 1999 / Berka financial transactions](https://relational.fel.cvut.cz/dataset/Financial) | unlabeled_banking_transactions | 仅入口：许可待核实 | Verify original PKDD provider terms; no explicit redistribution grant confirmed | not_ohlcv |
| `fifar` · [FiFAR / OpenL2D](https://github.com/feedzai/fifar-dataset) | related_application_fraud | 原提供方入口 | Check FiFAR data license and underlying BAF terms | not_ohlcv |
| `unified_crypto_pd` · [Unified cryptocurrency P&D OHLCV 2017–2025](https://zenodo.org/records/21377192) | market_manipulation | 文件访问受限 | Restricted record; no public file license established | ohlcv_but_access_restricted |

## 逐项说明

### sapienza_pump_dump — Sapienza cryptocurrency pump-and-dump (ICCCN 2020)

发布的是事件表和5/15/25秒特征，不是原始逐笔交易；原始交易需上游 downloader 从 Binance 获取。三个分辨率是同一数据集，不计作三个独立数据集。

标签：event list and binary chunk labels。时间字段：UTC event date/hour; feature chunk timestamps。

实际保存 7 文件，16,914,717 字节；见 [原始目录](raw/sapienza_pump_dump/)。

获取入口：[Sapienza cryptocurrency pump-and-dump (ICCCN 2020)](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset)。

### s_ffsd — S-FFSD

模拟的小型金融欺诈数据；2不能当正常，训练不读Labels；保留上游原始ZIP和仓库许可。

标签：Labels: 0 normal, 1 fraud, 2 unlabeled。时间字段：Time (transaction-order index, not real clock time)。

实际保存 3 文件，626,989 字节；见 [原始目录](raw/s_ffsd/)。

获取入口：[S-FFSD](https://github.com/AI4Risk/antifraud)。

### ms_ffsd — MS-FFSD

S-FFSD语义增强派生数据，非独立金融来源；因上游派生许可关系未厘清，本次仅记录原始入口。

标签：Labels: 0 normal, 1 fraud, 2 unlabeled。时间字段：Datetime (synthesized using IEEE-CIS temporal priors)。

获取入口：[MS-FFSD](https://github.com/AI4Risk/MS-FFSD)。

### sd_fmm — SD-FMM released financial manipulation arrays

发布数组包含中国股票案例、加密货币和合成扰动，不能声称获得原始25案例完整行情；部分dictionary NPY是object，禁止自动allow_pickle。

标签：released test label/interval arrays。时间字段：array index; metadata varies by file。

实际保存 18 文件，3,225,150 字节；见 [原始目录](raw/sd_fmm/)。

获取入口：[SD-FMM released financial manipulation arrays](https://github.com/ys-Dai/SD-FMM)。

### ulb_creditcard — ULB/Worldline credit card fraud (2013)

284807交易、492欺诈；V1-V28是匿名PCA特征，不能重命名为OHLC。

标签：Class (0/1)。时间字段：Time (seconds since first transaction)。

实际保存 3 文件，69,183,324 字节；见 [原始目录](raw/ulb_creditcard/)。

获取入口：[ULB/Worldline credit card fraud (2013)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)。

### banksim — BankSim

594643模拟支付；原始包还包含网络版本，勿重复算独立数据集。

标签：fraud (0/1)。时间字段：step (day)。

实际保存 2 文件，13,733,106 字节；见 [原始目录](raw/banksim/)。

获取入口：[BankSim](https://www.kaggle.com/datasets/ealaxi/banksim1)。

### paysim — PaySim

6362620模拟交易；上游提示余额变量受到欺诈取消影响，不能把事后信息作输入。大文件不写入普通Git。

标签：isFraud; isFlaggedFraud is a rule output, not ground truth。时间字段：step (hour)。

获取入口：[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)。

### ieee_cis — IEEE-CIS / Vesta Fraud Detection

真实匿名电商支付；需登录并接受竞赛规则；不能使用未获标签的比赛test计算检测指标。

标签：isFraud only in labeled train files。时间字段：TransactionDT (relative seconds)。

获取入口：[IEEE-CIS / Vesta Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)。

### sparkov — Sparkov / Kartik Shenoy synthetic card fraud

模拟信用卡支付，保留官方train/test；不是ULB真实交易。

标签：is_fraud。时间字段：trans_date_trans_time / unix_time。

获取入口：[Sparkov / Kartik Shenoy synthetic card fraud](https://www.kaggle.com/datasets/kartik2112/fraud-detection)。

### fraud_handbook — Fraud Detection Handbook simulated transactions

183日pickle文件；原始和transformed版本不是独立数据集。未自动反序列化远程pickle。

标签：TX_FRAUD / TX_FRAUD_SCENARIO。时间字段：TX_DATETIME。

获取入口：[Fraud Detection Handbook simulated transactions](https://github.com/Fraud-Detection-Handbook/simulated-data-raw)。

### tabformer — IBM TabFormer synthetic credit card transactions

约24M合成记录，Git LFS/Box分发；不将LFS指针误记为数据。

标签：Is Fraud?。时间字段：Year/Month/Day/Time。

获取入口：[IBM TabFormer synthetic credit card transactions](https://github.com/IBM/TabFormer)。

### amlworld — IBM AMLworld / AML-Data

HI/LI与small/medium/large是一个数据家族；当前Kaggle全包约41.6GB原始数据。

标签：Is Laundering。时间字段：Timestamp。

获取入口：[IBM AMLworld / AML-Data](https://github.com/IBM/AML-Data)。

### amlsim — IBM AMLSim simulator

模拟器和参数不是已下载交易数据；须运行Java/Python生成并记录seed。

标签：SAR/alert patterns depend on simulator configuration。时间字段：simulation step / timestamp。

获取入口：[IBM AMLSim simulator](https://github.com/IBM/AMLSim)。

### synthaml — SynthAML

20000告警和16M以上交易；标签粒度是告警，不能直接当每笔交易的确诊洗钱标签。

标签：alert-level outcome; transactions linked to alerts。时间字段：transaction time; alert date。

实际保存 1 文件，505,481 字节；见 [原始目录](raw/synthaml/)。

获取入口：[SynthAML](https://doi.org/10.6084/m9.figshare.c.6504421.v1)。

### amlnet — AMLNet

1090173模拟交易；CSV约691.3MB，元数据包含风险评分等潜在答案泄漏字段；旧DOI 16482144已弃用。

标签：laundering label and scenario metadata。时间字段：timestamp / metadata.timestamp。

获取入口：[AMLNet](https://zenodo.org/records/16736515)。

### elliptic — Elliptic Bitcoin transaction graph

unknown不能作正常；非商业、不得分发改编版本；保留原始获取入口。

标签：class: illicit/licit/unknown。时间字段：49 time steps。

获取入口：[Elliptic Bitcoin transaction graph](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)。

### elliptic_plus — Elliptic++

交易节点203k、钱包822k；Git LFS/Drive，钱包版与交易版有来源重叠；不重分发许可不明的派生数据。

标签：transaction and wallet class labels, including unknown。时间字段：time step。

获取入口：[Elliptic++](https://github.com/git-disl/EllipticPlusPlus)。

### hu_pump_dump — Hu et al. cryptocurrency pump-and-dump

含Telegram会话pickle与市场特征；未自动加载pickle或发布无明确许可的内容。

标签：Telegram event/session and target coin labels。时间字段：event/session timestamps。

获取入口：[Hu et al. cryptocurrency pump-and-dump](https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency)。

### adma21 — ADMA 2021 labeled stock manipulation intervals

下载论文作者发布的5个事件区间表；行情不在该仓库，需要另从Yahoo获取，部分股票已退市。

标签：published anomaly intervals for TRBO/APPB/AEMD/NBDR/GME。时间字段：calendar date。

实际保存 3 文件，39,233 字节；见 [原始目录](raw/adma21/)。

获取入口：[ADMA 2021 labeled stock manipulation intervals](https://github.com/zhuye88/ADMA21)。

### midsummer_memes — A Midsummer Meme’s Dream (USENIX Security 2026)

包含小时/日OHLCV和操纵研究结果，26.4MB归档；未把潜在操纵算法输出当监管确认真值。

标签：wash trading/LPI/rug pull/pump-and-dump study outputs。时间字段：daily/hourly OHLCV by token。

获取入口：[A Midsummer Meme’s Dream (USENIX Security 2026)](https://zenodo.org/records/17830944)。

### cardsim — Federal Reserve CardSim

已下载官方Python包与2022/2023 Survey and Diary of Consumer Payment Choice公共输入表；包内未含已生成欺诈交易，不把公共调查交易数据当有欺诈真值的基准。

标签：simulation-dependent fraud indicators。时间字段：simulation-dependent。

实际保存 1 文件，4,279,464 字节；见 [原始目录](raw/cardsim/)。

获取入口：[Federal Reserve CardSim](https://www.federalreserve.gov/econres/feds/cardsim-a-bayesian-simulator-for-payment-card-fraud-detection-research.htm)。

### baf — Feedzai Bank Account Fraud suite

6个百万行合成开户申请表，非交易流水；许可冲突本次不镜像。

标签：fraud_bool。时间字段：month。

获取入口：[Feedzai Bank Account Fraud suite](https://github.com/feedzai/bank-account-fraud)。

### dgraphfin — DGraphFin

3700550节点/4300999边；边是紧急联系人关系，不是资金交易；需原站获取。

标签：fraud-user labels; background classes。时间字段：edge time 1..821。

获取入口：[DGraphFin](https://github.com/DGraphXinye/DGraphFin_baseline)。

### t_finance — T-Finance (BWGNN)

39357用户、约42M边，静态金融用户图；Google Drive/Baidu大文件，缺原始交易时序。

标签：anomalous user。时间字段：aggregated graph。

获取入口：[T-Finance (BWGNN)](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)。

### figraph — FiGraph

上市公司属性/关系/年报文本，非支付或K线交易数据；许可冲突暂不镜像。

标签：company abnormality labels。时间字段：year 2014..2022。

获取入口：[FiGraph](https://github.com/XNetLab/FiGraph)。

### bitcoin_alpha — Bitcoin Alpha trust network

这是交易平台信任评分网络，不是交易流水，也没有确诊欺诈标签。

标签：signed trust ratings, not fraud labels。时间字段：rating timestamp。

获取入口：[Bitcoin Alpha trust network](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)。

### bitcoin_otc — Bitcoin OTC trust network

负评分不等于洗钱/交易异常，不作为主异常基准。

标签：signed trust ratings, not fraud labels。时间字段：rating timestamp。

获取入口：[Bitcoin OTC trust network](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)。

### fi2010 — FI-2010 limit order book benchmark

可研究无监督市场状态异常；价格方向标签不能改称操纵真值。

标签：future mid-price direction, not manipulation labels。时间字段：LOB event sequence。

获取入口：[FI-2010 limit order book benchmark](https://arxiv.org/abs/1705.03233)。

### lobster_sample — LOBSTER public sample data

样本订单簿与消息可聚合成交K线；取消/隐藏订单不是自动异常标签。

标签：order event type, not anomaly labels。时间字段：seconds since midnight。

获取入口：[LOBSTER public sample data](https://lobsterdata.com/info/DataSamples.php)。

### binance_public — Binance Public Data archive

原生K线有OHLC、base volume、quote volume，可用于无标签推理，未提供金融异常真值；全量是持续增长档案，不能有限下载穷尽。

标签：none。时间字段：trade time / kline open time。

获取入口：[Binance Public Data archive](https://github.com/binance/binance-public-data)。

### ecommerce_fraud — E-commerce Fraud_Data benchmark

电商购买欺诈，带IP/设备/时间等；原始权属和许可不清，保留入口。

标签：class。时间字段：signup_time / purchase_time。

获取入口：[E-commerce Fraud_Data benchmark](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce)。

### elliptic2 — Elliptic2

49M背景节点、196M交易边，子图洗钱检测；体积巨大、需原提供方页面下载，不能把代码许可当数据许可。

标签：122k labeled subgraphs。时间字段：graph context; verify temporal availability。

获取入口：[Elliptic2](https://github.com/MITIBMxGraph/Elliptic2)。

### berka — PKDD 1999 / Berka financial transactions

约1056320笔银行交易，可做无标签异常探索；贷款违约标签不能当交易欺诈标签，原站访问不稳定。

标签：no transaction fraud labels; loan default status is a different target。时间字段：transaction date。

获取入口：[PKDD 1999 / Berka financial transactions](https://relational.fel.cvut.cz/dataset/Financial)。

### fifar — FiFAR / OpenL2D

基于BAF开户申请和模拟分析员，不是新的支付流水；OpenL2D是后续版本入口，预测分数不应作为无监督输入。

标签：fraud label plus 50 synthetic analyst predictions。时间字段：underlying BAF month。

获取入口：[FiFAR / OpenL2D](https://github.com/feedzai/fifar-dataset)。

### unified_crypto_pd — Unified cryptocurrency P&D OHLCV 2017–2025

记录公开但文件Restricted，需登录检查权限；不是可匿名下载公开数据，未计入下载成功数。

标签：author-provided unified P&D labels。时间字段：daily/intraday OHLCV。

获取入口：[Unified cryptocurrency P&D OHLCV 2017–2025](https://zenodo.org/records/21377192)。
