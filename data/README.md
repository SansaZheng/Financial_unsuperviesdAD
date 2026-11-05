# 金融交易异常检测数据目录

检索日期：**2026-11-06**。本次整理 **35 个数据源/数据家族**，覆盖支付欺诈、反洗钱、区块链及市场操纵，并把开户申请、公司财务异常、信任评分等相邻任务单独标出。**这不是“全球所有数据均已搜全或下载”的声明。** 可匿名访问的网页、可下载数据和可再分发数据是三种不同状态。

实际存入本仓库的是 **8 个来源包的 38 个文件，共 108,507,464 字节（约 103.5 MiB）**。其中只有下表前五项是已获取完整的所选上游数据发布件；其余是告警表、标注或模拟器与输入，不冒充完整欺诈交易数据。

## 已下载内容

| 数据源 | 本仓库内容 | 文件核验结果 | 适配当前 Kronos |
| --- | --- | --- | --- |
| ULB / Worldline | 原始完整 `creditcard.csv` ZIP | 284,807 笔，492 欺诈 | 不可直接使用，匿名 PCA 支付特征不是 OHLCV |
| BankSim | 原始完整 ZIP，含交易表及网络表 | 594,643 笔，7,200 欺诈；两表是同一批交易 | 不可直接使用 |
| Sapienza pump-and-dump | 事件表、组表、完整 5/15/25 秒特征发布件 | 1,110 事件；821,307 / 584,104 / 482,157 特征行 | 需要另获原始成交/K线；均价等不能重命名为 OHLC |
| S-FFSD | 原始完整 ZIP | 77,881 行：正常24,387、欺诈5,256、未标注48,238 | 不可直接使用 |
| SD-FMM 发布数组 | 上游全部 NPY 发布文件 | 加密货币337×160×1，股票27×160×1，另有价格/合成数组 | 单变量及处理后窗口，不能伪造六维K线 |
| SynthAML | **仅**完整 `synthetic_alerts.csv` | 20,000 告警；**未下载934.6MB交易表** | 不可直接使用 |
| ADMA 2021 | 作者原始 `data.md` 中的5个股票异常区间及说明 | TRBO、APPB、AEMD、NBDR、GME；**不含行情** | 需另外取得对应 OHLCV |
| Federal Reserve CardSim | 完整官方 Python ZIP，含2022/2023公共支付调查输入表 | ZIP CRC通过；**未生成带欺诈标签的模拟交易** | 不可直接使用 |

SD-FMM 的论文介绍提到25个中国股票案例、338个加密货币案例，但当前发布数组的数量如表所示。保留上游原文件，不擅自修正或声称已获取论文完整原始数据。NPY中6个object数组只读头部，不使用 `allow_pickle=True` 执行反序列化。

## 文件入口

- [`catalog.json`](catalog.json)：机器可读目录，记录主来源、许可声明、标签粒度、时间字段、Kronos兼容性、下载或受限状态、固定来源版本与下载地址。
- [`CATALOG.md`](CATALOG.md)：35项完整清单、逐项未下载原因与获取入口。
- [`raw/`](raw/)：实际下载的原始文件，保留压缩格式，不提交解压后的重复副本。
- [`raw/download_manifest.json`](raw/download_manifest.json)：逐文件来源URL、字节数、SHA256、获取时间和许可。
- [`integrity_report.json`](integrity_report.json)：SHA256、ZIP CRC、CSV行数/列名/标签数量、NPY形状及object识别。
- [`source_evidence/`](source_evidence/)：Kaggle/Figshare元数据核验快照和许可冲突记录。
- [`SEARCH_LOG.md`](SEARCH_LOG.md)：检索范围、关键词、纳入/排除规则及尚未解决的问题。

## 下载与校验

从项目根目录运行；下载工具仅用 Python 标准库。离线结构审计需要项目的 NumPy 依赖。

```bash
# 查看所有来源及获取状态
python scripts/download_datasets.py --list

# 重新下载本次可直接分发的文件；已存在文件先核对SHA256，匹配则跳过
python scripts/download_datasets.py

# 选择单个来源
python scripts/download_datasets.py --ids ulb_creditcard banksim sapienza_pump_dump

# 离线复核所有已下载文件
python scripts/audit_datasets.py
```

工具默认单文件上限95 MiB、每次新增总量256 MiB。不覆盖未核验的已有文件；下载中断或校验失败会删除该次临时文件，报错并返回非零退出状态。不会自动解压、加载pickle、接受竞赛条款、绕过登录或运行外部模拟器。源站改变文件时会失败，不能静默把新版当原始版本。完整再现时首先对照已提交的manifest。

本次执行机器仅剩约4.9GiB空间，且没有可用的授权 `/data` 或 `/date` 大容量目录，因此大型数据只保留可复现入口。以下命令应在具有该授权目录和足够空间的环境中执行；不要将大型原始数据放入普通 Git 历史。

```bash
# 原始PaySim、Sparkov或Elliptic：依原始条款获取，输出到已授权的大容量目录
python scripts/download_datasets.py --ids paysim sparkov elliptic \
  --include-large --output /data/kronos-ad-datasets \
  --max-file-mib 2048 --max-total-mib 4096

# SynthAML只缺交易表，需934.6MB空间；会同时获取相应告警表
python scripts/download_datasets.py --ids synthaml --include-large \
  --output /data/kronos-ad-datasets --max-file-mib 1024 --max-total-mib 1100

# IBM AMLworld全量约41.6GB原始数据，应优先在源站选择HI/LI-small单个版本
kaggle datasets files -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
  -f HI-Small_Trans.csv -p /data/kronos-ad-datasets/amlworld

# IEEE-CIS需本人先在官方页面接受竞赛规则，再使用已配置的Kaggle账号
kaggle competitions download -c ieee-fraud-detection \
  -p /data/kronos-ad-datasets/ieee_cis
```

源站当前匿名下载不需要登录的文件未来可能改变访问要求；工具会如实报错。IBM单文件名也应以 `kaggle datasets files` 的实际返回为准。TabFormer通过官方Box/Git LFS获取，Elliptic2通过原提供方页面获取；不把下载失败的LFS指针计作数据。B​​AF、FiGraph等许可冲突不由 `--include-large` 绕过。

## 无监督评估约定

1. **无监督是协议，不是“必须没有标签”。** 训练、预处理和阈值校准不得读取欺诈标签；标签单独留给最终评估。按标签筛出正常训练样本属于normal-only/oracle协议，必须单独命名，不能冒充完全无监督。
2. 有时间字段时按时间/资产/账户业务逻辑划分参考、校准、测试，不能任意打乱。仅提供窗口或静态图的来源不能自动保证真正时间外推。
3. 未标注不等于正常：S-FFSD的2、Elliptic的unknown应保持未知状态。没有异常真值的数据只能做探索或另加明确标识的人工注入实验。
4. `isFlaggedFraud`、人工审核结果、模型风险分、答案衍生列和未来统计不能作为无监督输入。开户违约、信任负评分、未来价格方向也不能充当交易欺诈标签。
5. 原始支付交易、金融图和单变量窗口不满足本项目的OHLCV接口。不要用支付金额、PCA列或图节点特征伪造开高低收。真正市场成交数据可按资产和固定频率构造真实OHLCV；成交额只有真实或可明确追溯到逐笔价格×数量时才可用，否则排除该维诊断。
6. 多分辨率、原始/网络表示、派生语义版、重采样版和共享来源不能按独立数据集重复计算泛化证据。

## 许可和出处

各目录保留自己的原许可；这些数据不因存入本仓库而获得统一的软件许可。

- ULB/Worldline：数据由 ULB Machine Learning Group 与 Worldline 合作提供。来源为 [原始Kaggle发布](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)，数据库 ODbL-1.0、内容 DbCL-1.0，完整条款在对应原始目录中。原始ZIP未改动。相关引用：Dal Pozzolo et al., *Calibrating Probability with Undersampling for Unbalanced Classification*, CIDM 2015。
- BankSim：Edgar Alonso Lopez-Rojas 与 Stefan Axelsson，*BankSim: A Bank Payments Simulator for Fraud Detection Research*, EMSS 2014；[原发布](https://www.kaggle.com/datasets/ealaxi/banksim1)，CC-BY-NC-SA-4.0。原始ZIP未改动。
- Sapienza：La Morgia, Mei, Sassi, Stefa，*Pump and Dumps in the Bitcoin Era*, ICCCN 2020；[原发布](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset)，保留MIT许可及README中的引用信息，文件未改动。
- S-FFSD：AI4Risk AntiFraud，参见 [原README](https://github.com/AI4Risk/antifraud) 中 GTAN 等论文；保留上游仓库GPL-3.0许可与原始ZIP。未发现单独数据许可，记录仓库声明，不将该声明延伸到其外部Yelp/Amazon数据。
- SD-FMM：[作者发布](https://github.com/ys-Dai/SD-FMM)，*Detecting Market Manipulation with Dual-branch Self-supervised Learning*；保留MIT许可及README，文件未改动。
- SynthAML：Jensen et al., *A synthetic data set to benchmark anti-money laundering methods*, Scientific Data 10, 661 (2023)；[原始Figshare合集](https://doi.org/10.6084/m9.figshare.c.6504421.v1)。数据文件元数据声明CC0，与论文正文CC-BY-4.0区分，告警表未改动。
- ADMA 2021：Jillian Tallboys、Ye Zhu、Sutharshan Rajasegarar，*Identification of Stock Market Manipulation with Deep Learning*；[作者发布](https://github.com/zhuye88/ADMA21)，保留GPL-3.0和原始标注说明，未镜像Yahoo行情。
- CardSim：US Federal Reserve Board，*CardSim: A Bayesian Simulator for Payment Card Fraud Detection Research*，2025；[官方页面](https://www.federalreserve.gov/econres/feds/cardsim-a-bayesian-simulator-for-payment-card-fraud-detection-research.htm)。包内LICENSE说明代码为美国政府雇员公务作品，属于公有领域；支付调查输入的引用参见包内README，原始ZIP未改动。
