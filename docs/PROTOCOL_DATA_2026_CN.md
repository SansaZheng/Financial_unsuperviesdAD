# 仓库 data/ 数据实验协议与现代评价指标

本轮在本仓库自带的 [`data/raw`](../data/raw) 上运行全部 12 种方法，并在既有 Precision／Recall／F1／AP／ROC-AUC 之外，引入近年时间序列与不平衡异常检测的评价指标。协议在看到任何测试指标前固定；训练、预处理与阈值校准沿用 [v1 协议](PROTOCOL_CN.md) 与 [2026 方法协议](PROTOCOL_2026_CN.md)，未作改动。

## 任务与划分

`scripts/prepare_repo_data.py` 直接从 `data/raw` 构建特征缓存，不再依赖外部 KronosAD 读取器。标签写入独立文件，构建特征时不读取。

| 任务 | 来源文件 | 粒度 | 划分规则 | 特征 |
| --- | --- | --- | --- | --- |
| ulb | ulb_creditcard.zip | 交易 | 与 v1 相同：Time≤86400 训练、≤129600 校准、其后测试 | V1–V28、符号对数金额 |
| banksim | banksim.zip（交易表） | 交易 | 按模拟步（天）累计计数 60/20/20，整步不拆 | 对数金额、年龄段／性别 one-hot、类别与商户哈希；无客户 ID，常数邮编丢弃 |
| s_ffsd | S-FFSD.zip | 交易 | 按 Time 累计 60/20/20 | 对数金额、Location／Type／Target 哈希；标签 2 作为未知 −1 保留在训练／校准中，评价时屏蔽；无 Source 账户 ID |
| sapienza_5S／15S／25S | features_{5S,15S,25S}.csv.gz | 事件序列中的块 | 与 v1 相同：按币种分组的确定性 60/20/20（种子串 `kronosad-supervised-v1`），同一币种不跨分区、不跨分辨率 | 发布者 12 维滚动特征 |
| sd_stock／sd_crypto | SD-FMM 发布数组 | 160 步序列中的点 | 与 v1 相同：按序列内容哈希分组 60/20/20 | 当前值、一阶差分、4／16／32 步滚动均值／标准差／极差 |
| sd_synth | SD-FMM all_DTWsim 数组 | 160 步序列中的点 | 同上（999 条合成序列） | 同上 |
| cardsim | feds-2025017-python.zip 自带模拟器 | 交易 | 固定参数（seed 42、4000 付款人、180 天）生成 626,009 笔，按天累计 60/20/20 | 对数金额、信用卡、远程、对数距离、时刻正余弦、星期 one-hot |

与 v1 共用的四个任务，缓存文件与 v1 公布的哈希逐一比对：24 个文件中 23 个字节一致，仅 `sd_crypto/train.x.f32` 不同（校准／测试文件一致；该任务的 ulb 等成绩在新环境下与 v1 公布值一致到小数点后四位）。未纳入：SynthAML 只有告警表、ADMA21 只有异常区间、`sd_fmm/prices` 仅 16 条序列。sd_synth 与 cardsim 为合成数据，单独标注。

## 方法、训练与阈值

12 种方法（Isolation Forest、LOF、kNN、OCSVM、PCA-SPE、HBOS、GMM、AE、VAE、Deep SVDD、RGLD-ID、RTTAD-TTA）、种子 42/43/44、训练最多 8192 条、校准最多 100,000 条、测试全量；POT（q=0.85、risk=0.01）与 99% 校准分位两套预定阈值；全部超参数固定。RTTAD-TTA 在最多 8192 条无标签测试特征上自适应，为 transductive 设定。与 v1 唯一不同的是运行环境（Linux x86-64 CPU、Python 3.13、torch 2.15）。

## 评价指标

**交易级／点级（全量测试集合并，所有任务）**

- 阈值无关：AUC-PR（Average Precision，主指标，随机参照等于正例率）、AUC-ROC、梯形 PR-AUC、Precision@n（n＝正例数）。
- 固定阈值：Precision、Recall、F1、FPR、**MCC**（Matthews 相关系数，对极度不平衡数据比 F1 更稳健）。
- 参照：标签选优的 best-F1，只用于说明上限，不是任何方法的操作点。

**序列视角（TSB-AD 协议，Liu & Paparrizos, *The Elephant in the Room: Towards a Reliable Time-Series Anomaly Detection Benchmark*, NeurIPS 2024）**

每条含异常的测试序列单独评价后宏平均：Sapienza 以 pump 事件（`pump_index`）为序列，SD-FMM 以 160 步窗口为序列。

- 阈值无关：AUC-PR、AUC-ROC、**VUS-PR／VUS-ROC**（Paparrizos et al., *Volume Under the Surface*, VLDB 2022；TSB-AD 的主指标）、**PATE**（Ghorbani et al., *PATE: Proximity-Aware Time series anomaly Evaluation*, KDD 2024）。
- 固定阈值：标准 F1、PA-F1（Xu et al., WWW 2018 的 point adjustment；已知会高估）、**PA%K-F1 面积**（Kim et al., *Towards a Rigorous Evaluation of Time-series Anomaly Detection*, AAAI 2022；K=0…100 步长 10 的均值）、事件级 F1（事件内任一报警即命中，精度按点计）、范围 F1（Tatbul et al., NeurIPS 2018，α=0.2）、**Affiliation Precision／Recall／F1**（Huet et al., KDD 2022）、PATE-F1。
- 合并事件视角：所有测试事件中事件内／±τ 内至少一次报警的比例（事件召回），所有报警片段中落在某事件 ±τ 内的比例（报警片段精度），及二者的 F1。

容差 τ（行）在看标签前按数据说明固定：Sapienza 取 120 s／块长（5S：24，15S：8，25S：5；发布者说明标注块最多早于公告 120 s），SD-FMM 取 160 步的 5%（8）。VUS 的滑动窗口取 2τ（两侧缓冲 0…τ），PATE 前／后缓冲取 τ（默认包含 0 缓冲组合）。无标注块的 Sapienza 测试事件（2 条）在序列视角跳过，在合并视角保留。无报警的序列 F1 记 0，Affiliation Precision 记为未定义并在宏平均中排除。

VUS、PATE、Affiliation、PA／事件／范围 F1 调用官方实现（TSB-AD 1.5、PATE 0.1.1，见 [`requirements-metrics-2026.txt`](../requirements-metrics-2026.txt)）；PA%K、MCC、合并事件视角在 [`finad/modern_metrics.py`](../finad/modern_metrics.py) 实现，`tests/test_modern_metrics.py` 检查与官方单序列数值一致、完美／空预测的边界与序列切分。

## 复现

```bash
pip install -e '.[report,test,modern2026]'
pip install statsmodels==0.15.0 && pip install --no-deps TSB-AD==1.5 PATE==0.1.1
python scripts/prepare_repo_data.py --output .local_data/feature_cache
python -m finad benchmark --cache .local_data/feature_cache --output .local_data/runs_data2026 \
  --datasets ulb banksim s_ffsd sapienza_5S sapienza_15S sapienza_25S sd_stock sd_crypto sd_synth cardsim \
  --methods iforest lof knn ocsvm pca hbos gmm ae vae deep_svdd rgld2026 rttad2026 --device cpu
python scripts/evaluate_modern.py --runs .local_data/runs_data2026 --cache .local_data/feature_cache --output results/data_2026
python -m finad verify --runs .local_data/runs_data2026 --cache .local_data/feature_cache --output results/data_2026/verification.json
```

正式运行为节省时间把任务分到三个输出目录并行执行（同一方法／种子计划），`evaluate_modern.py` 接受多个 `--runs`，拒绝缺失或重复的运行。
