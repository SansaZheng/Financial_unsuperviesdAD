# Financial_unsuperviesdAD

金融交易无监督异常检测：10 种经典方法及 2 种 2026 预印本方法核心的金融适配。

经典方法：**Isolation Forest、LOF、kNN、One-Class SVM、PCA-SPE、HBOS、GMM、Autoencoder、VAE、Deep SVDD**。

提供可运行的算法、统一无标签训练和阈值校准、模型保存／加载、CSV 预测及完整留出评价。算法核心复现与金融特征适配的边界见[方法及原论文](docs/METHODS_CN.md)；正式实验配置见[预先固定的协议](docs/PROTOCOL_CN.md)。模型代码不依赖 Kronos 或预训练权重。

本轮已完成 **9 组金融数据、10 个任务配置、10 种方法、3 个种子（42/43/44），共 300 次真实实验**。训练统一最多 8192 条，校准最多 100000 条，**测试全量**；异常标签只用于最后评价。统一报告固定 POT 与预先指定的 99% 校准分位阈值，二者分开；本轮 POT 全部成功。

**[查看完整中文结果](results/v1/README_CN.md)** · [结果解读](docs/RESULTS_INTERPRETATION_CN.md) · [600 套逐种子操作点](results/v1/metrics.csv) · [均值和标准差](results/v1/summary.csv) · [核验记录](results/v1/verification.json)

25 项测试通过；300 套测试分数、混淆矩阵、模型重载评分与 70 个输入文件哈希已核验。首轮 GMM 数值错误修复后，全部方法在新目录重新完成 300 次运行，详见[修复记录](docs/NUMERICAL_FIX_CN.md)。这些是经典目标与金融特征适配结果，不是原论文数值成绩的复刻，不能与此前 Kronos 实验直接混排。

## 新增 2026 方法

- **RGLD-ID**：基于 [RGLD（2026-06）](https://arxiv.org/abs/2606.28970) 的随机全局／局部密度与排名集成，采用训练集固定排名的归纳适配。
- **RTTAD-TTA**：基于 [RTTAD（2026-05）](https://arxiv.org/abs/2605.10242) 的掩码重构、GammaGMM 置信选择及测试时对比自适应，使用最多 8192 条无标签测试特征。

两者均为独立核心实现，论文未完整给出的设置已显式补齐，**不是原论文数据和数值成绩的逐项复刻**；RTTAD 与归纳方法的测试信息预算不同。自适应选择失败的运行保留基础模型，逐种子记录实际更新轮数。

**[2026 新方法的 60 次实验](results/2026/README_CN.md)** · [与原 300 次实验的对照](results/2026/COMPARISON_CN.md) · [结果解读](docs/RESULTS_2026_INTERPRETATION_CN.md) · [适配公式与实现边界](docs/METHODS_2026_CN.md) · [固定协议](docs/PROTOCOL_2026_CN.md) · [自适应核验](results/2026/adaptation_audit.csv) · [结果核验](results/2026/verification.json)

本轮未显示新方法全面优于经典方法。RTTAD 的 30 次运行中有 14 次未发生有效更新，最终网络与基础检查点已核对一致；这些运行如实保留。

新增实验沿用全部 9 组数据、10 个任务和三个种子，与 v1 使用相同训练／校准索引、完整测试分区和两套预定阈值；合计 **360 次运行、720 套阈值操作点**。旧 `results/v1` 原样保留。测试扩展至 34 项，并在 Python 3.10／3.12 CI 运行。

```bash
pip install -e '.[report,test,modern2026]'
# Python 3.12 正式实验依赖：requirements-2026-lock.txt
python -m finad benchmark --cache /path/to/feature_cache \
  --output .local_data/runs_2026 --methods rgld2026 rttad2026 --device cpu
python -m finad report --runs .local_data/runs_2026 --output results/2026
python -m finad verify --runs .local_data/runs_2026 \
  --cache /path/to/feature_cache --output results/2026/verification.json
python scripts/compare_2026.py --old results/v1 --new results/2026
```

支持 MPS／CUDA；正式新增实验使用本地 CPU／MPS。模型、原模型、适配索引及无损压缩测试分数保存在私有运行目录，不随聚合报告上传。两种实现均从头拟合，不依赖预训练权重；PyThresh 用于 RTTAD 的 GammaGMM 选择。`benchmark` 不指定 `--methods` 时仍默认运行原十种方法。

## 安装与运行

Python 3.10+，正式运行环境为 Python 3.12；锁定文件适用于该版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[report,test]'
# 使用本轮 Python 3.12 的完整依赖版本：
# pip install -r requirements-lock.txt
# pip install -e . --no-deps
```

已有上一轮缓存时可直接运行：

```bash
python -m finad benchmark --cache /path/to/feature_cache \
  --output .local_data/runs_v1 --device cpu
# 支持的机器可选 --device mps 或 cuda；传统方法仍使用 CPU。
python -m finad report --runs .local_data/runs_v1 --output results/v1
python -m finad verify --runs .local_data/runs_v1 \
  --cache /path/to/feature_cache --output results/v1/verification.json
```

只跑部分方法／数据：

```bash
python -m finad benchmark --cache /path/to/feature_cache --output .local_data/ulb_demo \
  --datasets ulb --methods iforest lof knn ocsvm pca hbos gmm ae vae deep_svdd --seeds 42
```

同一输出目录只允许完全相同配置续跑；改变参数需用新目录。`report` 拒绝缺失／重复运行，并拒绝覆盖既有报告。

`--batch-size` 控制校准／测试评分的分块，默认 8192；神经网络训练 batch 固定最多 256。不同设备／底层库不保证逐位相同，报告保留实际环境与逐种子结果。

## 获取／准备数据

本仓库不再重复上传大体积数据。公开数据及各自许可保留在 [KronosAD 固定版本](https://github.com/SansaZheng/KronosAD/tree/2c0e70eb9461081c49a4ef4cfd62f4560f345477/data/supervised_financial)，IEEE-CIS 原始数据只存本地。准备包装器只调用该固定版本的数据读取工具，检测器本身独立运行：

```bash
git clone https://github.com/SansaZheng/KronosAD.git /path/to/KronosAD
git -C /path/to/KronosAD checkout 2c0e70eb9461081c49a4ef4cfd62f4560f345477
python scripts/prepare_financial_data.py --source-repo /path/to/KronosAD \
  --output .local_data/feature_cache --ieee-dir /path/to/authorized/ieee-data
```

没有 IEEE-CIS 授权数据时在 `--datasets` 明确列出其他配置并省略 `ieee_cis`。不自动登录、接受规则、上传受限文件。

自有数值 CSV 可以先按时间／实体在外部分成 train/calibration/test，再导入；显式指定特征，不能包含目标、交易 ID 或未来不可用信息：

```bash
python scripts/prepare_csv.py --train train.csv --calibration calibration.csv --test test.csv \
  --name custom --output .local_data/cache --features amount previous_balance feature3 \
  --target is_anomaly
python -m finad benchmark --cache .local_data/cache --datasets custom --output .local_data/custom
```

标签须为 0=正常、1=异常、-1=未知；训练／校准可以没有目标列，未知测试不会产生虚构指标。

## 已保存模型预测新数据

```bash
python -m finad predict --run .local_data/runs_v1/ulb/iforest_seed42 \
  --csv new_features.csv --output predictions.csv
```

CSV 需包含原训练特征列。输出行位置、异常分数与两个阈值的标签；加载既有预处理，不在新数据重新拟合。仅加载你创建或信任的 joblib 检查点。

## 测试

```bash
pytest -q
```

覆盖十二种方法的分批一致性、检查点重载、距离／残差评分公式、SVDD 固定中心与无 bias、VAE KL／随机样本固定、训练预处理隔离、未知标签屏蔽、POT 退化显式失败及带符号分数平移不变性。测试的人工数据仅用于功能检查，金融结果必须另行真实运行。

新增测试包括重复交易记录的 RGLD 距离／分批稳定性、RTTAD 双符号梯度更新、GammaGMM 随机状态隔离、冻结检查点、无训练／校准标签运行及无损分数归档。
