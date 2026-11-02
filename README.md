# Financial_unsuperviesdAD

金融交易无监督异常检测的 10 种经典方法复现：**Isolation Forest、LOF、kNN、One-Class SVM、PCA-SPE、HBOS、GMM、Autoencoder、VAE、Deep SVDD**。

提供可运行的算法、统一无标签训练和阈值校准、模型保存／加载、CSV 预测及完整留出评价。算法核心复现与金融特征适配的边界见[方法及原论文](docs/METHODS_CN.md)；正式实验配置见[预先固定的协议](docs/PROTOCOL_CN.md)。模型代码不依赖 Kronos 或预训练权重。

本轮沿用 9 组金融数据、10 个任务配置，计划每种方法跑种子 42/43/44，共 300 次。训练统一最多 8192 条，校准最多 100000 条，**测试全量**；异常标签只用于最后评价。统一报告固定 POT 与预先指定的 99% 校准分位阈值，二者分开，POT 失败不隐瞒。

## 安装与运行

Python 3.10+，正式运行环境为 Python 3.12；锁定文件适用于该版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[report,test]'
# 精确依赖版本可参照 requirements-lock.txt
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

覆盖十种方法的分批一致性、检查点重载、距离／残差评分公式、SVDD 固定中心与无 bias、VAE KL／随机样本固定、训练预处理隔离、未知标签屏蔽、POT 退化显式失败及带符号分数平移不变性。测试的人工数据仅用于功能检查，金融结果必须另行真实运行。
