# 10 种经典无监督方法：来源、评分与复现范围

检索日期：2026-11-03。选择覆盖不同建模思路的经典方法，而不是将同一算法的若干参数变体凑成十种。它们是用于金融交易的通用异常检测方法，并非全部由金融领域首创，也没有一个公认唯一的“金融十大方法”名单。本仓库复现其核心目标／评分，在统一金融数据上重新实验；不宣称复现原论文的图像网络、加速算法或论文中的数值成绩。

| 方法 | 经典来源 | 本次评分 | 实现及适配边界 |
| --- | --- | --- | --- |
| Isolation Forest | [Liu、Ting、Zhou，ICDM 2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) | 隔离路径越短越异常，使用 `-score_samples` | scikit-learn 1.7.2；100 树、每树最多 256 样本 |
| LOF | [Breunig 等，SIGMOD 2000](https://brava.dbs.ifi.lmu.de/publications/676) | 查询点相对训练邻居的局部密度比 | scikit-learn，k=20，novelty=True；归纳模式，查询点不参与拟合或邻居分布更新，不是原论文联合转导模式 |
| kNN | [Ramaswamy、Rastogi、Shim，SIGMOD 2000](https://webdocs.cs.ualberta.ca/~zaiane/pub/check/ramaswamy.pdf) | 到训练集第 k 近邻的欧氏距离 | k=20，使用库的精确近邻计算；不复写原论文分区剪枝加速策略 |
| One-Class SVM | [Schölkopf 等，Neural Computation 2001](https://research.google/pubs/estimating-the-support-of-a-high-dimensional-distribution/) | `-decision_function`，越远离支持域越异常 | scikit-learn RBF、nu=0.05、gamma=scale；不利用标签估计 nu，不预筛正常训练样本 |
| PCA-SPE | [Jackson、Mudholkar，Technometrics 1979](https://www.tandfonline.com/doi/abs/10.1080/00401706.1979.10489779) | `sum((x-PCA_reconstruct(x))**2)` | 训练方差保留目标 95%，至少留下一个残差方向；不照搬该论文解析阈值，统一用独立校准 |
| HBOS | [Goldstein、Dengel，KI 2012](https://www.dfki.de/en/web/research/projects-and-publications/publication/6431) | 各维归一化直方图密度的负对数和 | 自行实现等宽 10 bins、加一平滑、各维峰值密度归一化；超出训练范围按未见 bin 计分；未实现可变宽 bins 变体 |
| GMM | [Dempster、Laird、Rubin，JRSS-B 1977](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x) 的 EM／有限混合模型基础 | `-log p(x)` | scikit-learn 4 个对角高斯分量、reg_covar=1e-4、最多 200 轮；EM 论文是估计方法基础，不是这套金融检测器的原始论文 |
| Autoencoder | [Hawkins 等，DaWaK 2002](https://togaware.com/papers/dawak02.pdf) 的 replicator／重构异常思想 | 平均平方重构误差 | 自行实现数值特征 MLP：F→128→latent→128→F、ReLU、MSE、Adam；不是原论文特殊离散激活／训练器的逐行重写 |
| VAE | [Kingma、Welling，2013／ICLR 2014](https://arxiv.org/abs/1312.6114) | 高斯重构负对数似然期望＋解析 KL，即负 ELBO 的 MC 估计 | 自行实现对角高斯编码器、单位方差高斯解码器、KL 权重 1；训练重参数化，评分固定 8 个对偶随机样本，确保不随查询顺序／batch 改变统计分布 |
| Deep SVDD | [Ruff 等，ICML 2018](https://proceedings.mlr.press/v80/ruff18a.html) · [作者 PyTorch 实现](https://github.com/lukasruff/Deep-SVDD-PyTorch) | `sum((encoder(x)-c)**2)` | 自行实现 one-class 目标，固定非零中心、无 bias、无有界激活、无 BN；无标签 AE 预训练 200 步，再训练 400 步；用金融 MLP 替代原图像 LeNet，不使用 normal_class 标签筛选 |

`latent=min(16,max(1,F//2))`。所有分数都统一为“越大越异常”。GMM 和 SVM 分数允许为负，POT 拟合的是正的超额值，不需要截断原始分数或用测试值决定平移量。

VAE 编码器为 F→128→64 后接均值／log variance 两个 latent 头，解码器为 latent→64→128→F，隐藏层 ReLU。SVDD 编码器为 F→128→64→latent，隐藏层 LeakyReLU(0.1)；预训练解码器结构反向，无 bias。SVDD 预训练 weight_decay=5e-4，正式单类阶段为 5e-7；中心从预训练后的全部训练抽样求均值，绝对值小于 0.1 的坐标按符号设为 ±0.1（恰为零设为 +0.1）。这些均为已运行代码中的固定配置。

实现接口可对照 [scikit-learn 1.7 的异常检测文档](https://scikit-learn.org/1.7/modules/outlier_detection.html)、[GaussianMixture 文档](https://scikit-learn.org/1.7/modules/generated/sklearn.mixture.GaussianMixture.html)。库包装与自行实现的边界在上表明确列出，包装不代表独立重写底层求解器。

金融任务背景可见 [Schreyer 等的会计分录异常检测研究（2017）](https://arxiv.org/abs/1709.05254)及[信用卡监督／无监督方法比较研究（2019）](https://arxiv.org/abs/1904.10604)。这两项研究支持金融场景背景，不表示本仓库复现了它们全部特征工程或论文结果，也不将其额外计为两种方法。
