# CAE-IF、EB-GAD、Audit-HCL：零标签金融适配协议（2026-11-14）

本协议的科学配置在金融指标用于比较前固定。门禁检查使用固定1/2更新的烟雾预算，不作为金融实验结果；架构核对补充了CAE紧凑瓶颈及其GPU重载门禁，未按金融指标选择配置。正式计算期间加入了一个遮罩填充索引的等价执行优化：不改变邻居、注意力公式、采样或超参数，独立GPU测试的前向、梯度及三步参数差均为0，减少无效梯度集中到节点0的竞争；保留原参考作业，结果注明实际源提交。目标是仓库金融数据上的可审计零标签比较，**不是原论文数据集或成绩的精确复刻**。原有 `results/v1`、`results/2026` 和原始数据不改动。

## 方法依据与复现边界

1. **CAE-IF**：[IEEE 原文入口](https://ieeexplore.ieee.org/document/11471437)，DOI `10.1109/ICMLA66185.2025.00066`；另核对[作者博士论文的大学元数据](https://digitalcommons.fau.edu/etd_general/223/)，下载全文返回403。只取得公开摘要，未取得全文、原架构及官方代码。本实现 `cae_if_core` 是独立核心适配：特征轴上的 Conv1d（1→16→4，核3，第二层步长2），自适应均值池化到 `max(1,floor(d/8))` 个位置、4通道，展平潜在维约为输入的一半；解码后以 MSE 无监督训练。编码展平后拟合100棵 Isolation Forest，max_samples=256。Adam lr=.001、400次更新、batch≤256。工程特征顺序固定但没有天然空间邻接，因此卷积假设本身需由实验检验。不可称原架构精确复现。
2. **EB-GAD**：[论文](https://arxiv.org/html/2609.38424v1)、[作者仓库](https://github.com/heraclixus/EBGAD)，固定提交 `cba2969e390ae6cd4381fe86393fa6cfcfc40d13`。使用作者库的模板、经验贝叶斯边缘似然拟合、J*/R/轨迹 profile/条件残差候选以及发布入口的无标签 mirror-null 选择器；**该选择器与论文附录 B.4 有序规则不同**，标记 `ebgad_release_zero`。保持发布入口选中统计量的原始分数。采用 paper 特征预处理顺序、无PCA、完整float64谱、无标签子图筛选。无需神经网络参数训练；每个校准/测试图独立转导拟合，评估图特征及拓扑可见，标签完全不可见。跨阶段原始分数尺度可能漂移，统一校准阈值的表现应照实报告。固定算法seed=0，三种子重复不代表三次独立随机拟合。上游无 LICENSE，因此不将其代码复制进本仓库；运行时下载到外部缓存并验证完整归档SHA256。
3. **Audit-HCL**：[Scientific Reports 原文](https://www.nature.com/articles/s41598-026-58938-5)。无公开官方代码，本实现 `audit_hcl_zero_adapted` 按公式独立实现：类型投影→两层多头元路径注意力（128维、8头，第二层残差，LayerNorm）→四快照GRU；结构视图按Eq.10进行扰动，时间视图按Eq.11使用原始快照序列；两个128→256→64投影头、对称InfoNCE、256个无标签困难负例，温度.07，负例采样温度.5；边丢弃.3、特征遮蔽.2、每路径最多64邻居。AdamW lr=.0005、weight_decay=.00001、梯度裁剪1。**不做监督微调、标签筛选、标签验证早停、标签阈值搜索。**

Audit-HCL 明确适配：原金融审计模式中的所有权/设备关系在现有数据中不可得，改用真实支付直接关系、共享收款方、共享付款方三种路径，不虚构设备关系。路径超过64时固定种子均匀抽样一次，不声称原文每训练步重采样。混合权重采用论文无标签段落的.5，而非实现段落经过有标签验证选出的.6。按论文“丢弃投影头”说明在推理使用原128维结构/时间表示；原文评分公式又使用投影表示，存在歧义，故此处预先明确选择。评分为 `.5*(1-cos(structural,temporal))+.5*mean_GRU_drift`；仅在账户出现时更新GRU，首次出现不计算漂移。困难负例使用无标签内存库，每25次更新刷新；两个方向的采样分布在块内固定，使用当前编码结果直接刷新（未补设原文未给定的EMA系数），batch256；**固定200次参数更新**以控制本次预算（不是原文200 epochs），不使用其验证AUC早停。窗口内使用完整离线图，不能解释为逐笔实时预警。

## 两个分开比较的任务轨道

- **交易/时序轨道**：ULB、BankSim、S-FFSD、Sapienza 5S/15S/25S、SD-FMM stock/crypto/synth 共9组。CAE-IF与新跑的 Isolation Forest、AE、PCA比较。沿用既有数据划分：ULB按日；BankSim/S-FFSD按时间累计60/20/20且相同时间戳不跨分区；Sapienza按币种分组；SD-FMM按去重后的序列分组。SD-FMM用发布者已处理序列，不宣称原始因果行情。数据准备脚本来源于本仓库分支 `claude/data-2026-modern-metrics` 的已完成工作；本次不重启/改写那个分支的360次实验。
- **原生关系图轨道**：仅 BankSim 和 S-FFSD；其他数据没有真实账户边，不伪造kNN图。每阶段将时间有序交易分成4个快照。为保证完整谱可计算，按固定字符串 `financial-zero-20261114:<sender_id>` 的 SHA256 顺序选择最多4096个付款账户，保留其全部阶段交易及所有收款方；**只评价这些付款账户**，不同阶段可有不同账户。此为有公开确定性预算的图子集，不是所有原始交易的测试结果。实际图中“发送角色”和“接收角色”使用不同节点类型；现有两数据的两端ID不重叠，不虚构身份映射。
- 图每快照特征仅为角色、入/出交易次数log、入/出金额signed-log的均值/标准差/最大值（10维）。比较器使用4快照展平的40维特征；CAE-IF、IF、AE、PCA只在训练付款账户拟合；图方法另外使用真实交易边及对应收款方属性，Audit用完整训练图自监督，EB在各评估图转导拟合，信息范围差异公开。所有方法使用相同测试付款账户与标签。**两轨道检测单位、样本和信息预算不同，不能混排名次或平均成一个总分。**

图账户标签仅在特征/边全部固定后聚合：阶段内任一已知欺诈交易→1；无欺诈且有未知交易→未知；全部已知正常→0。仅作为账户异常代理标签，不等于司法确认的洗钱/欺诈账户。S-FFSD 的交易 label=2 保留为未知，参与无标签拟合/阈值校准，仅在最终指标计算排除；不先筛掉未知节点。

## 固定评价标准与预算

种子42/43/44。交易轨道训练均匀抽样最多8192行（不按标签过滤），校准最多100000行，测试全量；图轨道使用上述固定原生关系子图。共同的鲁棒缩放器仅由训练特征拟合（median/IQR，std回退，clip[-8,8]）；图共用快照级训练缩放器。EB-GAD作者库随后在每个校准/测试图内部再次按mean/std标准化，并拟合该图的经验贝叶斯参数；这是额外的无标签转导步骤，不能称为全流程归纳、训练后固定的预处理。AE同原基线400更新、128隐藏、latent=min(16,d/2)；PCA保留95%方差但至少一维残差；IF100棵、max_samples≤256。超参数不看测试指标选择。

所有方法预先报告两套阈值：POT（初始q=.85、risk=.01、至少20个非退化超额样本）与独立99%校准分位数。POT失败必须显示N/A，不以分位数冒充POT。`score >= threshold`报警；因此有大量并列分数时，99%分位数不保证只有1%报警。

指标为未经点调整的 AP（Average Precision）、梯形PR-AUC、ROC-AUC，以及两阈值各自的 Precision、Recall、F1、FPR、TP/FP/FN/TN；报告类别占比和排除的未知数。AP与梯形PR-AUC分列，种子均值±样本标准差，按数据集比较，不以Accuracy掩盖不平衡。未使用训练或校准标签；禁止标签参与训练、特征、阈值或选择超参数。

计划9×4×3=108次交易/时序运行、2×6×3=36次图运行，共144次（EB确定性三次重复单独注明）。每次保存配置/源码指纹、抽样索引、模型、分数、阈值、测试标签文件SHA、指标；模型重载检查在读取测试标签前完成。合成烟雾测试仅检查数值/依赖，不能计入金融实验成绩。CPU/GPU软件版本写入运行清单。

## 运行

```bash
pip install -e '.[zero2026,test]'
python scripts/prepare_repo_data.py --output /data/financial-zero/tabular --datasets ulb banksim s_ffsd sapienza_5S sapienza_15S sapienza_25S sd_stock sd_crypto sd_synth
python -m finad_zero.graph_data --output /data/financial-zero/graphs
python -m finad_zero.experiment --data /data/financial-zero/tabular --graph-data /data/financial-zero/graphs --output /data/financial-zero/results --cache /data/financial-zero/eb_cache --upstream /data/financial-zero/upstream --device cuda
```

`/data/...`为示例，可替换为可用工作目录；生成文件不写入Git原始数据目录。CQ执行脚本为 `scripts/run_zero_label_cq.sh`，只使用免费Kaggle配额，不租赁付费机器。以通过的同提交GPU smoke为完整实验前置门禁。

如果使用GitHub ZIP或CQ源码归档而没有 `.git`，须将 `FINAD_CODE_COMMIT` 环境变量设为源码对应的完整40位小写SHA；程序同时保存各科学源文件的SHA256。若存在Git目录，声明SHA必须与实际提交一致。正式实验源版本以结果 `run_manifest.json` 为准；报告导出与说明文档可有后续提交。

## 已完成结果与独立核验

[144次正式运行报告](../results/zero_label_20261114/README_CN.md)对应源提交 `847238664dc06077271e06430e5ef7ac8b2e9cfd`，CQ作业 `j-1114-2e8178`。整轮完成且通过独立核验后才查看金融成绩；未从不同执行版本拼接有利结果。报告保留288条阈值记录、逐种子配置、校准参数、运行环境及完整性证明。原始模型、保存分数及精确输入归档保留在本地和CQ归档，未将大型二进制检查点加入Git。

持有完整CQ输出时，可依次复核源码和输入、分数指标、配对样本与共同训练缩放器，再导出报告；第一步需要本仓库包含上述源提交的Git历史，归档解包核验建议使用Python 3.12或更新版本：

```bash
python scripts/verify_zero_label_artifacts.py --job-output /path/to/job-output
python scripts/verify_zero_label.py --results /path/to/job-output/results \
  --data /path/to/job-output/exact_inputs/tabular --graph-data /path/to/job-output/work/graphs
python scripts/verify_zero_label_cohort.py --runs /path/to/job-output/results \
  --data /path/to/job-output/exact_inputs/tabular --graph-data /path/to/job-output/work/graphs \
  --output /path/to/job-output/results/cohort_verification.json
python scripts/report_zero_label.py --runs /path/to/job-output/results \
  --output /path/to/report
```

报告目录还包含人工整理的执行记录、CQ配置和门禁证据；这些不由报告导出器伪造。独立核验共同缩放器不意味着EB-GAD额外的转导标准化也只使用训练图，见前述信息预算说明。
