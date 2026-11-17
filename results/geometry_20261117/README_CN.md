# HUGE/HIMVH 新方案与独立验证：进展报告（完整实验未完成）

**目前没有本轮完整AP、F1或对比成绩，不能声称方法改善或验证成功。** 代码、数据与资格检查已完成；完整实验被Kaggle提供方启动队列及CQ槽位限制阻塞。上一轮已核验结果仍在[上一轮中文报告](../roleweight_20261117/README_CN.md)。

## 已完成

- 生成新验证数据前固定方法、全部变体、模拟器参数、独立划分、基线集合及评价标准：[固定协议](../../docs/geometry_protocol_CN.md)。
- HUGE实现原40维HALO教师与学生输入分离；同时准备直接HALO、训练后评分、固定融合及表示消融。HIMVH实现相对训练常量预测器的无标签重构增益权重，含尾尺度、维数先验、权重下限及m/cm上下文消融。
- 原101个输入文件及划分不改。新增CardSim固定模拟世界：80107笔完整交易、26个封存输入，按完整时间戳累计计数60/20/20划分；数据生成前已固定方案。新输入归档[data/derived/geometry_20261117_cardsim.tar.gz](../../data/derived/geometry_20261117_cardsim.tar.gz)，[来源和全部SHA256](../../docs/geometry_cardsim_archive.json)。
- 114项测试通过；7项真实形状GPU／EB检查通过，测得峰值CUDA0.668GiB、RSS2.088GiB；12个旧HIM父模型的无标签真实校准权重检查通过。资格检查未读取真实标签或计算金融评价成绩。[CPU测试](cq_cpu_tests.txt) · [GPU检查](gpu_qualification.json) · [无标签权重检查](unlabeled_old_weights_qualification.json)。
- 独立核验、完整对比、配对消融、中文解释和科学绘图脚本已实现；这些脚本尚未对本轮完整结果执行，不能把脚本存在当作实验完成。

## 预登记的实验规模（不是已完成数量）

计划231组配置／462条操作点／154条三种子消融汇总。最终分析集计划54次新拟合：36次HUGE、6次CardSim-HIMVH、12次新数据基线；另复用12个旧HIM模型，3个EB种子重复训练免费。资格检查中的2步小训练不计入这些最终拟合数。

主方案固定HUGE-fusion、HIMVH-cm-skill-tail。种子42/43/44；旧8192训练目标、最多100000校准样本和完整测试不变。POT初始q=.85、risk=.01、min20，失败N/A；q99独立报告。保存AP、梯形PR-AUC、ROC、Precision、Recall、F1、FPR和混淆计数，均值±样本SD，不做点调整、标签阈值或翻转分数。

新CardSim会重新运行旧任务AP及两种F1获胜方法的并集：交易HBOS、LOF、VAE；账户EB-GAD发布版与Audit-HCL零标签适配。所有新数据候选使用同一新划分／测试样本，不借用旧数据成绩。全部HUGE和HIMVH评分变体见固定协议。

## 阻塞的实际状态

截至本报告更新时：

| CQ任务 | 用途 | 最近确认状态 |
|---|---|---|
| j-1117-12ffdf | 完整HUGE实验 | 提供方QUEUED、未确认RUNNING；已申请撤销，但尚未确认停止 |
| j-1117-8c114e | CardSim模型及最佳基线 | 提供方QUEUED、未确认RUNNING |
| j-1117-55bbe3 | 免费CPU替代的真形状资格检查 | 等待；CQ提示已有2本Kaggle笔记本占用槽位 |

CQ对撤销申请返回：`provider completion unconfirmed, notebook slot and GPU budget retained`。因此没有重复提交同一批完整实验，也没有启用付费租赁。需要CQ管理员核查并确认本项目未启动笔记本的停止／槽位释放；不能仅凭客户端发出撤销就认定提供方已经停止。已有资格检查、原始数据、父模型及其他项目作业保留。

管理员确认释放后，先检查队列和输出，避免重复拟合；完成CPU资格检查或已有GPU任务后再执行剩余实验。完整结果通过独立核验后，本进展报告将被最终中文结果报告替换；此处不填写预计或推测的实验成绩。

## 独立验证的边界

CardSim是新生成的合成来源，其数据此前没有参与本项目实验；这使后续结果可作为固定方案的独立模拟来源验证。它不是新的真实银行外部数据，也不是将旧银行模型零样本迁移。各方法将在新世界自己的训练／校准分区拟合，再评价测试分区。欺诈标签由模拟器规则生成；一个世界、一个生成种子不能替代多个独立真实来源。

原BankSim和S-FFSD测试已被用于设计本轮方法，原任务仍是回顾性开发。权重的重构可预测性不等于欺诈相关性，直接HALO的收益也不能自动归功于学生训练；最终报告必须用相应消融检验这两点。

## 文件与后续执行

[阶段007任务说明](../../paper/workflow/prompts/007-geometry-skill-independent.md) · [阶段008核验与报告说明](../../paper/workflow/prompts/008-geometry-verification-report.md) · [当前状态](../../paper/workflow/current.md) · [执行状态与规格](execution.json)。

数值源固定为a683fcb152b8db48d5d6398f45e44331f605ac91，数据生成前协议源为2f5d44176bb5252f5d266a7d726cf79b547a95f4。后续报告／状态提交不改变冻结数值源；独立核验会逐文件检查这些核心定义一致。

## Plain English

The method code, frozen protocol, unseen synthetic CardSim data and qualification checks are ready. The full experiments have not been confirmed running: Kaggle is queueing the notebooks and CQ retains their slots while cancellation is unconfirmed. There are no new complete benchmark results and no improvement claim. Release/verify the provider slots first, then finish the fixed experiments, independent replay and final report. Synthetic-source validation is distinct from real-bank external validation.
