# Experiment report / 实验说明 — revision2

**中文：** 原划分下完成174配置、156次拟合和348条操作点，全部独立复核通过。SD-FMM主修正版在5/6任务提高AP，HUGE1/2、HIMVH0/2，三个主版本均未取得候选池最高AP或任一F1。HIM固定检查点的评分权重消融与后验字段分析显示，等字段权重放大了弱信号；完整上下文效应依数据而变。SD末点损失有帮助，但保持约束的生成器在Sapienza15S/25S不如仅改末点目标。HUGE角色重参数化对BankSim有害，1200更新未补回。不能把机制修正直接等同于总体性能提升。全部负结果、原版复跑波动、信息预算和回顾性限制保留。中文报告见results/recent_repair_20261117/README_CN.md；发布状态见current.md。

**English:** All174 configurations and348 operating records pass independent source/input/cohort/checkpoint/threshold/metric checks. Prespecified primary AP improves on5/6 SD tasks,1/2 HUGE tasks and0/2 HIM tasks; none tops the eligible pool by meanAP or eitherF1. Fixed-checkpoint ablations identify harmful equal-field scoring; native context and coherent augmentation effects depend on the dataset. Correcting counterexamples is not a performance guarantee. No posthoc winning-variant substitution, supervised-paper or globalSOTA claim is made. Field-level explanations are explicitly posthoc; all prior outcomes and failed technical batches remain traceable.

## Revision3 / roleweight-20261117
[Chinese full report](../../results/roleweight_20261117/README_CN.md). HUGE dual worsens Bank/SFFSD AP by0.13/3.30pp vs fresh role anchors;HIM cm_tail improves7.23/2.00pp vs frozen cm_field. Fixed primaries retained, negative outcomes complete. No true labels used for training, weights or thresholds. Statistical rarity still differs from fraud relevance; role channel information retention does not preserve HALO geometry. Needs independent future holdout before broader claims.
