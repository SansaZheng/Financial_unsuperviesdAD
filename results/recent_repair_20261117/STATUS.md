# 当前状态：等待CQ控制机释放磁盘

本轮尚未完成，**这里没有最终检测成绩或排名**。不能把门禁通过／文件完整性通过当作模型效果改善。

- 已完成训练特征与合成反例审计，并固定HUGE-role、HIMVH-cm、SD-FMM-ge及全部消融：最终174配置、156次拟合、348操作点。
- 科学提交`09de7f313d57470293f1309ed611914a88cbe62c`；92项本地与CQ CPU测试、真实形状GPU门禁通过。
- 已接收HUGE24配置、SD-FMM种子43/44各36配置，共96配置；源代码、模型／分数／阈值摘要及原AE抽样索引核对通过。**还未查看新金融指标，也未完成最终指标重算。**
- SD-FMM种子42仍在运行；HIMVH三个种子分片因控制机mac-mini的`No space left on device`无法启动。已请求管理员保留旧实验并安全归档，释放至少5GB。队列保留，不重复提交。
- 大包归档超过500MiB的批次不进入最终分析；按种子分片只改变执行方式，不改变模型、数据、训练步数或阈值。全部失败／替代记录保留。

待空间恢复后，完成剩余配置、无损合并分片，独立重算AP／ROC／PR、POT／q99及混淆矩阵，再生成中文报告、对比图和主榜并推送main与默认分支。

已有证据：[训练审计](train_only_audit.json)、[CPU测试](cq_cpu_tests.txt)、[GPU门禁](gpu_qualification.json)、[部分完整性接收记录](partial_integrity.json)、[执行记录](execution.json)、[固定协议与执行说明](../../docs/recent_repair_protocol_CN.md)。
