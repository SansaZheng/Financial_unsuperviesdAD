## 12. HUGE、HIMVH-Zero、SD-FMM-Zero 同划分零标签扩展（2026-11-17）

全部30次正式运行和60条操作点指标已完成并独立核验。HUGE在2组账户图运行；HIMVH-Zero在2组交易数据运行；SD-FMM-Zero在6组行情时序运行，各3种子。原81个交易缓存及20个正式图输入、抽样／测试对象／共同缩放器均核对，未知标签只在评价排除。

**HIMVH与SD-FMM是显式零标签适配，不能称原论文监督结果复现。** HUGE为跨阶段归纳端口；上下文信息、预训练和计算量不同，不声称完全等信息预算。本轮仍用共同原POT与独立q99，POT失败N/A。前述第2节保留11月16日15方法历史表，最新排名以本节链接为准。

[最新完整中文报告、交易与账户图对比、各数据集最佳方法及原因分析](../results/recent_zero_20261117/README_CN.md)。

| 轨道 | 数据集 | 新方法 | AP | 旧最佳AP | 差值（百分点） |
|---|---|---|---:|---:|---:|
| graph | BankSim | HUGE金融归纳端口 | 39.48 | EB-GAD发布版零标签 61.25 | -21.77 |
| graph | S-FFSD | HUGE金融归纳端口 | 84.13 | Audit-HCL零标签适配 90.38 | -6.25 |
| tabular | BankSim | HIMVH-Zero | 9.90 | HBOS 40.92 | -31.02 |
| tabular | S-FFSD | HIMVH-Zero | 36.03 | HBOS 56.11 | -20.08 |
| tabular | Sapienza 15S | SD-FMM-Zero | 0.10 | kNN 82.56 | -82.46 |
| tabular | Sapienza 25S | SD-FMM-Zero | 0.14 | kNN 84.01 | -83.88 |
| tabular | Sapienza 5S | SD-FMM-Zero | 0.18 | kNN 76.16 | -75.97 |
| tabular | SD-FMM加密货币 | SD-FMM-Zero | 0.44 | 持续值残差对照（N+C） 33.65 | -33.21 |
| tabular | SD-FMM股票 | SD-FMM-Zero | 6.68 | Isolation Forest 63.85 | -57.17 |
| tabular | SD-FMM合成 | SD-FMM-Zero | 2.85 | RGLD-ID 77.14 | -74.29 |
