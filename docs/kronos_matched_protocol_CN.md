# KronosAD同协议全量重跑：预先固定协议（2026-11-15）

目标是将KronosAnomalyDetection中的冻结Kronos-mini逐特征数值代理预测误差方案，迁移到本仓库**完全相同的9任务输入缓存、划分、特征、校准抽样、全量测试位置和评价代码**。这是`kronos_proxy_matched`，不是原生OHLCVA，也不沿用此前小样本Proxy结果。

## 固定输入与零标签规则

- 任务：ULB、BankSim、S-FFSD、Sapienza 5S/15S/25S、SD-FMM stock/crypto/synth。逐文件与`results/zero_label_20261114/input_cache_files_sha256.json`核验；输入不一致立即失败。
- 种子42/43/44。按基线相同规则抽样最多8192条训练行，拟合相同`Preprocessor`（median/IQR、std回退、[-8,8]裁剪），没有更换特征或金额变换。冻结模型不做有监督或自监督更新，8192行用于共享预处理，不声称与400更新AE具有相同预训练预算。
- 校准按`seed+100`相同规则最多100000行；只这些位置的分数拟合阈值。测试包含原完整分区每一行，S-FFSD未知标签仅在最终评价屏蔽。训练／校准标签不被解释或用于筛选；哈希校验只按字节比较。
- 不搜索标签阈值，不按测试指标选方法版本或超参数。模型／分数／阈值文件在打开测试标签前写完。

## 固定模型及评分

- 实际使用冻结`NeoQuasar/Kronos-mini`和`Kronos-Tokenizer-2k`。官方代码固定`67b630e67f6a18c9e9be918d9b4337c960db1e9a`；3份源码、2份模型配置和2份权重均逐一校验SHA256，与已发布源实验清单一致。运行时下载到外部工作区，不将权重加入Git。
- 每个共享工程特征独立映射到OHLC四槽，volume/amount为零占位，不参与评分。回看严格此前32个观测，官方历史输入截断[-5,5]，sample_count=1、T=1、top_p=.9、top_k=0；推理float32，生产batch=1024。各数据／种子／阶段／批次通过固定SHA256键设置随机流，GPU调度与可恢复执行不改变同一批次随机种子。
- 代理窗口按其均值和总体标准差归一化，分母`std+1e-5`；模型close输出恢复到共享特征单位并存为float32，实际评分直接使用已保存的相同预测。
- **标量分数**为所有特征标准化平方预测误差的最大值：`max_i ((z_ti - pred_ti)/(std_i(past_32)+1e-6))**2`。不剔除任意特征，也不按标签选择特征。epsilon在共享缩放后的单位中固定。
- 为遵守统一阈值标准，原源仓库“逐维POT，再取score/threshold最大值”的组合改为上述固定最大误差，再用与基线相同的`finad.thresholds.calibrate`：POT初始q=.85、risk=.01、至少20个非退化超额；另报99%校准分位数。两者均用`>=`报警，POT失败明确N/A，不替换为分位数或原逐维阈值。此评分／阈值适配在查看本轮金融指标前固定，不能称源仓库默认评分的逐项原样重跑。

## 历史边界与完整测试覆盖

- ULB、BankSim、S-FFSD按既有缓存行顺序形成流，校准可使用此前训练观测，测试可使用此前校准／已发生测试观测；没有当前或未来行进入自己的预测。历史提供额外过去特征信息，这是预测方法的固有信息预算，与逐行表格方法并非所有信息完全相同。
- Sapienza和SD-FMM按缓存的`*.seq.i32`在每个事件／案例重置，不跨序列、分区或币种拼接。原缓存顺序保留，BankSim内部顺序不等于已知真实交易时钟，SD-FMM已处理序列不等于可部署实时行情。
- **冷启动**：不足32个历史位置时左填训练中位数（共享中心化空间中为零），不填当前值或未来值。所有测试位置仍评分，不删去序列前32点；各阶段冷启动和完全无历史数量单独记录。与只评价热身后位置的旧Proxy实验不同。
- 模型日历仍为2000-01-01起的人工分钟序号；流使用全流位置，分组序列重置位置。没有声称实际交易频率为一分钟。原生时间嵌入、冷启动和域外表示的限制需要随结果报告。

## 评价、证据与范围

报告AP、梯形PR-AUC、ROC-AUC、两套操作点各自P／R／F1／FPR和混淆矩阵、未知数、异常占比；三个种子均值与样本标准差。不做point adjustment、best-F1标签选阈值或混合跨数据集总排名。保留每特征预测、标量分数、配对索引、预处理器、阈值、源码／权重／输入哈希。独立脚本重构每个过去窗口并从全部保存预测复算标量分数，再重算全部指标。

先通过本地测试、真实权重CPU检查、同提交GPU无标签smoke及吞吐检查，再提交正式作业。仅使用免费CQ Kaggle资源，禁止付费租赁或占用他人作业；不重启已结束历史任务。生产多GPU只改变调度，不能选择成绩较好的设备运行。

**账户图轨道不新增虚构Kronos图方法。** 本轮Kronos重跑属于9任务交易／点级轨道；原账户图六方法结果保持不变。冻结预训练权重可能有数据重叠，不能声称各方法训练语料和计算预算完全一致，也不能据此把同评价样本的公平性扩展为全部信息预算一致。

```bash
pip install -e '.[kronos,test]'
python scripts/prepare_repo_data.py --output /data/financial-kronos/tabular \
  --datasets ulb banksim s_ffsd sapienza_5S sapienza_15S sapienza_25S sd_stock sd_crypto sd_synth
python -m finad_kronos.experiment --data /data/financial-kronos/tabular \
  --output /data/financial-kronos/results --assets /data/financial-kronos/assets --device cuda --workers 2
python scripts/verify_kronos_matched.py --results /data/financial-kronos/results \
  --data /data/financial-kronos/tabular
```

`/data`是示例授权工作目录。没有`.git`的CQ源码归档须设置`FINAD_CODE_COMMIT`为匹配的完整40位提交；实验记录同时保存实际文件SHA256。报告导出可在后续提交完成，实际拟合／评分源版本由运行清单确定。
