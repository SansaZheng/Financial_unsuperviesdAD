# KronosAD同协议全量重跑：预先固定协议（2026-11-15）

目标是将KronosAnomalyDetection中的冻结Kronos-mini逐特征数值代理预测误差方案，迁移到本仓库**完全相同的9任务输入缓存、划分、特征、校准抽样、全量测试位置和评价代码**。主方法为`kronos_potratio_matched`，不是原生OHLCVA，也不沿用此前小样本Proxy结果。

**方法保真修订（在查看任何本轮金融检测指标前登记）**：首版`782ff9d`冻结了全量预测、共同阈值及最大误差标量。代码核对发现最大误差省略了源方法的逐维POT相对标尺，因此改以源方法的`max(error_i / D_i)`聚合作为主方法，首版最大误差明确保留为消融。GPU预测配置、输入及全部预测均不变，没有重新推理，也不按测试成绩选择两者较好值。正式推理源提交与后续无标签评分源提交分别记录；这是方法定义核对造成的公开修订，不声称两版最初完全相同。

## 固定输入与零标签规则

- 任务：ULB、BankSim、S-FFSD、Sapienza 5S/15S/25S、SD-FMM stock/crypto/synth。逐文件与`results/zero_label_20261114/input_cache_files_sha256.json`核验；输入不一致立即失败。
- 种子42/43/44。按基线相同规则抽样最多8192条训练行，拟合相同`Preprocessor`（median/IQR、std回退、[-8,8]裁剪），没有更换特征或金额变换。冻结模型不做有监督或自监督更新，8192行用于共享预处理，不声称与400更新AE具有相同预训练预算。
- 校准按`seed+100`相同规则最多100000行；只这些位置的分数拟合阈值。测试包含原完整分区每一行，S-FFSD未知标签仅在最终评价屏蔽。训练／校准标签不被解释或用于筛选；哈希校验只按字节比较。
- 不搜索标签阈值，不按测试指标选方法版本或超参数。模型／分数／阈值文件在打开测试标签前写完。

## 固定模型及评分

- 实际使用冻结`NeoQuasar/Kronos-mini`和`Kronos-Tokenizer-2k`。官方代码固定`67b630e67f6a18c9e9be918d9b4337c960db1e9a`；3份源码、2份模型配置和2份权重均逐一校验SHA256，与已发布源实验清单一致。运行时下载到外部工作区，不将权重加入Git。
- 每个共享工程特征独立映射到OHLC四槽，volume/amount为零占位，不参与评分。回看严格此前32个观测，官方历史输入截断[-5,5]，sample_count=1、T=1、top_p=.9、top_k=0；推理float32，生产batch=1024。各数据／种子／阶段／批次通过固定SHA256键设置随机流，GPU调度与可恢复执行不改变同一批次随机种子。
- 代理窗口按其均值和总体标准差归一化，分母`std+1e-5`；模型close输出恢复到共享特征单位并存为float32，实际评分直接使用已保存的相同预测。
- 每维平方误差为`e_i=((z_ti-pred_ti)/(std_i(past_32)+1e-6))**2`，epsilon在共享缩放后的单位中固定。全部输入维度都参与真实预测。
- **主方法保留源聚合**：只用配对校准位置的逐维误差拟合各自POT标尺`D_i`，沿用源方法内部q=.90、risk=.01、至少30个超额；标量为成功拟合维度的`max_i(e_i/D_i)`。恒定／失败维度及原因逐一公开，与源方法一样仅在聚合时排除，绝不按标签选特征；全部维度失败则明确失败，不虚构回退分数。
- **共同操作点**：对上述标量使用与基线相同的`finad.thresholds.calibrate`，POT初始q=.85、risk=.01、至少20个非退化超额；另报99%校准分位数。两者均用`>=`报警。源方法内部逐维标尺属于方法评分，与共同最终阈值分开记录；Kronos额外使用校准特征拟合标尺的信息范围公开。原逐维OR对应的标量阈值1只作补充，不参加共同阈值排名。
- **保留消融**：首版最大误差`max_i(e_i)`及其相同共同操作点全部保留，不用测试分数选择主方法。不得把主方法共同操作点冒充源默认OR报警率，也不将消融较好值替换主方法结果。两版POT失败均明确N/A。

## 历史边界与完整测试覆盖

- ULB、BankSim、S-FFSD按既有缓存行顺序形成流，校准可使用此前训练观测，测试可使用此前校准／已发生测试观测；没有当前或未来行进入自己的预测。历史提供额外过去特征信息，这是预测方法的固有信息预算，与逐行表格方法并非所有信息完全相同。
- Sapienza和SD-FMM按缓存的`*.seq.i32`在每个事件／案例重置，不跨序列、分区或币种拼接。原缓存顺序保留，BankSim内部顺序不等于已知真实交易时钟，SD-FMM已处理序列不等于可部署实时行情。
- **冷启动**：不足32个历史位置时左填训练中位数（共享中心化空间中为零），不填当前值或未来值。所有测试位置仍评分，不删去序列前32点；各阶段冷启动和完全无历史数量单独记录。与只评价热身后位置的旧Proxy实验不同。
- 模型日历仍为2000-01-01起的人工分钟序号；流使用全流位置，分组序列重置位置。没有声称实际交易频率为一分钟。原生时间嵌入、冷启动和域外表示的限制需要随结果报告。

## 评价、证据与范围

报告AP、梯形PR-AUC、ROC-AUC、两套操作点各自P／R／F1／FPR和混淆矩阵、未知数、异常占比；三个种子均值与样本标准差。不做point adjustment、best-F1标签选阈值或混合跨数据集总排名。保留每特征预测、标量分数、配对索引、预处理器、阈值、源码／权重／输入哈希。独立脚本重构每个过去窗口并从全部保存预测复算标量分数，再重算全部指标。

推理阶段先通过本地测试、真实权重CPU检查、同一推理提交的GPU无标签smoke及吞吐检查，再提交正式作业。恢复源逐维POT聚合属于后续CPU评分阶段，有独立测试与单独源码／环境记录，不冒充该评分修订也执行过额外GPU推理。仅使用免费CQ Kaggle资源，禁止付费租赁或占用他人作业；不重启已结束历史任务。生产多GPU只改变调度，不能选择成绩较好的设备运行。

**账户图轨道不新增虚构Kronos图方法。** 本轮Kronos重跑属于9任务交易／点级轨道；原账户图六方法结果保持不变。冻结预训练权重可能有数据重叠，不能声称各方法训练语料和计算预算完全一致，也不能据此把同评价样本的公平性扩展为全部信息预算一致。

```bash
pip install -e '.[kronos,test]'
python scripts/prepare_repo_data.py --output /data/financial-kronos/tabular \
  --datasets ulb banksim s_ffsd sapienza_5S sapienza_15S sapienza_25S sd_stock sd_crypto sd_synth
python -m finad_kronos.experiment --data /data/financial-kronos/tabular \
  --output /data/financial-kronos/results --assets /data/financial-kronos/assets --device cuda --workers 2
python scripts/verify_kronos_matched.py --results /data/financial-kronos/results \
  --data /data/financial-kronos/tabular
python -m finad_kronos.rescore --raw-runs /data/financial-kronos/results \
  --data /data/financial-kronos/tabular --output /data/financial-kronos/potratio
python scripts/verify_kronos_matched.py --results /data/financial-kronos/potratio \
  --data /data/financial-kronos/tabular
```

`/data`是示例授权工作目录。没有`.git`的CQ源码归档须设置`FINAD_CODE_COMMIT`为匹配的完整40位提交；实验记录同时保存实际文件SHA256。报告导出可在后续提交完成，实际拟合／评分源版本由运行清单确定。
