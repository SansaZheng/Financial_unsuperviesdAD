"""Export all fixed variants and a primary-only fair comparison."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from report_recent_zero import readcsv,savecsv,METRICS,DATASETS,fmt,ms,winner,NAMES
from finad_roleweight.adapters import VARIANTS as R
from finad_roleweight.fields import VARIANTS as F
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--update-main',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 proof=json.loads((a.runs/'verification.json').read_text());assert proof['status']=='passed' and proof['runs']==144
 records=json.loads((a.runs/'records.json').read_text());manifests=json.loads((a.runs/'run_manifests.json').read_text())
 for old,new in [('records.json','per_seed.json'),('verification.json','verification.json'),('run_manifests.json','run_manifests.json')]:shutil.copyfile(a.runs/old,a.output/new)
 rows=[]
 for r in records:
  for op,v in r['metrics']['operating_points'].items():
   row={k:r[k] for k in ['track','dataset','family','variant','method','seed']};row.update(operating_point=op,status='ok' if v else 'failed',seconds=r['seconds'],peak_cuda_gb=r.get('peak_cuda_gb'),new_fit=not r['model_info']['fit_reused'])
   row.update({k:value for k,value in r['metrics'].items() if k!='operating_points'})
   row.update({k:v[k] if v else None for k in ['precision','recall','f1','fpr','tp','fp','fn','tn','alarm_fraction']});rows.append(row)
 assert len(rows)==288;savecsv(a.output/'metrics.csv',rows);summary=[]
 for f,d,v,op in sorted({(r['family'],r['dataset'],r['variant'],r['operating_point']) for r in rows}):
  g=[r for r in rows if (r['family'],r['dataset'],r['variant'],r['operating_point'])==(f,d,v,op)];assert len(g)==3
  row={k:g[0][k] for k in ['track','dataset','family','variant','method','operating_point']};row.update(seeds=3,threshold_successes=sum(x['status']=='ok' for x in g),source_commit=manifests[f]['git_commit'],source_path='finad_roleweight/experiment.py',source_kind='fresh_fit' if f=='huge' else 'frozen_parent_fresh_scoring')
  for m in METRICS:
   vals=[r[m] for r in g if r[m] is not None];row[m+'_mean']=float(np.mean(vals)) if len(vals)==3 else None;row[m+'_std']=float(np.std(vals,ddof=1)) if len(vals)==3 else None
  for k in ['evaluated','unknown_excluded','positives','normal','prevalence']:assert len({r[k] for r in g})==1;row[k]=g[0][k]
  summary.append(row)
 assert len(summary)==96;savecsv(a.output/'summary.csv',summary)
 previous=readcsv(ROOT/'results/recent_repair_20261117/combined_transaction_summary.csv')+readcsv(ROOT/'results/recent_repair_20261117/combined_account_graph_summary.csv')
 combined=[r for r in previous if r['method'] not in ['huge_repair_legacy','huge_repair_role']]+[r for r in summary if (r['family']=='huge' and r['variant'] in ['legacy','role','dual']) or (r['family']=='himvh' and r['variant']=='cm_tail')]
 assert len(combined)==378
 for r in summary:
  old=next(x for x in previous if (x['track'],x['dataset'])==(r['track'],r['dataset']))
  for k in ['evaluated','unknown_excluded','positives','normal','prevalence']:assert r[k]==old[k]
 savecsv(a.output/'combined_transaction_summary.csv',[r for r in combined if r['track']=='tabular']);savecsv(a.output/'combined_account_graph_summary.csv',[r for r in combined if r['track']=='graph'])
 names=NAMES.copy()
 for fam,variants in [('huge',['legacy','role','role_long','role_noself']),('himvh',['legacy','f00','c','m','cm','m_flat','cm_flat']),('sdfmm',['legacy','f00','e','g','ge','ge_nocontrast'])]:
  for v in variants:names[f'{fam}_repair_{v}']=fam.upper()+'上一轮-'+v
 for v in R:names['huge_roleweight_'+v]='HUGE-'+v+('（本轮主方案）' if v=='dual' else '（本轮锚点）' if v in ['legacy','role'] else '')
 for b in ['m','cm']:
  for v in F:names['himvh_roleweight_'+b+'_'+v]='HIMVH-'+b+'-'+v+('（本轮主方案）' if (b,v)==('cm','tail') else '')
 ix={(r['family'],r['dataset'],r['variant'],r['operating_point']):r for r in summary};ri={(r['family'],r['dataset'],r['variant'],r['seed'],r['operating_point']):r for r in rows};effects=[]
 hugepairs=[('去角色标识','numeric','legacy'),('对齐及压缩','aligned','numeric'),('分角色中心','centered','aligned'),('分角色尺度','role','centered'),('恢复原数值通道','dual','aligned'),('主方案对旧legacy','dual','legacy'),('主方案对上一轮role','dual','role'),('双通道下分角色中心','dual_center','dual'),('双通道下分角色尺度','dual_role','dual_center')]
 for d in ['banksim','s_ffsd']:
  pairs=[('huge',*x) for x in hugepairs]+[('himvh',b+'：'+v+'－field',b+'_'+v,b+'_field') for b in ['m','cm'] for v in F if v!='field']+[('himvh','tail下完整－抽样上下文','cm_tail','m_tail')]
  for f,label,new,old in pairs:
   for op in ['pot','quantile']:
    e=dict(family=f,dataset=d,comparison=label,new=new,old=old,operating_point=op,seeds=3)
    for m in METRICS:
     vals=[ri[f,d,new,s,op][m]-ri[f,d,old,s,op][m] for s in [42,43,44] if ri[f,d,new,s,op][m] is not None and ri[f,d,old,s,op][m] is not None]
     e[m+'_delta_mean']=float(np.mean(vals)) if len(vals)==3 else None;e[m+'_delta_std']=float(np.std(vals,ddof=1)) if len(vals)==3 else None
    effects.append(e)
 savecsv(a.output/'paired_effects.csv',effects)
 best=[]
 for track,tasks in [('tabular',DATASETS),('graph',['banksim','s_ffsd'])]:
  for d in tasks:
   for metric,op in [('average_precision','pot'),('roc_auc','pot'),('f1','pot'),('f1','quantile')]:
    for r in winner(combined,track,d,metric,op):best.append(dict(track=track,dataset=d,metric=metric,operating_point=op,method=r['method'],mean=r[metric+'_mean'],std=r[metric+'_std']))
 savecsv(a.output/'best_methods.csv',best)
 drift=[]
 for r in summary:
  if r['family']=='huge' and r['variant'] in ['legacy','role']:
   old=next(x for x in previous if (x['track'],x['dataset'],x['method'],x['operating_point'])==('graph',r['dataset'],'huge_repair_'+r['variant'],r['operating_point']))
   drift.append(dict(dataset=r['dataset'],variant=r['variant'],operating_point=r['operating_point'],AP_difference=r['average_precision_mean']-old['average_precision_mean'],F1_difference=None if r['f1_mean'] is None or old['f1_mean'] is None else r['f1_mean']-old['f1_mean']))
 savecsv(a.output/'anchor_reproduction.csv',drift)
 findings=[]
 text=['# HUGE 角色适配与 HIMVH 字段评分改进：中文实验报告','','本轮48次HUGE新训练、96组HIMVH冻结骨干重评分，共144配置／288条操作点／96条三种子消融汇总。主方案在新成绩产生前固定为 **HUGE-dual** 和 **HIMVH-cm-tail**。训练与字段权重不使用真实标签，POT和q99也不使用标签。沿用相同输入、划分、测试样本和评价标准。**使用旧测试集诊断后继续开发，属于回顾性评估，不是独立盲测。**','','## 主结论','','百分数，均值±样本标准差；AP差值单位为百分点。AP评价排序，F1/FPR评价既定阈值；不能以高AP替代可用的报警效果。','','| 轨道／任务 | 方法 | 上轮对应主版本AP | 新主方案AP | AP差值 | 上轮POT F1 | 新POT F1 | 新q99 F1 |','|---|---|---:|---:|---:|---:|---:|---:|']
 for f,primary,oldv in [('huge','dual','role'),('himvh','cm_tail','cm_field')]:
  count=0
  for d in ['banksim','s_ffsd']:
   old=ix[f,d,oldv,'pot'];new=ix[f,d,primary,'pot'];q=ix[f,d,primary,'quantile'];diff=new['average_precision_mean']-old['average_precision_mean'];count+=diff>0
   text.append('| '+new['track']+'／'+DATASETS[d]+' | '+f.upper()+' | '+' | '.join([ms(old,'average_precision'),ms(new,'average_precision'),f'{100*diff:+.2f}',ms(old,'f1'),ms(new,'f1'),ms(q,'f1')])+' |')
  find=dict(family=f,AP_improved_vs_previous_primary=int(count),tasks=2)
  for label,metric,op in [('AP_first','average_precision','pot'),('POT_F1_first','f1','pot'),('q99_F1_first','f1','quantile')]:find[label]=sum(any(x['method']==f+'_roleweight_'+primary for x in winner(combined,'graph' if f=='huge' else 'tabular',d,metric,op)) for d in ['banksim','s_ffsd'])
  findings.append(find)
  pass # Findings are displayed after the whole table, preserving Markdown structure.
 for find in findings:
  text+=['',f'**{find["family"].upper()}：** AP较上轮对应主版本改善{find["AP_improved_vs_previous_primary"]}/2任务；共同方法池中，AP均值最高{find["AP_first"]}/2，POT F1最高{find["POT_F1_first"]}/2，q99 F1最高{find["q99_F1_first"]}/2。仅为均值排名，不代表显著性。']
 text+=['','上表HUGE使用本轮重训role锚点代表上一轮主方案，以控制环境差异；HIMVH使用与上一轮逐项一致的cm_field冻结模型对照。旧法legacy和所有原始锚点差异见下表与anchor_reproduction.csv。']
 (a.output/'findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
 text+=['','![主方案与对照](comparison.png)','','## 改了什么，为什么这样设计','','**HUGE角色通道。** 上轮同时改变字段位置、移除角色标识、分角色中心和分角色尺度，BankSim严重退步，无法直接把结果归因于某一个因素。本轮逐步拆开：原40维→去角色标识32维→角色对应方向对齐16维→仅分角色中心→再分角色尺度。主方案dual将原32维与共享尺度对齐16维合并，分别除以维数平方根，保留付款／收款方向及绝对行为信息，同时提供对应角色行为的可比通道。训练统计只从训练图拟合。双通道改变输入维数和参数量，所以该效应不能单独解释为角色对齐的因果收益。网络、损失、300更新和边均不变。','','**HIMVH字段误差。** 等权平均原始重构误差，并不代表各字段贡献相同：一个数值字段可能比许多类别字段误差大得多；按维数又会放大宽哈希块。新主评分将每个误差换成“在无标签校准分布中有多罕见”的证据。若校准有n条，p=(1+误差≥当前值的条数)/(n+1)，证据为-log(p)，然后字段等权平均。并列保守处理，常量校准字段证据为0。校准和测试应用同一固定映射，不用测试分布定权。字段异常罕见度仍不等于欺诈相关性。','','另外比较均值／中位数归一化、按维数聚合、最大字段、校准正相关冗余逆权重；完整上下文cm与抽样上下文m都保留。HIMVH本轮没有新增模型训练：复用12个已核验的冻结模型及逐字段误差，重新拟合评分校准器和阈值，隔离字段评分的作用。field/flat对照与上一轮分数逐项一致。不能把它描述成重新训练了96个网络。','','[新一轮固定协议](../../docs/roleweight_protocol_CN.md) · [旧实现审计和训练消融](../recent_repair_20261117/README_CN.md)。','','## 全部消融，含退步结果']
 for f,variants in [('huge',R),('himvh',[b+'_'+v for b in ['m','cm'] for v in F])]:
  for d in ['banksim','s_ffsd']:
   text+=['','### '+f.upper()+'／'+DATASETS[d],'','| 配置 | AP ↑ | ROC-AUC ↑ | POT F1 ↑ | q99 F1 ↑ | POT FPR ↓ | q99 FPR ↓ |','|---|---:|---:|---:|---:|---:|---:|']
   for v in variants:
    r=ix[f,d,v,'pot'];q=ix[f,d,v,'quantile'];label='**'+v+'（主）**' if (f,v) in [('huge','dual'),('himvh','cm_tail')] else v
    text.append('| '+label+' | '+' | '.join([ms(r,'average_precision'),ms(r,'roc_auc'),ms(r,'f1'),ms(q,'f1'),ms(r,'fpr'),ms(q,'fpr')])+' |')
 text+=['','## 配对效应及原因分析','','每个种子先做差，再计算均值±样本标准差。变化单位为百分点；AP/F1为正表示提高，FPR为正表示误报增加。3种子不足以直接宣称显著性；下列对照只说明本设置中的效果。','','| 方法／任务 | 改变 | AP差值 | POT F1差值 | POT FPR差值 |','|---|---|---:|---:|---:|']
 for e in effects:
  if e['operating_point']=='pot':text.append('| '+e['family'].upper()+'／'+DATASETS[e['dataset']]+' | '+e['comparison']+' | '+' | '.join('N/A' if e[m+'_delta_mean'] is None else f'{100*e[m+"_delta_mean"]:+.2f} ± {100*e[m+"_delta_std"]:.2f}' for m in ['average_precision','f1','fpr'])+' |')
 text+=['','解读时先看HUGE的“去角色标识”“分角色中心”“分角色尺度”三个对照。若单独去标识改善而分角色缩放退步，说明把角色标识与行为统计混合、或把异常绝对幅度标准化掉，可能是适配损失；不能把所有变化都归因于角色异配性。dual与aligned的差异衡量恢复原数值通道的整体作用，含维数变化。','','HIMVH的tail对field衡量的是统一字段误差尺度的作用，tail_flat和tail_corr再衡量聚合规则。若尾概率提升但最大值或相关权重退步，说明“最罕见字段”或“最不冗余字段”并非总是最有欺诈信息的字段；无标签校准只能衡量统计罕见度，无法保证分配到欺诈相关性最强的权重。上述结果不用于逐数据集挑选主方法。完整上下文不一定更好：它会改变可重构性与邻域平滑，cm_tail与m_tail的差异保留在表中。','','当前仍无法由这些实验单独证实：哈希编码损失、真实欺诈与重构目标不匹配、无标签训练污染，以及时间／图聚合丢失异常证据。它们是待检验假设。AP低不能只归因于POT；尾证据有界且校准变换和POT共用样本，测试分布漂移后仍可能报警过多、过少或出现POT拟合失败。未重新按标签调风险、翻转分数或改用有利阈值。','','## 与现有方法的完整对比','','主榜只加入预先固定dual和cm_tail；其他新消融不进入最佳方法挑选。旧HUGE legacy/role由本轮重训锚点替换，其余已核验经典方法、KronosAD、CAE-IF、EB-GAD、Audit-HCL、RGLD、RTTAD及SD-FMM成绩复用，未声称本轮全部重训。共378条主榜汇总，交易和账户分别评价。']
 for track,tasks in [('tabular',DATASETS),('graph',['banksim','s_ffsd'])]:
  text+=['','### '+('交易／行情点轨道' if track=='tabular' else '付款账户图轨道')]
  for d in tasks:
   text+=['','#### '+DATASETS[d],'','| 方法 | AP ↑ | ROC-AUC ↑ | POT F1 ↑ | q99 F1 ↑ | POT FPR ↓ | q99 FPR ↓ |','|---|---:|---:|---:|---:|---:|---:|']
   for r in [r for r in combined if (r['track'],r['dataset'],r['operating_point'])==(track,d,'pot')]:
    q=next(x for x in combined if (x['track'],x['dataset'],x['method'],x['operating_point'])==(track,d,r['method'],'quantile'))
    text.append('| '+names[r['method']]+' | '+' | '.join([ms(r,'average_precision'),ms(r,'roc_auc'),ms(r,'f1'),ms(q,'f1'),ms(r,'fpr'),ms(q,'fpr')])+' |')
 text+=['','## 各数据集最佳方法','','| 轨道 | 任务 | AP最高 | POT F1最高 | q99 F1最高 |','|---|---|---|---|---|']
 for track,tasks in [('tabular',DATASETS),('graph',['banksim','s_ffsd'])]:
  for d in tasks:
   cells=[]
   for metric,op in [('average_precision','pot'),('f1','pot'),('f1','quantile')]:
    wins=winner(combined,track,d,metric,op);cells.append('／'.join(names[r['method']] for r in wins)+' '+fmt(wins[0][metric+'_mean']))
   text.append('| '+track+' | '+DATASETS[d]+' | '+' | '.join(cells)+' |')
 text+=['','## 公平条件、局限与复核','','101个输入文件SHA256一致；种子42/43/44；原训练8192目标、校准最多100000、测试全集及未知标签排除规则不变。HUGE所有变体拟合完整训练图；保存的8192索引是参考集合，不是其图损失上限。HIMVH沿用原12模型的训练索引、训练预处理和原生分区内离线上下文，不是逐笔因果检测。POT初始分位数0.85／风险0.01／最少20峰，失败标N/A，q99独立报告，不做点调整。AP与梯形PR-AUC分开存储，不能混称。','','9个交易任务来自5个来源，Sapienza三个粒度不是三份独立数据。两个图任务并非新增数据来源。S-FFSD账户已知异常占87.48%，常数AP即87.48%，全报警F1约93.32%但FPR100%；只看高AP或F1容易误判。金融涉诈标签与统计稀有性不同。各方法的图关系、预训练、历史和测试自适应信息预算仍有差异，见历史报告。','','独立复核包括所有源文件、旧索引、完整输入、训练统计、全图CPU重放、12个HIM模型的稀疏CPU字段误差探针、全量经验尾计数及权重、旧评分bitwise一致、全部POT/q99重拟合和指标混淆计数。GPU散射跨设备存在微小数值差异，沿用既有验证容差，模型持久化张量严格一致；不把重载数值噪声当作金融收益。','','[逐种子144配置](per_seed.json) · [288条操作点](metrics.csv) · [全部消融汇总](summary.csv) · [配对效应](paired_effects.csv) · [独立核验](verification.json) · [来源与环境](run_manifests.json) · [锚点复现差异](anchor_reproduction.csv) · [执行和归档](execution.json) · [交易主表](combined_transaction_summary.csv) · [账户主表](combined_account_graph_summary.csv)。','','## 复现命令','','模型与逐点误差按execution.json记录的CQ归档获取；仓库仅保存审计所需逐种子指标和哈希，不重复上传大量检查点。先按旧归档恢复101个输入文件与12个父模型目录。','','```bash','python -m finad_roleweight.experiment --family huge --device cuda --data INPUTS/tabular --graphs INPUTS/graphs --output HUGE_RESULTS','python -m finad_roleweight.experiment --family himvh --parents PARENT_RESULTS --data INPUTS/tabular --graphs INPUTS/graphs --output HIM_RESULTS','PYTHONPATH=. python scripts/verify_roleweight.py --huge HUGE_RESULTS --himvh HIM_RESULTS --parents PARENT_RESULTS --inputs INPUTS --old OLD_AE_RESULTS --output VERIFIED','PYTHONPATH=. python scripts/diagnose_roleweight.py --huge HUGE_RESULTS --himvh HIM_RESULTS --parents PARENT_RESULTS --inputs INPUTS --output results/roleweight_20261117', 'PYTHONPATH=. python scripts/interpret_roleweight.py --results results/roleweight_20261117', 'PYTHONPATH=. python scripts/report_roleweight.py --runs VERIFIED --output results/roleweight_20261117 --update-main','PYTHONPATH=. python scripts/plot_roleweight.py --results results/roleweight_20261117','```','','CQ固定源版本见run_manifests.json；所有GPU计算通过CQ免费资源执行。已有实验、源数据、其他项目作业保留。','','## Plain English','','We separated role indicators, semantic alignment, role centering and role scaling for HUGE. The fixed dual-channel primary retains original directional numeric information and adds shared-scale aligned channels. For HIMVH we froze12 verified parent models and recalibrated field errors using unlabeled empirical upper-tail evidence;96 score configurations isolate weighting effects without retraining. All48 new HUGE fits and all96 HIM score configurations are reported on unchanged cohorts and metrics. Improvements and regressions are retained. This is retrospective development using an already examined test set, not a blind validation or reproduction of the original supervised claims.','']
 if (a.output/'posthoc_interpretation_CN.md').exists():
  at=text.index('## 与现有方法的完整对比');text[at:at]=[(a.output/'posthoc_interpretation_CN.md').read_text(),'']
 (a.output/'README_CN.md').write_text('\n'.join(text))
 if a.update_main:
  path=ROOT/'docs/financial_ad_comparison_CN.md';current=path.read_text();marker='\n## 14. 角色通道与字段评分改进'
  if marker in current:current=current.split(marker)[0]
  current+='\n\n## 14. 角色通道与字段评分改进（2026-11-17）\n\n固定主方案HUGE-dual和HIMVH-cm-tail，在相同划分和评价规则下完成48次HUGE新拟合及96组HIM冻结模型重评分。144配置、288条操作点全部保留；没有真实标签训练或阈值调参。旧测试已用于诊断，属于回顾性评估。\n\n[最新完整中文报告、消融、378条主榜汇总与最佳方法](../results/roleweight_20261117/README_CN.md)。\n'
  for f in findings:current+=f'\n- {f["family"].upper()}：较上轮主方案AP改善{f["AP_improved_vs_previous_primary"]}/2；AP均值最高{f["AP_first"]}/2、POT F1最高{f["POT_F1_first"]}/2、q99 F1最高{f["q99_F1_first"]}/2。\n'
  notice='**最新角色／评分改进：** [HUGE与HIMVH新一轮144配置及中文报告](../results/roleweight_20261117/README_CN.md)，见第14节。旧报告及旧结果保留。\n\n'
  if notice not in current:current=current.replace('# 金融无监督异常检测综合对比报告\n\n','# 金融无监督异常检测综合对比报告\n\n'+notice,1)
  path.write_text(current)
 print('EXPORTED144 configs,288 operating rows,96 ablation summaries,378 main summaries')
if __name__=='__main__':main()
