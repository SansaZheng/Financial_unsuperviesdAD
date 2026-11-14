"""Export a verified whole 27-run cohort and update the Chinese comparison tables."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--update-main-report',action='store_true')
p.add_argument('--ablation-runs',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
records=json.loads((a.runs/'records.json').read_text());verified=json.loads((a.runs/'verification.json').read_text())
tasks=['ulb','banksim','s_ffsd','sapienza_5S','sapienza_15S','sapienza_25S','sd_stock','sd_crypto','sd_synth']
assert verified['status']=='passed' and verified['runs']==27
assert {(r['dataset'],r['seed']) for r in records}=={(d,s) for d in tasks for s in (42,43,44)} and len(records)==27
assert json.loads((a.runs/'complete.json').read_text())==dict(finished=27,failed=0,expected=27)
ablation=json.loads((a.ablation_runs/'records.json').read_text())
assert len(ablation)==27 and {(r['dataset'],r['seed']) for r in ablation}=={(d,s) for d in tasks for s in (42,43,44)}
assert json.loads((a.ablation_runs/'verification.json').read_text())['status']=='passed'
for r in records:
    original=next(x for x in ablation if (x['dataset'],x['seed'])==(r['dataset'],r['seed']))
    assert r['checks']['forecasts_sha256']==original['checks']['forecasts_sha256']
a.output.mkdir(parents=True,exist_ok=True)
for name in ['run_manifest.json','records.json','verification.json','complete.json','failures.json','input_files_sha256.json','assets.json']:
    shutil.copyfile(a.runs/name,a.output/('per_seed.json' if name=='records.json' else name))
def write(name,value):
    (a.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def savecsv(name,rows):
    with (a.output/name).open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
write('max_error_ablation_per_seed.json',ablation)
savecsv('max_error_ablation_metrics.csv',[
    {**{k:r[k] for k in ('dataset','seed','method')},'operating_point':op,
     **{k:v for k,v in r['metrics'].items() if k!='operating_points'},
     **{k:values[k] if values is not None else None for k in ('precision','recall','f1','fpr','tp','fp','fn','tn')}}
    for r in ablation for op,values in r['metrics']['operating_points'].items()])
metrics=[];thresholds=[];feature_tails=[];original_or=[];tail_diagnostics=[]
for r in records:
    feature_tails.append(dict(dataset=r['dataset'],seed=r['seed'],fitted=json.loads((a.runs/r['dataset']/str(r['seed'])/'feature_pot.json').read_text())))
    thresholds.append(dict(dataset=r['dataset'],seed=r['seed'],thresholds=json.loads((a.runs/r['dataset']/str(r['seed'])/'thresholds.json').read_text())))
    original=json.loads((a.runs/r['dataset']/str(r['seed'])/'original_or_metrics.json').read_text())
    original_or.append(dict(dataset=r['dataset'],seed=r['seed'],operating_point='original_or_threshold_1',included_in_common_ranking=False,
                            **{k:v for k,v in original.items() if k!='operating_points'},**original['operating_points']['original_or']))
    th=thresholds[-1]['thresholds'];pot=th['pot']
    with np.load(a.runs/r['dataset']/str(r['seed'])/'scores.npz',allow_pickle=False) as z:cal,test=z['calibration'],z['test']
    diagnostic=dict(dataset=r['dataset'],seed=r['seed'],active_dimensions=r['model_info']['active_dimensions'],total_dimensions=r['model_info']['total_dimensions'])
    for phase,scores in [('calibration',cal),('test',test)]:
        for label,q in [('q50',.5),('q85',.85),('q99',.99),('max',1.)]:diagnostic[phase+'_'+label]=float(np.quantile(scores,q))
    diagnostic.update(pot_status=pot['status'],pot_shape=pot.get('shape'),pot_threshold=pot.get('threshold'),quantile_threshold=th['quantile']['threshold'],
                      pot_threshold_over_test_max=pot['threshold']/float(test.max()) if pot['status']=='ok' and test.max()>0 else None,
                      pot_alarm_count_all_scored=int(np.count_nonzero(test>=pot['threshold'])) if pot['status']=='ok' else None,
                      quantile_alarm_count_all_scored=int(np.count_nonzero(test>=th['quantile']['threshold'])))
    tail_diagnostics.append(diagnostic)
    for op,values in r['metrics']['operating_points'].items():
        row={k:r[k] for k in ('track','dataset','method','seed','seconds','device')}
        row.update({k:v for k,v in r['metrics'].items() if k!='operating_points'})
        row.update(operating_point=op,status='ok' if values is not None else 'failed')
        for k in ['precision','recall','f1','fpr','tp','fp','fn','tn','alarm_fraction']:row[k]=values[k] if values is not None else None
        metrics.append(row)
savecsv('metrics.csv',metrics);write('thresholds.json',thresholds);write('feature_pot.json',feature_tails);savecsv('original_or_metrics.csv',original_or)
savecsv('threshold_diagnostics.csv',tail_diagnostics)
summary=[]
metric_names=['average_precision','pr_auc_trapezoid','roc_auc','precision','recall','f1','fpr']
for d in tasks:
    for op in ['pot','quantile']:
        group=[x for x in metrics if x['dataset']==d and x['operating_point']==op]
        row=dict(track='tabular',dataset=d,method='kronos_potratio_matched',operating_point=op,seeds=3,threshold_successes=sum(x['status']=='ok' for x in group),
                 source_commit=json.loads((a.runs/'run_manifest.json').read_text())['git_commit'],source_path='finad_kronos/rescore.py',source_kind='scientific_code')
        # Require all three operating points for their aggregate. Do not hide failed seeds.
        for metric in metric_names:
            vals=[x[metric] for x in group if x[metric] is not None]
            row[metric+'_mean']=float(np.mean(vals)) if len(vals)==3 else None
            row[metric+'_std']=float(np.std(vals,ddof=1)) if len(vals)==3 else None
        for k in ['evaluated','unknown_excluded','positives','normal','prevalence']:
            assert len({x[k] for x in group})==1;row[k]=group[0][k]
        summary.append(row)
savecsv('summary.csv',summary)
with (root/'results/comparison_20261114/transaction_summary.csv').open() as f:baseline=list(csv.DictReader(f))
for row in baseline:
    row['source_kind']='published_summary'
    for metric in metric_names:
        for suffix in ['mean','std']:row[metric+'_'+suffix]=float(row[metric+'_'+suffix])
    for k in ['seeds','threshold_successes','evaluated','unknown_excluded','positives','normal']:row[k]=int(row[k])
    row['prevalence']=float(row['prevalence'])
combined=baseline+summary;assert len(combined)==216
for r in summary:
    other=next(x for x in baseline if x['dataset']==r['dataset'] and x['method']=='cae_if_core' and x['operating_point']==r['operating_point'])
    assert all(r[k]==other[k] for k in ['evaluated','positives','normal','unknown_excluded','prevalence'])
savecsv('combined_transaction_summary.csv',combined)
methods=['iforest','lof','knn','ocsvm','pca','hbos','gmm','ae','vae','deep_svdd','cae_if_core','kronos_potratio_matched']
names=dict(iforest='IF',lof='LOF',knn='kNN',ocsvm='OCSVM',pca='PCA-SPE',hbos='HBOS',gmm='GMM',ae='AE',vae='VAE',deep_svdd='Deep SVDD',cae_if_core='CAE-IF',kronos_potratio_matched='KronosAD-M')
dn=dict(ulb='ULB信用卡',banksim='BankSim',s_ffsd='S-FFSD',sapienza_5S='Sapienza 5S',sapienza_15S='Sapienza 15S',sapienza_25S='Sapienza 25S',sd_stock='SD-FMM股票',sd_crypto='SD-FMM加密货币',sd_synth='SD-FMM合成')
index={(r['dataset'],r['method'],r['operating_point']):r for r in combined}
def value(d,m,metric,op='pot'):return index[d,m,op][metric+'_mean']
def winners(d,metric,op='pot'):
    available=[m for m in methods if value(d,m,metric,op) is not None]
    best=max(value(d,m,metric,op) for m in available)
    return [m for m in available if abs(value(d,m,metric,op)-best)<1e-12]
best_rows=[]
for d in tasks:
    for metric,op in [('average_precision','pot'),('roc_auc','pot'),('f1','pot'),('f1','quantile')]:
        for m in winners(d,metric,op):best_rows.append(dict(track='tabular',dataset=d,metric=metric,operating_point=op,method=m,mean=value(d,m,metric,op),std=index[d,m,op][metric+'_std']))
savecsv('best_methods.csv',best_rows)
def fmt(v):return 'N/A' if v is None else f'{100*v:.2f}'
def table(metric,op='pot'):
    lines=['| 数据集 | '+' | '.join(names[m] for m in methods)+' |','|---|'+'---:|'*len(methods)]
    for d in tasks:
        best=winners(d,metric,op)
        lines.append('| '+dn[d]+' | '+' | '.join('**'+fmt(value(d,m,metric,op))+'**' if m in best else fmt(value(d,m,metric,op)) for m in methods)+' |')
    return '\n'.join(lines)
def bestcell(d,metric,op='pot'):
    ms=winners(d,metric,op);return '、'.join(names[m] for m in ms)+'：'+fmt(value(d,ms[0],metric,op))
best_table='\n'.join(['| 数据集 | AP最高 | POT F1最高 | 99%分位F1最高 |','|---|---|---|---|']+['| '+dn[d]+' | '+bestcell(d,'average_precision')+' | '+bestcell(d,'f1')+' | '+bestcell(d,'f1','quantile')+' |' for d in tasks])
ablation_table=['| 数据集 | 主方法POT比值 AP | 最大误差消融 AP | 主方法POT比值 F1 | 最大误差消融 F1 |','|---|---:|---:|---:|---:|']
for d in tasks:
    group=[r for r in ablation if r['dataset']==d]
    ap=float(np.mean([r['metrics']['average_precision'] for r in group]))
    f1s=[r['metrics']['operating_points']['pot']['f1'] for r in group if r['metrics']['operating_points']['pot'] is not None]
    f1=float(np.mean(f1s)) if len(f1s)==3 else None
    ablation_table.append('| '+dn[d]+' | '+fmt(value(d,'kronos_potratio_matched','average_precision'))+' | '+fmt(ap)+' | '+fmt(value(d,'kronos_potratio_matched','f1'))+' | '+fmt(f1)+' |')
ablation_table='\n'.join(ablation_table)
def meanstd(d,metric,op='pot'):
    r=index[d,'kronos_potratio_matched',op]
    return 'N/A' if r[metric+'_mean'] is None else fmt(r[metric+'_mean'])+' ± '+fmt(r[metric+'_std'])
ktable='\n'.join(['| 数据集 | 全量已知评价数／异常数（占比） | AP | ROC-AUC | POT F1 | 99%分位F1 | POT FPR |','|---|---:|---:|---:|---:|---:|---:|']+['| '+dn[d]+f" | {index[d,'kronos_potratio_matched','pot']['evaluated']}／{index[d,'kronos_potratio_matched','pot']['positives']}（{100*index[d,'kronos_potratio_matched','pot']['prevalence']:.2f}%） | "+' | '.join([meanstd(d,'average_precision'),meanstd(d,'roc_auc'),meanstd(d,'f1'),meanstd(d,'f1','quantile'),meanstd(d,'fpr')])+' |' for d in tasks])
apwins=sum('kronos_potratio_matched' in winners(d,'average_precision') for d in tasks)
f1wins=sum('kronos_potratio_matched' in winners(d,'f1') for d in tasks)
potok=sum(x['status']=='ok' for x in metrics if x['operating_point']=='pot')
cold=[]
for d in tasks:
    sample=next(r for r in records if r['dataset']==d and r['seed']==42)
    c=sample['model_info']['cold_start_counts']['test']
    cold.append(dict(dataset=d,test_rows=c['scored'],cold_start_rows=c['cold_start'],no_history_rows=c['no_history']))
savecsv('cold_start_counts.csv',cold)
description=f'''# KronosAD同协议全量对比：27次运行（2026-11-15）

完成9任务×种子42/43/44共27次运行，54条操作点评价。POT成功{potok}/27，其余失败明确N/A。所有测试位置均评分，未知标签只在最终评价屏蔽；每个保存预测到标量分数、配对抽样、训练缩放器、阈值及指标独立复算通过。

**方法名KronosAD-M = `kronos_potratio_matched`。** 使用真实冻结Kronos-mini＋Tokenizer-2k，在相同共享工程特征上逐特征预测，逐维标准化平方误差先除以其校准POT标尺，再取最大值合成标量，再用与基线相同的POT和99%校准分位阈值。它保留源方法的逐维POT标尺聚合，最终操作点按共同标准重校准；不是原生OHLCVA或旧小样本成绩。最大误差版在查看本轮金融指标前明确改列消融，全部保留而非按成绩挑选。规则在查看本轮指标前固定，见[预先固定协议](../../docs/kronos_matched_protocol_CN.md)。

完整输入、划分、校准／测试位置、缩放器和指标统一，不意味着预训练语料、历史上下文和计算量完全相同。Kronos使用过去观测及外部冻结预训练，其他基线的访问范围详见综合报告；不作速度或严格同信息预算结论。

## 新增方法结果

数值为百分数，三个种子均值±样本标准差，无点调整、无标签调参。完整表中AP最高的任务数为{apwins}/9，POT F1最高为{f1wins}/9，均为最高均值描述而非统计显著性结论。

异常占比是常数分数的AP参照：AP必须结合它、ROC-AUC和误报率解释，不能把不同数据集的AP直接排成总榜。三个种子的标准差反映本轮固定划分下的随机性，不是独立新时期验证或置信区间。

{ktable}

## 全部12种交易方法：POT F1

{table('f1')}

## 全部12种交易方法：AP

{table('average_precision')}

## 各数据集最佳方法

{best_table}

## 文件与核验

**保留预先登记的聚合消融**：两版共用完全相同的冻结预测；主方法在查看本轮指标前按源实现保真确定，不挑选下表每行较好的版本参与主表排名。

{ablation_table}

- [54条逐种子指标](metrics.csv)、[18条Kronos汇总](summary.csv)、[216条全部方法汇总](combined_transaction_summary.csv)、[各指标最佳方法](best_methods.csv)。CSV使用0到1比例，POT任一种子失败不以其余成功种子平均冒充完整成绩。
- [原最大误差消融全部逐种子记录](max_error_ablation_per_seed.json)与[54条消融操作点](max_error_ablation_metrics.csv)完整保留；[逐维POT标尺、失败维度及原因](feature_pot.json)记录主方法的额外无标签校准步骤。
- [原逐维OR规则的固定比值阈值1](original_or_metrics.csv)仅作补充，不参与共同POT／99%分位排名，不选择其与共同操作点较好的成绩。
- [校准／测试分数分位、最大值与POT外推](threshold_diagnostics.csv)用于区分数学拟合成功与检测有效；全部评分位置的报警数包含未知标签，不等于已知正常误报数。
- [逐种子配置与指标](per_seed.json)、[阈值及GPD参数](thresholds.json)、[全量预测／指标独立复算](verification.json)、[输入文件哈希](input_files_sha256.json)、[源码及环境](run_manifest.json)、[冻结权重及官方源码哈希](assets.json)。
- [冷启动覆盖](cold_start_counts.csv)：不足32步左填训练中位数，不丢弃序列开头测试点、不跨案例拼接；原模型与全量逐特征预测保存在私有CQ归档，不加入Git。
- [固定阈值下的冷／热启动子集](cold_warm_metrics.csv)、[分数主导特征](dominant_features.csv)、[近恒定历史与误报关联](score_driver_diagnostics.csv)、[事后诊断边界](diagnostics_protocol.json)：只作解释，不重拟合、不剔除难点、不替代完整测试成绩，也不把关联写成已证实因果。
- 科学源码以运行清单为准，报告导出可以在后续提交完成。经典方法来自此前同缓存实验；账户图表保持独立，不将本方法虚构为账户图模型。旧Kronos历史参考不参与本轮排名。
- 汇总CSV的`source_kind`区分来源：既有基线的`source_commit/path`指向已发布汇总；新Kronos行指向实际科学代码提交，结果及复算证据在本目录，避免把后续报告提交冒充实际运行版本。
'''
(a.output/'README_CN.md').write_text(description)
if a.update_main_report:
    report=root/'docs/financial_ad_comparison_CN.md';text=report.read_text()
    text=re.sub(r'^整理日期：[^。]+。','整理日期：2026-11-15；包含此前基线、Kronos旧协议参考及本轮同协议全量结果。',text,count=1,flags=re.M)
    text=text.replace('本文汇总已完成实验，不增加训练、不重新选择阈值，也不改变历史结果。','本文保留历史实验，并新增真实冻结Kronos推理与按预先固定规则进行的无标签阈值校准；没有根据测试标签重选阈值。',1)
    for heading,nextheading,replacement in [('2.1','2.2',table('f1')),('2.2','2.3',table('average_precision'))]:
        pattern=rf'(### {re.escape(heading)} [^\n]+\n\n).*?(?=\n### {re.escape(nextheading)} )'
        text,n=re.subn(pattern,lambda match:match[1]+replacement+'\n',text,flags=re.S);assert n==1
    pattern=r'(### 2\.3 每个数据集的最佳方法\n\n).*?(?=\n## 3\.)'
    intro='“最佳”只指当前12种方法在相应指标上的最高均值，不是显著性结论。排序能力与固定阈值报警能力分开评价；SD-FMM股票测试仅6条序列，方差和泛化限制需要同时考虑。\n\n原先另有RGLD-ID和RTTAD-TTA，不属于本表10种经典方法，未计入此候选集合；不同候选集合下最佳方法可能改变。旧结果见第1.2节固定源报告。\n\n完整共同指标与标准差见[216条统一交易汇总](../results/kronos_matched_20261115/combined_transaction_summary.csv)，Kronos完整结果见[同协议27次运行报告](../results/kronos_matched_20261115/README_CN.md)。CSV使用原始比例。\n'
    text,n=re.subn(pattern,lambda match:match[1]+best_table+'\n\n'+intro,text,flags=re.S);assert n==1
    text=text.replace('交易／点级轨道有9个共同任务，比较10种经典方法与CAE-IF。','交易／点级轨道有9个共同任务，现比较10种经典方法、CAE-IF与同协议KronosAD-M共12种方法。',1)
    text=text.replace('新增方法为CAE-IF核心适配、EB-GAD作者发布版零标签设置、Audit-HCL零标签适配。','新增方法为CAE-IF核心适配、EB-GAD作者发布版零标签设置、Audit-HCL零标签适配，以及本轮真实冻结Kronos-mini的KronosAD-M代理适配。',1)
    text=text.replace('以下两张表完整列出10种经典方法及CAE-IF。','以下两张表完整列出10种经典方法、CAE-IF及KronosAD-M。**KronosAD-M即本轮同输入、全量测试的KronosAD-Proxy，采用保留源逐维POT标尺的比值聚合；旧小样本Proxy不在主表。** 详细边界见第10节。',1)
    text=text.replace('## 7. 新增：KronosAnomalyDetection结果参考对比（2026-11-15）','## 7. 历史参考：KronosAnomalyDetection旧协议结果（2026-11-15）\n\n**本节保留旧Proxy与11月14日主表的历史参考，表中原11种方法的范围不随新轮更新；当前同输入／同测试位置Kronos结果已加入第2节，完整说明见第10节。**',1)
    text=text.replace('在本表10种经典方法＋CAE-IF范围内，','在原10种经典方法＋CAE-IF的基线范围内，',1)
    text=text.replace('**经典方法在多数交易任务上仍有优势。** 在原10种经典方法＋CAE-IF的基线范围内，POT F1最高分别为：ULB的PCA、BankSim的VAE、S-FFSD的LOF、Sapienza三种粒度的GMM、股票的OCSVM、加密货币的IF，以及合成数据的CAE-IF。',
                      '**最佳方法依赖数据集及指标。** 加入KronosAD-M后的12法完整最高名单已更新在第2.3节，分别列出AP、共同POT F1和99%分位F1；不能用一种指标的第一名代表所有检测能力。',1)
    text=text.replace('超过本表其余方法，但该任务的AP仍低于IF。','超过此前其余10种经典基线，但AP仍低于IF；加入Kronos后的当前最高方法见第2.3节。',1)
    text=text.replace('- **已加入KronosAnomalyDetection结果，但不混排不同协议。** 第7节分别列出原生OHLCVA、原Proxy、固定联合无监督和验证选参方案，提供与现有主表的并排参考；测试划分、样本、特征和标签预算不同，原最佳名单不变。',f'- **KronosAD同协议全量重跑已加入交易主表。** 完成27次运行，POT成功{potok}/27；在当前12种方法中AP最高{apwins}/9、POT F1最高{f1wins}/9。完整测试对象、输入缓存和评价规则已配对核验。第7节仅保留旧协议历史参考，不纳入当前排名。',1)
    for heading in ['### 7.4 原Kronos-Proxy逐数据版本结果','### 7.5 相同Proxy测试位置下的评分方案对比','### 7.6 与经典方法及CAE-IF并排参考']:
        text=text.replace(heading+'\n',heading+'\n\n**此处是旧协议小样本历史结果。最新同输入、完整测试的KronosAD结果已进入第2节，并有[独立27次运行报告](../results/kronos_matched_20261115/README_CN.md)。**\n',1)
    text=text.replace('本次仅整理报告及已有聚合结果，没有重新提交CQ训练任务，没有改变历史实验、原始数据、模型或阈值。','11月14日的11法汇总仅整理既有结果；本轮另有真实Kronos全量冻结推理及无标签评分，详见第10节，不覆盖任何历史实验、原始数据或原阈值。',1)
    text=text.replace('**第2节、第3节的同协议排名保留；Kronos结果作为不同协议的历史参考，不把较大的数字直接加粗为“全体最佳”。** 本次未重新训练、微调或调用Kronos推理，也没有在新时间段获得独立验证。','**本节保留当时不同协议的历史参考，不参与当前第2节12法排名。** 当时的参考整理没有重新调用Kronos推理；本轮第10节已实际重跑，仍没有在独立新时期获得前瞻验证。',1)
    text=text.replace('主表最高方法仍只在原11种方法内部选出；Proxy的F1','此历史表左侧最高方法仅在原11种方法内选出，当前12法名单见第2.3节；旧Proxy的F1',1)
    text=text.replace('## 8. Kronos表现与风险的解释','## 8. 旧协议Kronos结果的解释与风险',1)
    text=text.replace('## 9. Kronos新增溯源与核验','## 9. 旧协议Kronos参考的溯源与核验',1)
    text=text.replace('## 6. 可追溯文件及核验范围\n','## 6. 可追溯文件及核验范围\n\n本节原链接保留11月14日的11法基线；当前含Kronos的12法[统一交易汇总](../results/kronos_matched_20261115/combined_transaction_summary.csv)、[最佳方法](../results/kronos_matched_20261115/best_methods.csv)及[全量复算证据](../results/kronos_matched_20261115/verification.json)在新结果目录，不能把旧最佳CSV误认为已更新的12法名单。\n',1)
    text=text.replace('**不包括Kronos仓库**。','**不包括旧Kronos仓库的缓存**；第10节新增KronosAD-M已另行核对相同81个输入文件。',1)
    text=text.replace('本文不把这项尚未完成的实验写成已有结论。','本节不把这些旧成绩写成公平比较结论；该共同协议实验现已在第10节实际完成。',1)
    text=text.replace('本次仅补充中文报告和来源明确的聚合参考数据，现有主表最佳方法、账户图结果、原始数据及历史实验均未改变。','本节所述历史补充仅整理聚合参考数据；当前第2节排名已根据第10节的新实验更新，账户图结果、原始数据及历史实验保留。',1)
    addition=f'''\n\n## 10. KronosAD同协议全量重跑：正式对比结果（2026-11-15）

本节替代“只有不同协议历史参考”的状态，使用与基线完全相同的9任务缓存、特征、划分、种子、校准抽样和全部测试位置，实际完成27次真实冻结模型推理运行。**第2节的AP／POT F1及最佳方法表已重新计算，包含KronosAD-M。** 旧第7节的数据不被混入。

{ktable}

KronosAD-M的全名为KronosAD-Proxy-Matched（`kronos_potratio_matched`）。为统一阈值规则，保留源方法的逐维POT阈值比值最大聚合，内部标尺只在配对校准位置拟合（q=.90、risk=.01、至少30个超额），然后调用与基线相同的标量POT（q=.85、risk=.01、至少20个超额）及99%分位函数。源默认比值阈值1另列补充，不用于共同排名；首版最大误差保留消融，不按成绩择优。冷启动按预先规定左填训练中位数，所有序列开头位置均评分；按事件／案例重置历史，严格不含当前或未来观测。没有标签训练、验证选参或测试最优阈值搜索。

本轮统一的是输入和评价对象／规则。Kronos仍有外部预训练和历史上下文信息，其他方法也有各自训练目标与信息范围；不能据此声称训练语料、计算量和所有信息预算相同。单一指标的最高均值不等于显著提升或部署有效。完整协议、冷启动统计、逐种子指标和独立复算见[27次运行报告](../results/kronos_matched_20261115/README_CN.md)及[固定协议](kronos_matched_protocol_CN.md)。账户图六方法表没有新增虚构Kronos图结果。
'''
    assert '## 10.' not in text;text+=addition
    quick='**最新Kronos同协议全量结果：** [27次运行与逐种子指标](../results/kronos_matched_20261115/README_CN.md)。交易主表已包含KronosAD-M；第7节是旧协议历史参考，账户图结果仍独立。\n\n'
    text=text.replace('## 主要发现\n',quick+'## 主要发现\n',1)
    report.write_text(text)
print(json.dumps(dict(runs=27,metric_rows=54,combined_rows=216,ap_wins=apwins,pot_f1_wins=f1wins,pot_successes=potok)))
