"""Export every fixed variant, policy, control and matched comparison."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--update-main-report',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1]
records=json.loads((a.runs/'records.json').read_text());manifest=json.loads((a.runs/'run_manifest.json').read_text())
assert len(records)==162 and json.loads((a.runs/'verification.json').read_text())['status']=='passed'
tasks=manifest['tasks'];variants=manifest['variants'];a.output.mkdir(parents=True,exist_ok=True)
for name in ['run_manifest.json','input_files_sha256.json','verification.json','score_records.json','complete.json','scoring_complete.json','cold_warm_metrics.csv','score_driver_diagnostics.csv']:
    shutil.copyfile(a.runs/name,a.output/name)
def write(name,obj):(a.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def savecsv(name,rows):
    with (a.output/name).open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
write('per_seed.json',records)
rows=[];tails=[];thmodels=[];priors=[];audit=[]
for d in tasks:
    for seed in manifest['seeds']:
        priors.append(dict(dataset=d,seed=seed,prior=json.loads((a.runs/d/str(seed)/'training_prior.json').read_text())))
for r in records:
    run=a.runs/r['dataset']/str(r['seed'])/r['variant'];th=json.loads((run/'thresholds.json').read_text())
    tails.append(dict(dataset=r['dataset'],seed=r['seed'],variant=r['variant'],fitted=json.loads((run/'feature_pot.json').read_text())))
    thmodels.append(dict(dataset=r['dataset'],seed=r['seed'],variant=r['variant'],thresholds=th))
    with np.load(run/'scores.npz',allow_pickle=False) as z:cal,test=z['calibration'],z['test']
    g=th['guarded_pot']
    audit.append(dict(dataset=r['dataset'],seed=r['seed'],variant=r['variant'],active_dimensions=r['active_dimensions'],total_dimensions=r['total_dimensions'],
                      original_pot_status=th['pot']['status'],original_pot_shape=th['pot'].get('shape'),original_pot_threshold=th['pot'].get('threshold'),
                      original_calibration_alarm_fraction=g['original_calibration_alarm_fraction'],selected=g['selected'],reasons=';'.join(g['rejection_reasons']),
                      guarded_threshold=g['threshold'],guarded_calibration_alarm_fraction=g['calibration_alarm_fraction'],
                      quantile_threshold=th['quantile']['threshold'],quantile_fallback_ties_outside_interval=g['fallback_ties_outside_interval'],
                      calibration_max=float(cal.max()),test_max=float(test.max())))
    for op,v in r['metrics']['operating_points'].items():
        row={k:r[k] for k in ['dataset','seed','variant','method']};row.update(operating_point=op,status='ok' if v is not None else 'failed')
        row.update({k:v for k,v in r['metrics'].items() if k!='operating_points'})
        for key in ['precision','recall','f1','fpr','tp','fp','fn','tn','alarm_fraction']:row[key]=v[key] if v else None
        rows.append(row)
assert len(rows)==486
savecsv('metrics.csv',rows);savecsv('threshold_audit.csv',audit);write('feature_pot.json',tails);write('thresholds.json',thmodels);write('training_priors.json',priors)
metric_names=['average_precision','pr_auc_trapezoid','roc_auc','precision','recall','f1','fpr']
summary=[]
for d in tasks:
    for variant in variants:
        for op in ['pot','quantile','guarded_pot']:
            group=[r for r in rows if (r['dataset'],r['variant'],r['operating_point'])==(d,variant,op)];assert len(group)==3
            row=dict(track='tabular',dataset=d,variant=variant,method=group[0]['method'],operating_point=op,seeds=3,threshold_successes=sum(r['status']=='ok' for r in group),
                     source_commit=manifest['git_commit'],source_path='finad_kronos/stable_experiment.py',source_kind='scientific_code')
            for name in metric_names:
                values=[r[name] for r in group if r[name] is not None]
                row[name+'_mean']=float(np.mean(values)) if len(values)==3 else None
                row[name+'_std']=float(np.std(values,ddof=1)) if len(values)==3 else None
            for key in ['evaluated','unknown_excluded','positives','normal','prevalence']:
                assert len({r[key] for r in group})==1;row[key]=group[0][key]
            summary.append(row)
savecsv('summary.csv',summary)
factorial=[]
for row in rows:
    if row['variant'] in ['legacy','n','c','nc'] and row['operating_point'] in ['pot','guarded_pot']:
        factorial.append({**row,'N':int(row['variant'] in ['n','nc']),'C':int(row['variant'] in ['c','nc']),'G':int(row['operating_point']=='guarded_pot')})
assert len(factorial)==216;savecsv('factorial_metrics.csv',factorial)
deltas=[]
comparisons=[('N','n','legacy','pot','pot'),('C','c','legacy','pot','pot'),('NC','nc','legacy','pot','pot'),
             ('C_given_N','nc','n','pot','pot'),('N_given_C','nc','c','pot','pot'),('G_given_NC','nc','nc','guarded_pot','pot')]
for d in tasks:
    for label,left,right,lop,rop in comparisons:
        delta=dict(dataset=d,comparison=label,left_variant=left,right_variant=right,left_policy=lop,right_policy=rop,seeds=3)
        for metric in metric_names:
            values=[]
            for seed in manifest['seeds']:
                l=next(r[metric] for r in rows if (r['dataset'],r['variant'],r['seed'],r['operating_point'])==(d,left,seed,lop))
                rr=next(r[metric] for r in rows if (r['dataset'],r['variant'],r['seed'],r['operating_point'])==(d,right,seed,rop))
                if l is not None and rr is not None:values.append(l-rr)
            delta[metric+'_delta_mean']=float(np.mean(values)) if len(values)==3 else None
            delta[metric+'_delta_std']=float(np.std(values,ddof=1)) if len(values)==3 else None
        deltas.append(delta)
savecsv('paired_deltas.csv',deltas)
with (root/'results/kronos_matched_20261115/combined_transaction_summary.csv').open() as f:old=list(csv.DictReader(f))
for r in old:
    for key in metric_names:
        for suffix in ['mean','std']:r[key+'_'+suffix]=float(r[key+'_'+suffix]) if r[key+'_'+suffix] else None
    for key in ['seeds','threshold_successes','evaluated','unknown_excluded','positives','normal']:r[key]=int(r[key])
    r['prevalence']=float(r['prevalence'])
new=[{k:v for k,v in r.items() if k!='variant'} for r in summary if r['variant'] in ['nc','prior','persist'] and r['operating_point']!='guarded_pot']
combined=old+new;assert len(combined)==270
for r in summary:
    matched=next(x for x in old if x['dataset']==r['dataset'] and x['method']=='kronos_potratio_matched' and x['operating_point']=='pot')
    for key in ['evaluated','unknown_excluded','positives','normal','prevalence']:assert r[key]==matched[key]
    if r['variant']=='legacy' and r['operating_point'] in ['pot','quantile']:
        prev=next(x for x in old if x['dataset']==r['dataset'] and x['method']=='kronos_potratio_matched' and x['operating_point']==r['operating_point'])
        for key in metric_names:
            for suffix in ['mean','std']:np.testing.assert_allclose(r[key+'_'+suffix],prev[key+'_'+suffix],rtol=1e-12,atol=1e-12)
savecsv('combined_transaction_summary.csv',combined)
names=dict(iforest='IF',lof='LOF',knn='kNN',ocsvm='OCSVM',pca='PCA-SPE',hbos='HBOS',gmm='GMM',ae='AE',vae='VAE',deep_svdd='Deep SVDD',cae_if_core='CAE-IF',kronos_potratio_matched='Kronos-M',kronos_stable_nc='Kronos-S',train_prior_potratio='训练先验',persistence_stable_nc='持续值')
methods=list(names);vn=dict(legacy='原版',n='仅N',c='仅C',nc='N+C',prior='训练先验',persist='持续值N+C')
dn=dict(ulb='ULB信用卡',banksim='BankSim',s_ffsd='S-FFSD',sapienza_5S='Sapienza 5S',sapienza_15S='Sapienza 15S',sapienza_25S='Sapienza 25S',sd_stock='SD-FMM股票',sd_crypto='SD-FMM加密货币',sd_synth='SD-FMM合成')
index={(r['dataset'],r['method'],r['operating_point']):r for r in combined};sx={(r['dataset'],r['variant'],r['operating_point']):r for r in summary}
def fmt(x):return 'N/A' if x is None else f'{100*x:.2f}'
def ms(r,metric):return 'N/A' if r[metric+'_mean'] is None else fmt(r[metric+'_mean'])+' ± '+fmt(r[metric+'_std'])
def win(d,metric,op='pot'):
    valid=[m for m in methods if index[d,m,op][metric+'_mean'] is not None]
    maximum=max(index[d,m,op][metric+'_mean'] for m in valid)
    return [m for m in valid if abs(index[d,m,op][metric+'_mean']-maximum)<1e-12]
best=[]
for d in tasks:
    for metric,op in [('average_precision','pot'),('roc_auc','pot'),('f1','pot'),('f1','quantile')]:
        for m in win(d,metric,op):best.append(dict(dataset=d,metric=metric,operating_point=op,method=m,mean=index[d,m,op][metric+'_mean'],std=index[d,m,op][metric+'_std']))
savecsv('best_methods.csv',best)
def table(metric,op='pot'):
    text=['| 数据集 | '+' | '.join(names[m] for m in methods)+' |','|---|'+'---:|'*15]
    for d in tasks:
        ws=win(d,metric,op)
        text.append('| '+dn[d]+' | '+' | '.join(('**'+fmt(index[d,m,op][metric+'_mean'])+'**') if m in ws else fmt(index[d,m,op][metric+'_mean']) for m in methods)+' |')
    return '\n'.join(text)
def bestcell(d,metric,op='pot'):
    winners=win(d,metric,op)
    return '、'.join(names[m] for m in winners)+'：'+fmt(index[d,winners[0],op][metric+'_mean'])
besttable='\n'.join(['| 数据集 | AP最高 | POT F1最高 | 99%分位F1最高 |','|---|---|---|---|']+['| '+dn[d]+' | '+bestcell(d,'average_precision')+' | '+bestcell(d,'f1')+' | '+bestcell(d,'f1','quantile')+' |' for d in tasks])
main=['| 数据集 | AP | ROC-AUC | 原POT F1 | 护栏G F1 | 99%分位F1 | 原POT FPR |','|---|---:|---:|---:|---:|---:|---:|']
for d in tasks:
    core=sx[d,'nc','pot']
    main.append('| '+dn[d]+' | '+' | '.join([ms(core,'average_precision'),ms(core,'roc_auc'),ms(core,'f1'),ms(sx[d,'nc','guarded_pot'],'f1'),ms(sx[d,'nc','quantile'],'f1'),ms(core,'fpr')])+' |')
maintable='\n'.join(main)
apab=['| 数据集 | '+' | '.join(vn[v] for v in variants)+' |','|---|'+'---:|'*6]
f1ab=['| 数据集 | N0C0G0 | N0C0G1 | N1C0G0 | N1C0G1 | N0C1G0 | N0C1G1 | N1C1G0 | N1C1G1 |','|---|'+'---:|'*8]
for d in tasks:
    apab.append('| '+dn[d]+' | '+' | '.join(fmt(sx[d,v,'pot']['average_precision_mean']) for v in variants)+' |')
    f1ab.append('| '+dn[d]+' | '+' | '.join(fmt(sx[d,v,op]['f1_mean']) for v in ['legacy','n','c','nc'] for op in ['pot','guarded_pot'])+' |')
apab='\n'.join(apab);f1ab='\n'.join(f1ab)
def improved(metric,op='pot'):
    return sum(sx[d,'nc',op][metric+'_mean'] is not None and sx[d,'legacy',op][metric+'_mean'] is not None and sx[d,'nc',op][metric+'_mean']>sx[d,'legacy',op][metric+'_mean']+1e-12 for d in tasks)
apbetter=improved('average_precision');f1better=improved('f1');apwins=sum('kronos_stable_nc' in win(d,'average_precision') for d in tasks);f1wins=sum('kronos_stable_nc' in win(d,'f1') for d in tasks)
overview=dict(scoring_configurations=162,operating_point_records=486,factorial_records=216,combined_method_rows=270,AP_improved_vs_original_tasks=apbetter,common_POT_F1_improved_vs_original_tasks=f1better,AP_best_in_15_methods_tasks=apwins,common_POT_F1_best_in_15_methods_tasks=f1wins,
              guard_fallbacks_by_variant={v:sum(r['selected']=='quantile_fallback' for r in audit if r['variant']==v) for v in variants},
              nc_AP_exceeds_prior_tasks=sum(sx[d,'nc','pot']['average_precision_mean']>sx[d,'prior','pot']['average_precision_mean'] for d in tasks),
              nc_AP_exceeds_persistence_tasks=sum(sx[d,'nc','pot']['average_precision_mean']>sx[d,'persist','pot']['average_precision_mean'] for d in tasks),
              no_new_backbone_inference=True,retrospective_development=True)
write('overview.json',overview)
body=f'''# KronosAD稳定评分、冷启动与POT护栏实验（2026-11-16）

**这是同一已查看测试集上的回顾性开发评估。** 改进来自11月15日的诊断；本轮参数和完整消融先提交后评价，但不能据此称为新的独立验证。九任务来自五种数据来源，三种Sapienza粒度不是三份独立原始数据。

已完成162条评分配置、486条操作点记录及独立复算。沿用相同81个缓存、42/43/44种子、训练／校准抽样和完整测试位置；不删除冷启动点。真实冻结骨干预测来自上一轮27条缓存、共87130386个逐特征预测，不重新推理或随机采样。

完整方案为 **KronosAD-S：N+C评分，加G阈值护栏**。N给局部残差分母加训练尺度下限；C按有效历史数混合训练先验；G只检查校准POT并显式回退分位阈值。固定公式及实验矩阵见[方案](../../docs/kronos_stable_protocol_CN.md)。与经典方法的15法排名使用原共同POT／99%分位，护栏G另列，避免比较不同阈值政策。

相对原KronosAD-M，N+C评分的AP在{apbetter}/9个任务提高，**相同原POT操作点**F1在{f1better}/9个任务提高；这些是均值变化，不是统计显著性结论。在含两个无骨干对照的15法集合中，新评分AP最高{apwins}/9、共同POT F1最高{f1wins}/9。

## 完整改进方案结果

百分数，三个种子均值±样本标准差。护栏改变操作点，不改变AP；CSV另有全部P/R/F1/FPR和混淆矩阵。分位回退是公开规则，不能把它计作原POT拟合成功。

{maintable}

## 消融：残差与冷启动对AP的影响

N0C0为原版，N为稳定分母，C为冷启动先验混合。下表为AP均值，标准差和逐种子结果保留在CSV。两个对照均无Kronos骨干。

{apab}

## 消融：完整2×2×2组合的F1

G0使用共同原POT；G1使用护栏。每行同一N/C的两列完全共用分数。不得把每行最好配置选为方法成绩，主方案固定为N1C1G1。

{f1ab}

## 15法共同原POT F1

原Kronos-M、新Kronos-S和两个无骨干对照均使用与经典方法相同的最终原POT；该表没有把护栏成绩混入。粗体仅为该指标最高均值。

{table('f1')}

## 15法AP

{table('average_precision')}

## 各数据集最佳方法

候选集合包含10种经典方法、CAE-IF、Kronos-M、Kronos-S及训练先验／持续值两个对照；不同指标分别选最高均值，不做混合总分。

{besttable}

## 文件和评价边界

- [486条逐种子操作点](metrics.csv)、[162条汇总](summary.csv)、[216条完整因素记录](factorial_metrics.csv)、[配对增量及其标准差](paired_deltas.csv)。POT有失败种子时不只平均成功种子。
- [270条共同操作点汇总](combined_transaction_summary.csv)、[全部指标最佳方法](best_methods.csv)、[概览](overview.json)。经典及CAE-IF沿用此前同缓存结果；原Kronos旧成绩在新复算中重现。
- [原拟合／护栏接受与回退审计](threshold_audit.csv)、[全部阈值](thresholds.json)、[逐维标尺和失败维度](feature_pot.json)、[仅训练集先验](training_priors.json)。G不承诺测试FPR或有限样本覆盖，并列分数导致的报警比例偏离如实记录。
- [冷／热启动固定阈值子集](cold_warm_metrics.csv)、[误报驱动](score_driver_diagnostics.csv)。仅用于解释，完整测试仍为主表。先验对照不依赖历史方差，不能将其主导维度恰有近恒定历史误读为该算法使用了局部分母。
- [逐配置原始指标与哈希](per_seed.json)、[输入身份](input_files_sha256.json)、[独立复算](verification.json)、[源码及环境](run_manifest.json)。实际评分代码固定`{manifest['git_commit']}`；报告导出提交可更晚。评分先全部写盘，再单独解码测试标签。
- N+C的AP高于训练先验对照的任务数为{overview['nc_AP_exceeds_prior_tasks']}/9，高于持续值对照为{overview['nc_AP_exceeds_persistence_tasks']}/9；这不等于骨干在所有任务上都有增量价值。对照同样需要校准，不是零校准比较。
- 三种子误差条不代表跨时期独立验证。外部预训练重叠、人工日历、工程特征代理、无监督污染及未知标签覆盖的原有限制继续适用。账户图六方法表保持独立，没有新增Kronos图结果。

## 复现

固定方案及依赖见[协议](../../docs/kronos_stable_protocol_CN.md)。先准备上一轮完整缓存和预测；新输出使用独立目录，不覆盖历史实验。数值库为NumPy2.1.3、SciPy1.16.3、Pandas2.3.3、scikit-learn1.7.2。

```bash
python -m finad_kronos.stable_experiment --raw-runs /data/kronos/old_raw \\
  --original-runs /data/kronos/old_ratio --data /data/kronos/tabular \\
  --output /data/kronos/stable --stage score
python -m finad_kronos.stable_experiment --raw-runs /data/kronos/old_raw \\
  --original-runs /data/kronos/old_ratio --data /data/kronos/tabular \\
  --output /data/kronos/stable --stage evaluate
python scripts/verify_kronos_stable.py --runs /data/kronos/stable \\
  --raw-runs /data/kronos/old_raw --original-runs /data/kronos/old_ratio \\
  --data /data/kronos/tabular
python scripts/report_kronos_stable.py --runs /data/kronos/stable \\
  --output .local_data/kronos_stable_report
```

本轮只发布源码、全部配置指标、参数和验证证据；逐点分数及冻结预测仍在私有工作归档，Git不新增大模型权重或原始逐点预测。
'''
(a.output/'README_CN.md').write_text(body)
if a.update_main_report:
    report=root/'docs/financial_ad_comparison_CN.md';text=report.read_text()
    text=re.sub(r'^整理日期：[^。]+。','整理日期：2026-11-16；包含历史基线、原Kronos同协议结果及稳定评分／冷启动／POT护栏新消融。',text,count=1,flags=re.M)
    text=text.replace('**最新Kronos同协议全量结果：**','**11月15日原Kronos同协议全量结果：**',1)
    quick='**最新改进与消融：** [KronosAD-S完整结果、2×2×2消融及无骨干对照](../results/kronos_stable_20261116/README_CN.md)。第2节更新为15法共同操作点；护栏G另列，不混入经典原POT排名。**本轮是同一已查看测试集上的回顾性开发评估。**\n\n'
    text=text.replace('## 主要发现\n',quick+'## 主要发现\n',1)
    text=text.replace('## 主要发现\n','## 主要发现\n\n'+f'- **稳定评分与冷启动消融完成。** 相对原Kronos-M，N+C的AP在{apbetter}/9任务提高、共同原POT F1在{f1better}/9任务提高；15法中AP最高{apwins}/9、共同POT F1最高{f1wins}/9。训练先验与持续值对照一并纳入，不能将评分改善全部归功于预训练骨干。第11节给出完整边界。\n',1)
    text=text.replace('在当前12种方法中','在11月15日当时12种方法中',1)
    oldsentence='交易／点级轨道有9个共同任务，现比较10种经典方法、CAE-IF与同协议KronosAD-M共12种方法。'
    text=text.replace(oldsentence,'交易／点级轨道有9个共同任务，当前第2节比较10种经典方法、CAE-IF、原Kronos-M、新Kronos-S及训练先验／持续值两个对照，共15种评分／方法。',1)
    for section,nextsection,replacement in [('2.1','2.2',table('f1')),('2.2','2.3',table('average_precision'))]:
        pattern=rf'(### {re.escape(section)} [^\n]+\n\n).*?(?=\n### {re.escape(nextsection)} )'
        text,n=re.subn(pattern,lambda m:m[1]+replacement+'\n',text,flags=re.S);assert n==1
    pattern=r'(### 2\.3 每个数据集的最佳方法\n\n).*?(?=\n## 3\.)'
    intro='“最佳”指当前15种评分／方法相应指标的最高均值，不是显著性结论。候选包含两个无Kronos对照；新护栏不是共同原POT操作点，单独见第11节。九任务来自五来源，不把不同粒度算作独立验证。\n\n完整均值、标准差及来源见[270条共同操作点汇总](../results/kronos_stable_20261116/combined_transaction_summary.csv)；RGLD-ID／RTTAD-TTA仍不在此15法集合，旧轮成绩继续单独保留。\n'
    text,n=re.subn(pattern,lambda m:m[1]+besttable+'\n\n'+intro,text,flags=re.S);assert n==1
    section_start=text.index('## 2. 交易检测／行情点级检测');section_end=text.index('### 2.1',section_start)
    text=text[:section_start]+'## 2. 交易检测／行情点级检测\n\n检测对象为一笔交易或序列数据点。以下15法表保留相同输入、测试对象及最终原POT／99%分位标准。Kronos-S使用稳定分母与冷启动先验混合；完整方案的护栏G另报，不能把新护栏F1当作本表原POT成绩。两个无骨干对照与其他方法同时参与最高均值判断。\n\n'+text[section_end:]
    marker='### 1.3 如何理解指标'
    note='新增评分源提交`'+manifest['git_commit']+'`及[固定方案](kronos_stable_protocol_CN.md)在本轮新结果评价前发布，但设计利用了上轮已查看的测试诊断，因此本轮不是独立验证。相同划分与零标签拟合不消除这种回顾性开发偏差。\n\n'
    text=text.replace(marker,note+marker,1)
    text=text.replace('## 10. KronosAD同协议全量重跑：正式对比结果（2026-11-15）','## 10. 11月15日原KronosAD-M同协议重跑（历史12法）\n\n**本节保留当时12法集合的原版结果与解释。当前第2节已加入新评分和两个对照，改进实验见第11节；本节旧排名不随新候选集合改变。**',1)
    text=text.replace('**第2节的AP／POT F1及最佳方法表已重新计算，包含KronosAD-M。**','当时的AP／POT F1及最佳方法表包含KronosAD-M；当前表还包括第11节的新评分与对照。',1)
    text+='\n\n## 11. KronosAD稳定评分、冷启动与POT护栏改进（2026-11-16）\n\n'+body[body.index('**这是'):body.index('## 文件和评价边界')]
    text+='\n完整逐种子指标、护栏回退审计、训练先验参数、冷／热启动诊断、独立复算及复现命令见[改进与消融中文报告](../results/kronos_stable_20261116/README_CN.md)。原数据、原实验目录和账户图六方法结果保持不变。\n'
    report.write_text(text)
print(json.dumps(overview),flush=True)
