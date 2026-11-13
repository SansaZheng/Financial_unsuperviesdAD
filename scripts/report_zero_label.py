"""Publish aggregate results without transactions, account IDs or raw scores."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
records=json.loads((args.runs/'records.json').read_text())
verify=json.loads((args.runs/'verification.json').read_text())
assert verify['status']=='passed' and verify['runs']==144
tabular=('ulb','banksim','s_ffsd','sapienza_5S','sapienza_15S','sapienza_25S','sd_stock','sd_crypto','sd_synth')
base=('cae_if_core','iforest','ae','pca')
expected={(track,d,m,s) for track,ds,methods in [('tabular',tabular,base),
 ('graph',('banksim','s_ffsd'),base+('ebgad_release_zero','audit_hcl_zero_adapted'))]
 for d in ds for m in methods for s in (42,43,44)}
assert {(r['track'],r['dataset'],r['method'],r['seed']) for r in records}==expected
for f in ('run_manifest.json','verification.json','complete.json','failures.json'):
    shutil.copyfile(args.runs/f,args.output/f)
(args.output/'per_seed.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
threshold_records=[]
for r in records:
    path=args.runs/r['track']/r['dataset']/r['method']/str(r['seed'])/'thresholds.json'
    threshold_records.append({**{k:r[k] for k in ('track','dataset','method','seed')},
                              'thresholds':json.loads(path.read_text())})
(args.output/'thresholds.json').write_text(json.dumps(threshold_records,indent=2,allow_nan=False)+'\n')
rows=[]
for r in records:
    for name,op in r['metrics']['operating_points'].items():
        row={k:r[k] for k in ('track','dataset','method','seed','seconds')}
        row.update({k:v for k,v in r['metrics'].items() if k!='operating_points'})
        row.update(operating_point=name,status='ok' if op is not None else 'failed')
        for k in ('precision','recall','f1','fpr','tp','fp','fn','tn','alarm_fraction'):
            row[k]=op[k] if op else None
        rows.append(row)
def save_csv(path,values):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
save_csv(args.output/'metrics.csv',rows)
summary=[]
for key in sorted({(r['track'],r['dataset'],r['method'],r['operating_point']) for r in rows}):
    group=[r for r in rows if (r['track'],r['dataset'],r['method'],r['operating_point'])==key]
    out=dict(zip(('track','dataset','method','operating_point'),key));out['runs']=len(group)
    out['threshold_successes']=sum(r['status']=='ok' for r in group)
    for metric in ('average_precision','pr_auc_trapezoid','roc_auc','precision','recall','f1','fpr','seconds'):
        vals=[r[metric] for r in group if r[metric] is not None]
        out[metric+'_n']=len(vals);out[metric+'_mean']=float(np.mean(vals)) if vals else None
        out[metric+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
    summary.append(out)
save_csv(args.output/'summary.csv',summary)
def fmt(row,metric):
    mean,std=row[metric+'_mean'],row[metric+'_std']
    value='N/A' if mean is None else (f'{100*mean:.2f} ± {100*std:.2f}' if std is not None else f'{100*mean:.2f}')
    if metric=='f1' and row['threshold_successes']!=row['runs']:
        value+=f"（{row['threshold_successes']}/{row['runs']}成功）"
    return value
names={'cae_if_core':'CAE-IF 核心适配','ebgad_release_zero':'EB-GAD 作者发布版','audit_hcl_zero_adapted':'Audit-HCL 零标签适配',
       'iforest':'Isolation Forest','ae':'AE','pca':'PCA-SPE'}
lines=['# 零标签金融异常检测：144次真实运行（2026-11-14）','',
       '**本结果为仓库金融数据上的独立适配实验，不是原论文成绩复刻。** CAE-IF只取得公开摘要；EB-GAD复现作者发布选择器，非论文附录有序选择器；Audit-HCL为真实支付关系和固定200更新预算的公式适配。详见[冻结协议](../../docs/zero_label_protocol_CN.md)。','',
       '数据来自5种完整可用来源：ULB、BankSim、S-FFSD、Sapienza和SD-FMM。Sapienza按3种时间粒度、SD-FMM按stock/crypto/synth分成9个交易/时序任务，并非9个相互独立的数据来源。仅有区间标签的ADMA21、仅有告警表的SynthAML和未生成交易的CardSim输入不纳入。','',
       '种子42/43/44，数值为百分数的均值 ± 样本标准差。所有模型与阈值均不使用标签；POT失败按N/A记录。AP是Average Precision，另在CSV提供梯形PR-AUC。无点调整、无标签阈值搜索。EB为确定性过程，只需计算4个不同的校准/测试图，三种子记录复用这些分数；零标准差不代表三次独立拟合，其缓存命中耗时也不宜与训练模型直接排名。','',
       '[288套逐种子操作点](metrics.csv) · [96组阈值汇总](summary.csv) · [模型配置与逐种子记录](per_seed.json) · [阈值、GPD参数及失败原因](thresholds.json) · [独立复算核验](verification.json) · [软件/源码版本](run_manifest.json)','']
for track,title in [('tabular','交易/时序轨道：9组数据，完整测试分区'),('graph','原生支付关系图轨道：仅评价固定子图中的付款账户')]:
    lines+=['## '+title,'','| 数据集 | 方法 | AP | ROC-AUC | POT F1 | 99%分位 F1 |','|---|---|---:|---:|---:|---:|']
    for row in summary:
        if row['track']!=track or row['operating_point']!='pot':continue
        q=next(v for v in summary if (v['track'],v['dataset'],v['method'],v['operating_point'])==(track,row['dataset'],row['method'],'quantile'))
        lines.append(f"| {row['dataset']} | {names[row['method']]} | {fmt(row,'average_precision')} | {fmt(row,'roc_auc')} | {fmt(row,'f1')} | {fmt(q,'f1')} |")
    lines+=['','| 数据集 | 测试总数 | 已知正常 | 已知异常 | 排除未知 | 已知样本异常占比 |','|---|---:|---:|---:|---:|---:|']
    for dataset in sorted({r['dataset'] for r in records if r['track']==track}):
        m=next(r['metrics'] for r in records if r['track']==track and r['dataset']==dataset)
        lines.append(f"| {dataset} | {m['evaluated']+m['unknown_excluded']} | {m['normal']} | {m['positives']} | {m['unknown_excluded']} | {100*m['prevalence']:.2f}% |")
    lines+=['']
lines+=['## 结果应如何理解','',
 '零标签指模型训练与阈值校准不用异常标签，不代表训练集无异常。BankSim是模拟器生成的合成支付数据；SD-FMM synth也是合成数据。图中的“原生关系”指直接保留发布数据中的付款/收款边，不表示BankSim来自真实银行业务。',
 '两轨道检测单位和信息范围不同，不能混合排名。交易轨道比较CAE-IF和相同新划分/预算的IF、AE、PCA；图轨道比较六种方法，其中图方法还看到真实交易关系及收款方属性。图每阶段仅选择最多4096个付款账户，保留其全部交易及对手方；不是全量交易评估。',
 '', 'S-FFSD未知标签参加无标签训练和校准，最后才从评价中排除；账户标签由阶段内交易标签聚合，仅是异常代理。尤其该图测试集已知样本可能高度偏向异常，AP必须与表中异常占比一起解读，不能推广到所有未知账户。',
 '', 'EB-GAD在校准图和测试图独立转导拟合，选中统计量和分数尺度可能不同；因此其跨图固定阈值F1仅描述本适配管线，不宜据此评价原论文方法能力，图内排序主要看AP/ROC-AUC。Audit-HCL使用离线四快照窗口，并未验证逐笔实时预警或原始设备/所有权图的结论。训练预算不是原论文200 epochs。按时间留出允许账户重复；每阶段快照时长可能不同。',
 '', 'POT与99%分位阈值都由无标签校准分数确定。AP/ROC-AUC只衡量排序；F1还取决于阈值、分数漂移和类别比例。大量并列分数时，99%分位阈值和 `>=` 规则不保证只有1%报警。完整Precision、Recall、FPR和混淆矩阵见CSV，不以Accuracy或单个F1代替评价。','',
 '核验从保存分数与独立测试标签重新计算AP、梯形PR-AUC、ROC-AUC、混淆矩阵、Precision/Recall/F1/FPR，并核验阈值公式、分位数和分数SHA256。神经/经典模型重载检查在读取测试标签前进行。私有模型/分数保留在CQ归档与本地实验目录，原始交易和账户ID不额外复制到公开结果目录。','']
(args.output/'README_CN.md').write_text('\n'.join(lines))
print(json.dumps({'runs':len(records),'operating_points':len(rows),'summary_rows':len(summary)}))
