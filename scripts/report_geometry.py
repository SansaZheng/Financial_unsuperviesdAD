"""Verified prespecified primaries, all ablations, separate blind source results."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from report_recent_zero import readcsv,savecsv,METRICS,DATASETS,fmt,ms,winner,NAMES
from finad_geometry.models import TASKS,HUGE_VARIANTS,FIELD_VARIANTS
ROOT=Path(__file__).resolve().parents[1]
NAMES.update({f'{f}_repair_{v}':f.upper()+'上轮-'+v for f,vs in dict(huge=['legacy','role','role_long','role_noself'],himvh=['legacy','f00','m','cm','m_flat','cm_flat'],sdfmm=['legacy','ge']).items() for v in vs})
NAMES.update({f'huge_roleweight_{v}':'HUGE上轮-'+v for v in ['legacy','role','dual']});NAMES['himvh_roleweight_cm_tail']='HIMVH上轮-cm-tail'
NAMES.update({f'huge_geometry_{v}':'HUGE-'+v for v in HUGE_VARIANTS});NAMES.update({f'himvh_geometry_{b}_{v}':'HIMVH-'+b+'-'+v for b in ['m','cm'] for v in FIELD_VARIANTS})
for m in ['hbos','lof','vae','ebgad_release_zero','audit_hcl_zero_adapted']:NAMES['baseline_geometry_'+m]=NAMES[m]
DATASETS['cardsim']='CardSim独立模拟世界'

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--update-main',action='store_true');a=p.parse_args();proof=json.loads((a.runs/'verification.json').read_text());assert proof['status']=='passed' and proof['configurations']==231;a.output.mkdir(parents=True,exist_ok=True)
 for old,new in [('records.json','per_seed.json'),('run_manifests.json','run_manifests.json'),('verification.json','verification.json'),('weights.json','weights.json')]:shutil.copyfile(a.runs/old,a.output/new)
 rec=json.loads((a.runs/'records.json').read_text());man=json.loads((a.runs/'run_manifests.json').read_text());rows=[]
 for r in rec:
  for op,val in r['metrics']['operating_points'].items():
   x={k:r[k] for k in ['track','dataset','family','variant','method','seed']};x.update(operating_point=op,status='ok' if val else 'failed',seconds_shared_group=r['seconds']);x.update({k:v for k,v in r['metrics'].items() if k!='operating_points'});x.update({k:val[k] if val else None for k in ['precision','recall','f1','fpr','tp','fp','fn','tn','alarm_fraction']});rows.append(x)
 assert len(rows)==462;savecsv(a.output/'metrics.csv',rows);summary=[]
 for f,d,v,op in sorted({(x['family'],x['dataset'],x['variant'],x['operating_point']) for x in rows}):
  g=[x for x in rows if (x['family'],x['dataset'],x['variant'],x['operating_point'])==(f,d,v,op)];assert len(g)==3;x={k:g[0][k] for k in ['track','dataset','family','variant','method','operating_point']};x.update(seeds=3,threshold_successes=sum(r['status']=='ok' for r in g),source_commit=man['validation' if f=='baseline' else f]['git_commit'],source_path='finad_geometry/experiment.py' if f!='himvh' else 'finad_geometry/score_himvh.py',source_kind='independent_generated_source' if d=='cardsim' else 'retrospective_original_cohort')
  for m in METRICS:
   vals=[r[m] for r in g if r[m] is not None];x[m+'_mean']=float(np.mean(vals)) if len(vals)==3 else None;x[m+'_std']=float(np.std(vals,ddof=1)) if len(vals)==3 else None
  for k in ['evaluated','unknown_excluded','positives','normal','prevalence']:assert len({r[k] for r in g})==1;x[k]=g[0][k]
  summary.append(x)
 assert len(summary)==154;savecsv(a.output/'summary.csv',summary);sx={(r['family'],r['dataset'],r['variant'],r['operating_point']):r for r in summary}
 previous=readcsv(ROOT/'results/roleweight_20261117/combined_transaction_summary.csv')+readcsv(ROOT/'results/roleweight_20261117/combined_account_graph_summary.csv')
 combined=[r for r in previous if r['method']!='huge_roleweight_legacy']+[r for r in summary if (r['family']=='huge' and (r['variant']=='fusion' or (r['variant']=='legacy_embed' and r['dataset']!='cardsim'))) or (r['family']=='himvh' and r['variant']=='cm_skill_tail') or r['family']=='baseline'];assert len(combined)==400
 for track in ['tabular','graph']:savecsv(a.output/('combined_transaction_summary.csv' if track=='tabular' else 'combined_account_graph_summary.csv'),[r for r in combined if r['track']==track])
 index={(r['family'],r['dataset'],r['variant'],r['seed'],r['operating_point']):r for r in rows};effects=[]
 for d in TASKS:
  pairs=[('huge','固定原教师：学生相同','decoupled_embed','dual_embed'),('huge','固定原教师：压缩学生','role_decoupled_embed','decoupled_embed'),('huge','原教师融合增益','fusion','decoupled_embed'),('huge','融合后教师训练作用','fusion','native_fusion'),('huge','融合相对直接HALO','fusion','teacher_only'),('huge','新主方案对原HUGE','fusion','legacy_embed')]
  pairs += [('himvh',b+'：尾尺度校准',b+'_tail_flat',b+'_flat') for b in ['m','cm']]+[('himvh',b+'：skill对维数先验',b+'_skill_tail',b+'_tail_flat') for b in ['m','cm']]+[('himvh',b+'：去维数先验',b+'_skill_equal_tail',b+'_skill_tail') for b in ['m','cm']]+[('himvh',b+'：去权重下限',b+'_skill_hard_tail',b+'_skill_tail') for b in ['m','cm']]+[('himvh',b+'：尾证据对原误差',b+'_skill_tail',b+'_skill_raw') for b in ['m','cm']]+[('himvh','skill下完整对抽样上下文','cm_skill_tail','m_skill_tail'),('himvh','新主方案对旧字段均值','cm_skill_tail','cm_field')]
  for f,label,new,old in pairs:
   for op in ['pot','quantile']:
    row=dict(family=f,dataset=d,comparison=label,new=new,old=old,operating_point=op,seeds=3)
    for m in METRICS:
     vals=[index[f,d,new,s,op][m]-index[f,d,old,s,op][m] for s in [42,43,44] if index[f,d,new,s,op][m] is not None and index[f,d,old,s,op][m] is not None];row[m+'_delta_mean']=float(np.mean(vals)) if len(vals)==3 else None;row[m+'_delta_std']=float(np.std(vals,ddof=1)) if len(vals)==3 else None
    effects.append(row)
 savecsv(a.output/'paired_effects.csv',effects);best=[]
 for track,tasks in [('tabular',list(DATASETS)),('graph',['banksim','s_ffsd','cardsim'])]:
  for d in tasks:
   for m,op in [('average_precision','pot'),('roc_auc','pot'),('f1','pot'),('f1','quantile')]:
    for r in winner(combined,track,d,m,op):best.append(dict(track=track,dataset=d,metric=m,operating_point=op,method=r['method'],mean=r[m+'_mean'],std=r[m+'_std'],independent_source=d=='cardsim'))
 savecsv(a.output/'best_methods.csv',best)
 text=['# HALO教师几何与字段重构增益：中文实验报告','','本轮预登记231配置、462条操作点、154条三种子消融汇总；54次新拟合（36 HUGE、6 HIMVH、12经典／Audit），另3个EB种子重复为训练免费。HIMVH复用12个旧检查点，加上6个新CardSim模型；字段评分共144配置，不是训练了144个网络。主版本固定为 **HUGE-fusion** 和 **HIMVH-cm-skill-tail**。全部消融与负结果保留。','','## 原任务与独立验证必须分开','','原BankSim／S-FFSD划分、原101个输入文件和完整测试样本不变；旧测试已经参与方法设计，所以原任务是回顾性开发。新CardSim在固定方法／模拟参数后才生成，是此前没有参与实验的独立模拟世界，80107笔完整交易，时间累计计数60/20/20划分。**这是独立合成来源的协议验证，不是真实银行外部验证，也不是旧银行模型直接零样本迁移。** 各方法都只在新世界的训练／校准分区拟合，测试不定权、不定阈值。','','[生成前固定协议](../../docs/geometry_protocol_CN.md) · [新数据封存与26个输入摘要](../../docs/geometry_cardsim_archive.json) · [独立核验](verification.json)。','','## 固定主方案结果','','所有数值为百分数，均值±样本标准差；AP看排序，F1和FPR看实际阈值效果。两种阈值均不用标签。原阈值POT q=.85/risk=.01/min20，失败N/A；q99独立报告。','','| 验证范围 | 轨道／任务 | 方法 | 原始对照AP | 新主方案AP | AP差值（百分点） | 新POT F1 | 新q99 F1 |','|---|---|---|---:|---:|---:|---:|---:|']
 findings=[]
 for f,primary,anchor in [('huge','fusion','legacy_embed'),('himvh','cm_skill_tail','cm_field')]:
  for d in TASKS:
   old=sx[f,d,anchor,'pot'];new=sx[f,d,primary,'pot'];q=sx[f,d,primary,'quantile'];delta=new['average_precision_mean']-old['average_precision_mean'];text.append('| '+('独立模拟验证' if d=='cardsim' else '旧任务回顾性')+' | '+new['track']+'／'+DATASETS[d]+' | '+f.upper()+' | '+' | '.join([ms(old,'average_precision'),ms(new,'average_precision'),f'{100*delta:+.2f}',ms(new,'f1'),ms(q,'f1')])+' |')
   findings.append(dict(family=f,dataset=d,independent=d=='cardsim',anchor=anchor,AP_delta=delta,AP_first=any(r['method']==f+'_geometry_'+primary for r in winner(combined,new['track'],d,'average_precision','pot')),POT_F1_first=any(r['method']==f+'_geometry_'+primary for r in winner(combined,new['track'],d,'f1','pot')),q99_F1_first=any(r['method']==f+'_geometry_'+primary for r in winner(combined,new['track'],d,'f1','quantile'))))
 (a.output/'findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
 text+=['','原始对照是本轮匹配实现的HUGE-legacy-embed与相同冻结HIM骨干的字段均值；不只与上一轮失败的dual或tail相比。旧任务完整主表保留上一轮role、dual及cm-tail。最佳方法仅在固定候选池中比较，不代表文献SOTA或显著性。','','![主方案、直接教师与对照](comparison.png)','','## 方法改动与边界','','**HUGE：教师与学生分开。** 原40维公共训练尺度的HALO保留作教师；学生仍可接受48维双通道。原排序／蒸馏／正则损失、300更新、隐藏128和原生图边不变。另比较16维压缩学生。这样可以检查，上轮对齐压缩的失败是否来自教师几何本身改变，而不只来自学生丢失方向信息。训练教师在CPU固定计算，所有新对照相同，旧CUDA运行不保证逐位一致；本轮配对对照使用同源环境。','','主评分fusion把训练后负邻居余弦与原始HALO各自按无标签校准上尾概率转换为-log(p)，再等权平均。它引入测试阶段原始图几何，**不是作者原始HUGE评分**。teacher_only是直接原HALO、无需训练的消融，必须与fusion比较，避免把原始几何的效果错误归功于训练。含教师评分使用完整阶段图，与EB-GAD等相同是离线分区内信息，不是严格逐笔因果报警。','','**HIMVH：用无标签可预测性定权。** 对每字段，训练8192目标的逐坐标中位数构成常量预测器；在同一校准样本上计算常量MSE b和冻结模型MSE e。skill=clip(1-mean(e)/max(mean(b),1e-8),0,1)。主权重正比字段宽度×(0.25+0.75skill)，再聚合经验尾证据。下限防止某字段因没有重构增益而被彻底删除；宽度先验和下限都预先固定，不按数据集调。校准器的学习无需正常／欺诈标签。','','field/flat/tail/tail_flat为既有评分对照；skill_raw、skill_equal_tail、skill_hard_tail分别拆出尾尺度、字段维数先验和权重下限的作用。m/cm比较抽样与完整训练邻居，原8192损失目标不变。可预测性不等于欺诈相关性，因此必须看skill_tail对tail_flat的配对效果；若没有收益，不能把整个提升归因于新权重。','','## 所有消融']
 for f,vs in [('huge',HUGE_VARIANTS),('himvh',[b+'_'+v for b in ['m','cm'] for v in FIELD_VARIANTS])]:
  for d in TASKS:
   text+=['','### '+f.upper()+'／'+DATASETS[d]+('（独立验证）' if d=='cardsim' else '（旧任务）'),'','| 配置 | AP ↑ | ROC ↑ | POT F1 ↑ | q99 F1 ↑ | POT FPR ↓ | q99 FPR ↓ |','|---|---:|---:|---:|---:|---:|---:|']
   for v in vs:
    r=sx[f,d,v,'pot'];q=sx[f,d,v,'quantile'];label='**'+v+'（固定主）**' if (f,v) in [('huge','fusion'),('himvh','cm_skill_tail')] else v;text.append('| '+label+' | '+' | '.join([ms(r,'average_precision'),ms(r,'roc_auc'),ms(r,'f1'),ms(q,'f1'),ms(r,'fpr'),ms(q,'fpr')])+' |')
 text+=['','## 配对效应与原因分析','','每个种子先做差，再求均值±样本SD，单位为百分点。正值对AP/F1有利，对FPR表示误报增多；三种子不自动构成显著性证据。所有权重见[逐字段技能与权重](weights.json)。','','| 范围／任务 | 方法 | 对照改变 | AP差值 | POT F1差值 |','|---|---|---|---:|---:|']
 for e in effects:
  if e['operating_point']=='pot':text.append('| '+('独立／' if e['dataset']=='cardsim' else '旧／')+DATASETS[e['dataset']]+' | '+e['family'].upper()+' | '+e['comparison']+' | '+' | '.join('N/A' if e[m+'_delta_mean'] is None else f'{100*e[m+"_delta_mean"]:+.2f} ± {100*e[m+"_delta_std"]:.2f}' for m in ['average_precision','f1'])+' |')
 if (a.output/'interpretation_CN.md').exists():text+=['',(a.output/'interpretation_CN.md').read_text()]
 text+=['','## 与现有最佳方法及完整旧方法池比较','','旧9个交易任务及2个账户图任务沿用此前核验成绩；本轮原适用任务新增固定主方案，以本轮legacy锚点替换上一轮同名原HUGE，其他旧结果不覆盖。新CardSim重新运行旧BankSim/S-FFSD的AP和两种F1获胜方法并集：交易HBOS、LOF、VAE，账户EB-GAD发布版与Audit-HCL零标签端口。新数据没有借用旧数据分数。**主榜仅有预先固定版本，其他变体只作消融，不能将某数据集最高消融替换为主方法。** 共400条主榜汇总；CardSim只有这些预登记对照，并非所有经典方法的穷尽评测。']
 for track,tasks in [('tabular',list(DATASETS)),('graph',['banksim','s_ffsd','cardsim'])]:
  text+=['','### '+('交易／行情点轨道' if track=='tabular' else '付款账户图轨道')]
  for d in tasks:
   text+=['','#### '+DATASETS[d]+('（独立来源验证）' if d=='cardsim' else ''),'','| 方法 | AP ↑ | ROC ↑ | POT F1 ↑ | q99 F1 ↑ | POT FPR ↓ | q99 FPR ↓ |','|---|---:|---:|---:|---:|---:|---:|']
   for r in [x for x in combined if (x['track'],x['dataset'],x['operating_point'])==(track,d,'pot')]:
    q=next(x for x in combined if (x['track'],x['dataset'],x['method'],x['operating_point'])==(track,d,r['method'],'quantile'));text.append('| '+NAMES[r['method']]+' | '+' | '.join([ms(r,'average_precision'),ms(r,'roc_auc'),ms(r,'f1'),ms(q,'f1'),ms(r,'fpr'),ms(q,'fpr')])+' |')
 text+=['','## 各任务最佳方法','','| 范围 | 轨道／任务 | AP最高 | POT F1最高 | q99 F1最高 |','|---|---|---|---|---|']
 for track,tasks in [('tabular',list(DATASETS)),('graph',['banksim','s_ffsd','cardsim'])]:
  for d in tasks:
   cells=[]
   for m,op in [('average_precision','pot'),('f1','pot'),('f1','quantile')]:
    win=winner(combined,track,d,m,op);cells.append('／'.join(NAMES[r['method']] for r in win)+' '+fmt(win[0][m+'_mean']))
   text.append('| '+('独立模拟验证' if d=='cardsim' else '历史／回顾性')+' | '+track+'／'+DATASETS[d]+' | '+' | '.join(cells)+' |')
 text+=['','## 常数参考、信息预算与局限','','| 轨道／任务 | 异常占比＝恒定分数AP | 全报警F1 | 全报警FPR |','|---|---:|---:|---:|']
 for track,d in [('tabular','banksim'),('tabular','s_ffsd'),('graph','banksim'),('graph','s_ffsd'),('tabular','cardsim'),('graph','cardsim')]:
  r=next(r for r in combined if (r['track'],r['dataset'])==(track,d));pr=r['prevalence'];text.append('| '+track+'／'+DATASETS[d]+' | '+fmt(pr)+' | '+fmt(2*pr/(1+pr))+' | 100.00 |')
 text+=['','账户标签表示阶段内是否发生过欺诈，不等于该账户统计上罕见。高异常占比时全报警F1可能很高，但误报率100%；校准风险1%并不保证测试误报1%。排序改善也不保证POT或q99 F1改善。POT与校准CDF共用样本，没有给出有限样本误报保证；分数饱和、并列和阶段漂移都须结合操作点解读。','','HUGE在完整训练图拟合，保存8192参考索引不是图损失限制；HIMVH固定8192目标，但cm使用完整训练邻居。训练无标签源中可能已有欺诈，未按正常标签清洗。图和原生超图使用各阶段完整上下文，没有跨分区边，仍允许同阶段未来邻居；构图字段可能携带遮蔽类别的结构信息。','','CardSim官方模型以公开支付调查数据校准，但本轮欺诈标签由模拟器生成，不能代表真实银行处置确认；只有一个新模拟世界、一个固定生成种子和90天窗口，3个训练种子不是3个独立数据来源。参数在生成前固定，生成后未再按结果修正。所有新世界的测试数值在整轮实验完成后统一分析；旧两任务仍有历史测试重用。后续独立真实业务验证和额外模拟世界未执行，不伪造泛化证据。','','## 复核、来源与复现','','[231条逐种子配置](per_seed.json) · [462条操作点](metrics.csv) · [154条消融汇总](summary.csv) · [配对效应](paired_effects.csv) · [最佳方法](best_methods.csv) · [独立核验](verification.json) · [源与环境](run_manifests.json) · [执行和原始归档](execution.json)。共享组训练的seconds不应按评分配置直接求和；资源时间仅按CQ作业和唯一拟合组解释。','','CardSim：[官方来源](https://www.federalreserve.gov/econres/feds/cardsim-a-bayesian-simulator-for-payment-card-fraud-detection-research.htm)，官方ZIP及公共输入已封存于仓库；26个新输入在data/derived/geometry_20261117_cardsim.tar.gz。旧输入及12个冻结父模型沿用前报告归档，新模型和逐点误差从execution.json所列CQ归档恢复。新增校准器和逐点分数保留私有工作区，公开哈希和完整指标，避免重复提交大量检查点。','','```bash','python -m finad_geometry.experiment --family huge --device cuda --data INPUTS/tabular --graphs INPUTS/graphs --output HUGE','python -m finad_geometry.experiment --family validation --device cuda --data INPUTS/tabular --graphs INPUTS/graphs --output VALIDATION','python -m finad_geometry.score_himvh --old-parents OLD_HIM --new-parents VALIDATION --data INPUTS/tabular --graphs INPUTS/graphs --output HIM_SCORES','PYTHONPATH=. python scripts/verify_geometry.py --huge HUGE --validation VALIDATION --himvh HIM_SCORES --old-parents OLD_HIM --inputs INPUTS --raw-validation raw_transactions.csv.gz --old-AE OLD_AE --output VERIFIED','PYTHONPATH=. python scripts/report_geometry.py --runs VERIFIED --output results/geometry_20261117 --update-main','PYTHONPATH=. python scripts/interpret_geometry.py --results results/geometry_20261117','PYTHONPATH=. python scripts/report_geometry.py --runs VERIFIED --output results/geometry_20261117 --update-main','PYTHONPATH=. python scripts/plot_geometry.py --results results/geometry_20261117','```','','## Plain English','','We freeze the original HALO teacher geometry independently of the student representation and evaluate direct-teacher and fixed fusion controls. HIMVH weights use unlabeled reconstruction skill relative to a training-only constant predictor, with width prior and fixed weight floor ablations. The old cohorts are unchanged. A previously unused CardSim world was generated after freezing the design, and the same best-method baseline set was rerun on its own training/calibration/test split. This is independent synthetic-source protocol validation, not real-bank external validation or zero-shot transfer. All prescribed variants and negative outcomes are retained; no per-dataset test-selected primary is substituted.','']
 (a.output/'README_CN.md').write_text('\n'.join(text))
 if a.update_main:
  p=ROOT/'docs/financial_ad_comparison_CN.md';t=p.read_text();mark='\n## 15. HALO教师几何、字段重构增益与独立验证'
  if mark in t:t=t.split(mark)[0]
  notice='**最新教师／权重及独立验证：** [HUGE和HIMVH：231配置与CardSim独立模拟验证中文报告](../results/geometry_20261117/README_CN.md)，见第15节。\n\n'
  if notice not in t:t=t.replace('# 金融无监督异常检测综合对比报告\n\n','# 金融无监督异常检测综合对比报告\n\n'+notice,1)
  t+=mark+'（2026-11-17）\n\n保持旧101输入／划分／测试／评价；固定原HALO教师和学生表示分离、无标签字段重构增益权重。生成前固定新CardSim世界，80107交易、26封存文件；新数据各方法同分区训练／校准后评价，属于独立合成来源验证，不是真实外部银行验证。231配置、462操作点、54新拟合；所有消融保留。\n\n[完整中文报告、旧／新验证分表、最佳方法与局限](../results/geometry_20261117/README_CN.md)。\n';p.write_text(t)
 print('REPORT231 configs /462 operations /154 ablations /400 primary summaries')
if __name__=='__main__':main()
