"""Verified primary comparisons, full prespecified ablations and paired deltas."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from report_recent_zero import readcsv,savecsv,METRICS,DATASETS,fmt,ms,winner,NAMES as OLD_NAMES
from finad_repair.experiment import PRIMARY,VARIANTS

ROOT=Path(__file__).resolve().parents[1]
TITLES=dict(huge='HUGE',himvh='HIMVH',sdfmm='SD-FMM')
EFFECT_NAMES=dict(primary_vs_legacy='主修正－旧复跑',role_alignment='角色对齐及缩放－旧复跑',
                 budget1200_vs300='1200更新－300更新',no_self='去自环－含自环评分',
                 new_solver_vs_legacy='新公共求解器－旧复跑',C_given_M='一致遮蔽下：完整邻居－抽样重建',
                 M_given_C='完整邻居下：一致遮蔽－旧遮蔽',C_average='恢复原生上下文平均效应',
                 M_average='一致遮蔽任务平均效应',C_M_interaction='上下文×遮蔽交互',
                 field_average_vs_flat='字段等权－按维数加权',separate_rng_vs_legacy='独立RNG公共对照－旧复跑',
                 E_given_G='约束生成器下：末点BCE－全位置BCE',G_given_E='末点BCE下：约束生成器－旧生成器',
                 E_average='末点损失平均效应',G_average='约束生成器平均效应',
                 E_G_interaction='损失×生成器交互',contrastive_branch='有对比分支－无对比分支')


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--update-main',action='store_true');a=p.parse_args()
    proof=json.loads((a.runs/'verification.json').read_text());assert proof['status']=='passed' and proof['runs']==174
    records=json.loads((a.runs/'records.json').read_text());manifests=json.loads((a.runs/'run_manifests.json').read_text());a.output.mkdir(parents=True,exist_ok=True)
    for n in ['records.json','run_manifests.json','verification.json']:shutil.copyfile(a.runs/n,a.output/('per_seed.json' if n=='records.json' else n))
    if (a.runs/'shard_provenance.json').exists():shutil.copyfile(a.runs/'shard_provenance.json',a.output/'shard_provenance.json')
    rows=[]
    for r in records:
        for op,v in r['metrics']['operating_points'].items():
            row={k:r[k] for k in ['track','dataset','family','variant','method','seed']};row.update(operating_point=op,status='ok' if v else 'failed',seconds=r['seconds'],peak_cuda_gb=r['peak_cuda_gb'])
            row.update({k:value for k,value in r['metrics'].items() if k!='operating_points'})
            row.update({k:v[k] if v else None for k in ['precision','recall','f1','fpr','tp','fp','fn','tn','alarm_fraction']});rows.append(row)
    assert len(rows)==348;savecsv(a.output/'metrics.csv',rows)
    summary=[]
    for family,task,variant,op in sorted({(r['family'],r['dataset'],r['variant'],r['operating_point']) for r in rows}):
        g=[r for r in rows if (r['family'],r['dataset'],r['variant'],r['operating_point'])==(family,task,variant,op)];assert len(g)==3
        row={k:g[0][k] for k in ['track','dataset','family','variant','method','operating_point']};row.update(seeds=3,threshold_successes=sum(x['status']=='ok' for x in g),source_commit=manifests[family]['git_commit'],source_path='finad_repair/experiment.py',source_kind='fresh_scientific_run')
        for m in METRICS:
            vals=[r[m] for r in g if r[m] is not None]
            row[m+'_mean']=float(np.mean(vals)) if len(vals)==3 else None;row[m+'_std']=float(np.std(vals,ddof=1)) if len(vals)==3 else None
        for k in ['evaluated','unknown_excluded','positives','normal','prevalence']:
            assert len({r[k] for r in g})==1;row[k]=g[0][k]
        summary.append(row)
    assert len(summary)==116;savecsv(a.output/'summary.csv',summary)
    previous=readcsv(ROOT/'results/recent_zero_20261117/combined_transaction_summary.csv')+readcsv(ROOT/'results/recent_zero_20261117/combined_account_graph_summary.csv')
    old_ids=['huge_financial_inductive','himvh_zero_reconstruction','sdfmm_zero_synthetic']
    base=[r for r in previous if r['method'] not in old_ids];assert len(base)==330
    selected=[r for r in summary if r['variant'] in ['legacy',PRIMARY[r['family']]]]
    combined=base+selected;assert len(combined)==370
    for r in selected:
        prior=next(x for x in previous if x['track']==r['track'] and x['dataset']==r['dataset'])
        for k in ['evaluated','unknown_excluded','positives','normal','prevalence']:assert r[k]==prior[k]
    savecsv(a.output/'combined_transaction_summary.csv',[r for r in combined if r['track']=='tabular'])
    savecsv(a.output/'combined_account_graph_summary.csv',[r for r in combined if r['track']=='graph'])
    names=OLD_NAMES.copy()
    for family in VARIANTS:
        for variant in VARIANTS[family]:names[f'{family}_repair_{variant}']=TITLES[family]+'-'+variant
        names[f'{family}_repair_legacy']=TITLES[family]+'旧适配（复跑）'
        names[f'{family}_repair_{PRIMARY[family]}']=TITLES[family]+'修正版'
    sx={(r['family'],r['dataset'],r['variant'],r['operating_point']):r for r in summary}
    index={(r['family'],r['dataset'],r['variant'],r['seed'],r['operating_point']):r for r in rows}
    effects=[]
    def effect(f,d,label,terms,op):
        record=dict(family=f,dataset=d,comparison=label,operating_point=op,seeds=3)
        for m in METRICS:
            values=[]
            for seed in [42,43,44]:
                vals=[index[f,d,v,seed,op][m] for v,_ in terms]
                if all(x is not None for x in vals):values.append(sum(index[f,d,v,seed,op][m]*w for v,w in terms))
            record[m+'_delta_mean']=float(np.mean(values)) if len(values)==3 else None;record[m+'_delta_std']=float(np.std(values,ddof=1)) if len(values)==3 else None
        effects.append(record)
    for f,d in sorted({(r['family'],r['dataset']) for r in summary}):
        comparisons=[('primary_vs_legacy',[(PRIMARY[f],1),('legacy',-1)])]
        if f=='huge':comparisons += [('role_alignment',[('role',1),('legacy',-1)]),('budget1200_vs300',[('role_long',1),('role',-1)]),('no_self',[('role_noself',1),('role',-1)])]
        elif f=='himvh':comparisons += [('new_solver_vs_legacy',[('f00',1),('legacy',-1)]),('C_given_M',[('cm',1),('m',-1)]),('M_given_C',[('cm',1),('c',-1)]),('C_average',[('c',.5),('f00',-.5),('cm',.5),('m',-.5)]),('M_average',[('m',.5),('f00',-.5),('cm',.5),('c',-.5)]),('C_M_interaction',[('cm',1),('c',-1),('m',-1),('f00',1)]),('field_average_vs_flat',[('cm',1),('cm_flat',-1)])]
        else:comparisons += [('separate_rng_vs_legacy',[('f00',1),('legacy',-1)]),('E_given_G',[('ge',1),('g',-1)]),('G_given_E',[('ge',1),('e',-1)]),('E_average',[('e',.5),('f00',-.5),('ge',.5),('g',-.5)]),('G_average',[('g',.5),('f00',-.5),('ge',.5),('e',-.5)]),('E_G_interaction',[('ge',1),('e',-1),('g',-1),('f00',1)]),('contrastive_branch',[('ge',1),('ge_nocontrast',-1)])]
        for label,terms in comparisons:
            for op in ['pot','quantile']:effect(f,d,label,terms,op)
    savecsv(a.output/'paired_effects.csv',effects)
    best=[]
    for track,tasks in [('tabular',DATASETS),('graph',['banksim','s_ffsd'])]:
        for d in tasks:
            for m,op in [('average_precision','pot'),('roc_auc','pot'),('f1','pot'),('f1','quantile')]:
                for r in winner(combined,track,d,m,op):best.append(dict(track=track,dataset=d,metric=m,operating_point=op,method=r['method'],mean=r[m+'_mean'],std=r[m+'_std']))
    savecsv(a.output/'best_methods.csv',best)
    old_records=json.loads((ROOT/'results/recent_zero_20261117/per_seed.json').read_text());legacy_audit=[]
    map_id=dict(huge='huge_financial_inductive',himvh='himvh_zero_reconstruction',sdfmm='sdfmm_zero_synthetic')
    for r in records:
        if r['variant']!='legacy':continue
        old=next(x for x in old_records if (x['dataset'],x['method'],x['seed'])==(r['dataset'],map_id[r['family']],r['seed']))
        legacy_audit.append(dict(family=r['family'],dataset=r['dataset'],seed=r['seed'],AP_old=old['metrics']['average_precision'],AP_rerun=r['metrics']['average_precision'],AP_delta=r['metrics']['average_precision']-old['metrics']['average_precision'],score_file_identical=r['score_sha256']==old['score_sha256']))
    savecsv(a.output/'legacy_reproduction.csv',legacy_audit)
    resources=[{**{k:r[k] for k in ['family','dataset','variant','seed','seconds','peak_cuda_gb']},**r['model_info']} for r in records]
    savecsv(a.output/'training_resources.csv',resources)
    findings=[]
    text=['# 三方法实现审计、修正与消融：中文报告','','日期：2026-11-17。**最终分析集174组配置、348条操作点，156次实际拟合；旧法复跑30次已包含其中。** 所有预先登记配置都保留；主版本预先固定为HUGE-role、HIMVH-cm、SD-FMM-ge，不按数据集选择最优消融。真实训练／校准标签完全不参与学习或阈值。','','## 结论先看','','| 轨道／任务 | 修正方法 | 旧复跑AP | 修正AP | AP差值（百分点） | 旧原POT F1 | 修正原POT F1 |','|---|---|---:|---:|---:|---:|---:|']
    for f,d in sorted({(r['family'],r['dataset']) for r in summary}):
        old=sx[f,d,'legacy','pot'];new=sx[f,d,PRIMARY[f],'pot']
        text.append(f'| {old["track"]}／{DATASETS[d]} | {TITLES[f]} | {ms(old,"average_precision")} | {ms(new,"average_precision")} | {100*(new["average_precision_mean"]-old["average_precision_mean"]):+.2f} | {ms(old,"f1")} | {ms(new,"f1")} |')
    for f in VARIANTS:
        eligible=[r for r in summary if r['family']==f and r['variant']==PRIMARY[f] and r['operating_point']=='pot']
        find=dict(family=f,tasks=len(eligible),AP_improved=sum(r['average_precision_mean']>sx[f,r['dataset'],'legacy','pot']['average_precision_mean'] for r in eligible))
        for key,m,op in [('AP_first','average_precision','pot'),('POT_F1_first','f1','pot'),('q99_F1_first','f1','quantile')]:
            find[key]=sum(any(x['method']==r['method'] for x in winner(combined,r['track'],r['dataset'],m,op)) for r in eligible)
        findings.append(find)
        text += ['',f'**{TITLES[f]}修正版：** {find["tasks"]}个适用任务中AP较旧复跑提高{find["AP_improved"]}个；在共同方法池中，AP均值最高{find["AP_first"]}个、原POT F1最高{find["POT_F1_first"]}个、q99 F1最高{find["q99_F1_first"]}个。这些是三种子均值排名，不是显著性结论。']
    write_json=lambda name,obj:(a.output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    write_json('findings.json',findings)
    text += ['','数值均为百分数，均值±样本标准差；AP差值为百分点。图与交易单位不同，不合并平均。**本轮依赖已查看的旧测试结果提出修正，是回顾性开发，不是新的独立验证。**','','![修正前后AP与POT-F1](comparison.png)','','## 查到了什么，不能据此声称什么','','1. **没有发现评分方向写反或标签／样本错位。** 原评分方向固定；旧代码不变，另做当轮旧法复跑。原源码公式、样本索引、输入摘要、模型重载和保存分数的指标重算都有检查。低于随机参照也不根据测试标签翻转分数。',
             '2. **HUGE主要有角色语义适配风险。** 把付款方出账和收款方入账放在不同列，再让角色one-hot参与HALO，会把正常类型差异算入异配性。修正以对应方向的计数／均值／std／最大值对齐，按角色仅用训练数据标准化；原生边及付款账户评价集合不变。它改变了方法内部预处理，不声称与旧40维最终输入完全相同。角色改进、训练预算和自环评分分别有对照。',
             '3. **HIMVH的训练图和遮蔽任务有具体缺陷。** 抽样后重建窗口改变了实际邻域／时间跨度；新的C用完整训练上下文、原8192损失目标及每视图精确3层ego块。干净HCDP又能携带被遮蔽值，新的M先遮蔽再计算辅助量，并采用一致的遮蔽字段预测。这里是自监督目标通路，不是真实标签泄漏。图拓扑仍可编码用于构图的类别键，不能声称所有被遮蔽类别的结构信息也被移除。',
             '4. **SD-FMM旧自产任务与金融特征及末点评价不一致。** 旧逐列扰动破坏时钟单位圆和派生关系；旧全位置BCE把单个末点梯度稀释160倍。新G保留时钟、共同改变金融字段，SD序列从发布值重算衍生量；新E只优化真实评价末点。产生的仍是人工干预分布，不保证等于真实欺诈。无标签源窗口可能本来就异常，自产0不等于已知正常。',
             '5. **阈值与排序分开解释。** AP不依赖阈值，因此低AP不能只归因于POT。校准尾部风险1%也不等于测试FPR必为1%；分布漂移、校准污染及高异常占比都可能影响操作点。保持相同POT/q99，分别报告AP与F1/FPR，不能根据测试标签改风险或翻转分数。',
             '6. **修复不等于复刻原有标签论文或保证SOTA。** HIMVH仍是独立零标签重构适配，SD-FMM仍无真实异常片段或标签验证，HUGE仍为金融归纳迁移。固定预算、信息范围、图假设和源污染仍有限制；具体效果看完整消融，不能只据反例把所有低AP归因于某一项。','','诊断为训练特征／合成数据实验：[训练审计](train_only_audit.json)。完整定义及字段映射：[固定协议](../../docs/recent_repair_protocol_CN.md)。']
    text += ['','## 全部预登记消融','','f00是新公共求解器／采样RNG对照；legacy是原代码复跑。HIMVH公共求解器800×128与旧200×512具有相同损失目标次数，但更新次数、BN批大小和遮蔽轨迹不同；SD新因子分离了采样与生成RNG。先看这些对照，再解释C/M或E/G。m_flat/cm_flat、role_noself不额外训练。']
    for f,d in sorted({(r['family'],r['dataset']) for r in summary}):
        text += ['',f'### {TITLES[f]}／{DATASETS[d]}','','| 配置 | AP ↑ | ROC-AUC ↑ | 原POT F1 ↑ | q99 F1 ↑ | 原POT FPR ↓ |','|---|---:|---:|---:|---:|---:|']
        for v in VARIANTS[f]:
            r=sx[f,d,v,'pot'];q=sx[f,d,v,'quantile'];label=('**'+v+'（主）**') if v==PRIMARY[f] else v
            text.append('| '+label+' | '+' | '.join([ms(r,'average_precision'),ms(r,'roc_auc'),ms(r,'f1'),ms(q,'f1'),ms(r,'fpr')])+' |')
    text += ['','## 配对消融效应','','正值表示该指标提高，FPR除外。每个效应先逐种子做差再求均值；2×2平均效应和交互项按固定代数式计算，不能把多个成组修正拆成未测量的独立原因。POT缺失时不只平均有利种子。完整[配对效应CSV](paired_effects.csv)。','','| 方法／任务 | 对照 | AP差值（百分点） | 原POT F1差值（百分点） |','|---|---|---:|---:|']
    for e in effects:
        if e['operating_point']!='pot':continue
        text.append('| '+TITLES[e['family']]+'／'+DATASETS[e['dataset']]+' | '+EFFECT_NAMES[e['comparison']]+' | '+fmt(e['average_precision_delta_mean'])+' | '+fmt(e['f1_delta_mean'])+' |')
    text += ['', '## 用通俗的话解读消融', '', '以下是逐种子配对的AP变化（百分点，均值±样本SD），正值表示提高。它们描述本设置内的受控对照；三种子不足以自动支持显著性结论。']
    ex={(e['family'],e['dataset'],e['comparison']):e for e in effects if e['operating_point']=='pot'}
    def delta(f,d,key):
        e=ex[f,d,key]
        return f"{100*e['average_precision_delta_mean']:+.2f} ± {100*e['average_precision_delta_std']:.2f}"
    for f,d in sorted({(r['family'],r['dataset']) for r in summary}):
        if f=='huge':
            explanation=f"角色对齐加分角色缩放相对旧法变化{delta(f,d,'role_alignment')}；把训练更新增加到1200相对300变化{delta(f,d,'budget1200_vs300')}；仅去掉评分自环变化{delta(f,d,'no_self')}。因此角色适配、训练预算与自环作用不能混为一谈；对齐与缩放在本轮仍是成组修正。"
        elif f=='himvh':
            explanation=f"新求解器本身相对旧法变化{delta(f,d,'new_solver_vs_legacy')}；在一致遮蔽条件下，恢复原生训练邻居变化{delta(f,d,'C_given_M')}；在完整训练上下文下，一致遮蔽任务变化{delta(f,d,'M_given_C')}；只把字段评分从按维数加权改为等权变化{delta(f,d,'field_average_vs_flat')}。不能把主版本的全部收益或退步都归因于辅助分支；M同时改了遮蔽、训练目标和评分任务。"
        else:
            explanation=f"仅分离采样与生成随机数的公共对照相对旧法变化{delta(f,d,'separate_rng_vs_legacy')}；在约束生成器下改用末点损失变化{delta(f,d,'E_given_G')}；在末点损失下改用约束生成器变化{delta(f,d,'G_given_E')}；启用对比分支相对关闭变化{delta(f,d,'contrastive_branch')}。破坏约束的反例已经证实，但它在真实任务上造成多少损失须看本数据集对照，不能从反例直接推出所有AP差值。"
        text += ['', '**'+TITLES[f]+'／'+DATASETS[d]+'：** '+explanation]
    text += ['', '仍未单独证实的原因包括：有限训练预算、真实异常与自产异常不相似、类别哈希丢失信息、图／时间聚合压缩了欺诈证据以及无标签源污染。HUGE有额外预算消融，其余原因没有独立实验就只作假设。监督HIMVH的判别目标和SD-FMM的异常示例被移除后，不再能直接套用原论文高分；新的自监督目标可能学到可重构性或人工干预痕迹，而非真实欺诈含义。提高AP也不保证固定POT操作点的F1同步提高。']
    text += ['','## 与现有方法的完整同协议对比','','主榜只加入预先指定role/cm/ge，并以当轮legacy替换上轮三个旧适配值；其他消融不参与挑选主榜最优。其余经典、Kronos、CAE-IF、EB-GAD、Audit-HCL及RGLD/RTTAD的已核验同协议成绩复用，不声称本轮重训所有方法。旧轮标准差已统一为样本SD。原生关系、历史长度、预训练和RTTAD测试自适应的信息预算差异仍披露。']
    for track,tasks in [('tabular',DATASETS),('graph',['banksim','s_ffsd'])]:
        text += ['', '### '+('交易／行情点轨道' if track=='tabular' else '付款账户图轨道')]
        for d in tasks:
            group=[r for r in combined if r['track']==track and r['dataset']==d and r['operating_point']=='pot']
            text += ['',f'#### {DATASETS[d]}','','| 方法 | AP ↑ | ROC-AUC ↑ | 原POT F1 ↑ | q99 F1 ↑ | 原POT FPR ↓ |','|---|---:|---:|---:|---:|---:|']
            for r in group:
                q=next(x for x in combined if (x['track'],x['dataset'],x['method'],x['operating_point'])==(track,d,r['method'],'quantile'))
                text.append('| '+names[r['method']]+' | '+' | '.join([ms(r,'average_precision'),ms(r,'roc_auc'),ms(r,'f1'),ms(q,'f1'),ms(r,'fpr')])+' |')
    text += ['','## 各任务最佳方法','','| 轨道 | 数据集 | AP最高 | 原POT F1最高 | q99 F1最高 |','|---|---|---|---|---|']
    for track,tasks in [('tabular',DATASETS),('graph',['banksim','s_ffsd'])]:
        for d in tasks:
            cells=[]
            for metric,op in [('average_precision','pot'),('f1','pot'),('f1','quantile')]:
                win=winner(combined,track,d,metric,op);cells.append('／'.join(names[x['method']] for x in win)+' '+fmt(win[0][metric+'_mean']))
            text.append('| '+track+' | '+DATASETS[d]+' | '+' | '.join(cells)+' |')
    text += ['','HIMVH按各阶段完整关系构建离线超图，允许同阶段邻居上下文；没有跨训练／校准／测试分区的边。它不是严格逐笔因果在线检测。', '', '训练信息量并非完全相同：HUGE所有配置都在完整训练图上拟合，每步排序抽512图节点，正则与对齐作用于完整图；保存的8192付款账户参考索引不是其损失节点限制。角色缩放使用完整训练图。HIMVH-C使用完整训练邻居，但只对原8192交易目标施加损失。其余方法的历史、图关系、预训练和TTA预算见历史报告。', '', '9任务来自5来源，3种Sapienza粒度并非3份独立数据。S-FFSD账户已知异常占87.48%，常数AP也达87.48%；交易已知异常占38.87%。这些条件与常见“正常占绝大多数”的无监督假设不同；高AP须结合ROC/F1/FPR解读。ULB无适用原生关系／资产序列，本轮三个修正方法仍不强行应用。','','## 复核、环境与文件','','- [174条逐种子配置](per_seed.json)、[348条操作点评价](metrics.csv)、[116条消融汇总](summary.csv)、[最佳方法](best_methods.csv)。','- [交易主表CSV](combined_transaction_summary.csv)、[账户主表CSV](combined_account_graph_summary.csv)、[旧法复跑偏差](legacy_reproduction.csv)。','- [训练参数和资源](training_resources.csv)、[独立核验](verification.json)、[三个家族的冻结源与环境](run_manifests.json)、[执行与归档](execution.json)。','- 真实训练／校准标签不用；测试标签在模型、分数、阈值保存后打开。POT q=.85/risk=.01/min20和独立q99，未经点调整，AP和梯形PR-AUC分开保存；失败N/A。所有156次拟合来自一个科学提交，不拼接挑选有利结果。',
             '- 复核使用完整CQ归档与原AE基线；`PYTHONPATH=. python scripts/verify_repair.py --jobs /path/to/three-families --old-runs /path/to/old/results --output /path/to/merged`。报告：`PYTHONPATH=. python scripts/report_repair.py --runs /path/to/merged --output results/recent_repair_20261117`。模型和逐点分数保留在归档，不将大型权重混入Git。','','## 执行中的数值门禁问题', '', '第一批HIMVH在首个模型的重载检查停止，尚未产生该配置的分数／阈值／指标。两次GPU前向约8.7e-6的数值差异触发旧1e-6绝对容差。最终门禁逐张量检查参数及缓冲完全相同，再比较CPU副本前向逐位相等；没有放宽阈值、改变训练或评分。新增回归测试后，全部三家族在新同一提交下从头运行；初批部分输出未查看、未混入最终结果。平台无法即时强制终止，取消请求后旧作业可能继续完成；这些归档仍保留，取消状态不表示其检测指标失败。156次是最终分析集的拟合数，不包含未纳入的旧批次计算。完整过程见[执行记录](execution.json)与[原始失败日志](superseded_reload_gate_failure.txt)。这是一项核验环节修正，不能用来解释旧金融AP低。', '', '## 原始方法来源', '', '[HUGE论文与核心公式](https://arxiv.org/html/2502.13308v3) · [HUGE作者代码](https://github.com/CampanulaBells/HUGE-GAD) · [HIMVH论文](https://arxiv.org/html/2601.11073v3) · [SD-FMM作者训练代码固定提交](https://github.com/ys-Dai/SD-FMM/blob/0d1a2158ee849fdc31a127bd049e622174f5c893/train.py)。作者提交与端口逐项边界见[原固定协议](../../docs/recent_zero_protocol_CN.md)。本轮修改不是作者原始有标签成绩的复刻。', '', '## Plain English','','We audited the existing zero-label ports and identified role-semantic mismatch, altered training hypergraph neighborhoods, clean auxiliary paths carrying masked feature values, and incoherent synthetic financial features with endpoint-loss dilution. We froze primary repairs and full controlled ablations before reading newly fitted test results. All174 configurations use the previous cohorts; the original ports were rerun. Improvements and negative effects are both reported. These retrospective financial adaptations do not reproduce the papers’ original supervised claims, establish global SOTA, or prove each hypothesized cause in isolation.','']
    (a.output/'README_CN.md').write_text('\n'.join(text))
    if a.update_main:
        path=ROOT/'docs/financial_ad_comparison_CN.md';current=path.read_text();marker='\n## 13. 三方法审计修正与消融'
        if marker in current:current=current.split(marker)[0]
        notice='**最新修正与消融：** [三方法审计修正、174组配置与完整中文报告](../results/recent_repair_20261117/README_CN.md)。新主榜见第13节；前文保留历史结果。\n\n'
        if notice not in current:current=current.replace('# 金融无监督异常检测综合对比报告\n\n','# 金融无监督异常检测综合对比报告\n\n'+notice,1)
        current=current.replace('**2026-11-17最新扩展：**','**同日上轮扩展（历史记录）：**').replace('最新最佳方法见第12节及该报告；','该轮结果见第12节及该报告；')
        current+='\n\n## 13. 三方法审计修正与消融（2026-11-17）\n\n确认适配风险与目标缺陷后，原划分重跑30条旧配置，并完成主版本及全部消融，共174条配置、348条操作点、156次拟合。全部预先登记配置均保留；采用已有测试诊断设计，属于回顾性开发。主版本固定为HUGE-role、HIMVH-cm、SD-FMM-ge，未按各数据集挑选消融。\n\n[完整对比、配对效应、最佳方法与局限](../results/recent_repair_20261117/README_CN.md)。\n'
        for f in findings:current+=f'\n- {TITLES[f["family"]]}：AP较旧复跑改善{f["AP_improved"]}/{f["tasks"]}任务；AP均值最高{f["AP_first"]}、原POT F1最高{f["POT_F1_first"]}、q99 F1最高{f["q99_F1_first"]}。\n'
        path.write_text(current)
    print('REPORT174 configurations /348 operating records /370 primary-comparison summaries exported')


if __name__=='__main__':main()
