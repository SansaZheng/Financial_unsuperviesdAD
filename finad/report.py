"""Publish aggregate evidence only; refuse missing/duplicate experiments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .experiment import write_json

LABELS = {'iforest':'Isolation Forest','lof':'LOF','knn':'kNN','ocsvm':'OCSVM','pca':'PCA-SPE',
          'hbos':'HBOS','gmm':'GMM','ae':'AE','vae':'VAE','deep_svdd':'Deep SVDD'}
OPERATING = ['precision','recall','f1','fpr','alarm_fraction']
RANKING = ['average_precision','pr_auc_trapezoid','roc_auc']


def collect(runs):
    runs=Path(runs)
    env=json.loads((runs/'environment.json').read_text())
    plan=env['configuration']
    expected={(d,m,s) for d in plan['datasets'] for m in plan['methods'] for s in plan['seeds']}
    records=[json.loads(p.read_text()) for p in sorted(runs.glob('*/*/result.json'))]
    found={(r['dataset'],r['method'],r['seed']) for r in records}
    if found!=expected or len(records)!=len(expected) or any(r['status']!='complete' for r in records):
        raise ValueError(f'Incomplete/duplicate sweep: missing={expected-found}, unexpected={found-expected}, count={len(records)}')
    return env,records


def export(runs,output):
    env,records=collect(runs)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    # Do not publish workstation usernames/absolute data paths.
    env['configuration']['cache']='/path/to/private/feature_cache'
    env['configuration']['output']='.local_data/runs_v1'
    write_json(output/'environment.json',env)
    write_json(output/'runs.json',records)
    flat=[]
    for r in records:
        for policy,value in r['thresholds'].items():
            row={'dataset':r['dataset'],'method':r['method'],'seed':r['seed'],'threshold_policy':policy,
                 'threshold_status':value['status'],'threshold':value['threshold'],
                 'threshold_failure':value.get('reason'), 'train_rows_used':r['train_rows_used'],
                 'calibration_rows_used':r['calibration_rows_used'],'total_seconds':r['total_seconds']}
            row.update({k:v for k,v in r['metrics'].items() if k!='operating_points'})
            operating=r['metrics']['operating_points'][policy]
            row.update(operating if operating is not None else {k:None for k in OPERATING+['tp','fp','fn','tn']})
            flat.append(row)
    frame=pd.DataFrame(flat).sort_values(['dataset','method','seed','threshold_policy'])
    frame.to_csv(output/'metrics.csv',index=False)
    summaries=[]
    for (d,m,policy),g in frame.groupby(['dataset','method','threshold_policy'],sort=False):
        all_valid=bool(g.f1.notna().all())
        row=dict(dataset=d,method=m,threshold_policy=policy,seeds=len(g),
                 valid_operating_seeds=int(g.f1.notna().sum()),evaluated_per_seed=int(g.evaluated.iloc[0]),
                 positives_per_seed=int(g.positives.iloc[0]),unknown_excluded_per_seed=int(g.unknown_excluded.iloc[0]),
                 train_rows_used=int(g.train_rows_used.iloc[0]))
        for metric in RANKING+OPERATING:
            eligible=(metric in RANKING or all_valid) and g[metric].notna().all()
            row[metric+'_mean']=float(g[metric].mean()) if eligible else None
            row[metric+'_std']=float(g[metric].std(ddof=0)) if eligible else None
        summaries.append(row)
    summary=pd.DataFrame(summaries);summary.to_csv(output/'summary.csv',index=False)
    plan=env['configuration'];datasets=plan['datasets'];methods=plan['methods']
    pot=summary[summary.threshold_policy.eq('pot')]
    quantile=summary[summary.threshold_policy.eq('quantile')]
    _plot(pot,datasets,methods,output/'comparison.png',len(plan['seeds']))
    failures=[dict(dataset=r['dataset'],method=r['method'],seed=r['seed'],
                   reason=r['thresholds']['pot']['reason']) for r in records if r['thresholds']['pot']['status']!='ok']
    write_json(output/'pot_failures.json',failures)
    def table(data,metric):
        matrix=data.pivot(index='dataset',columns='method',values=metric+'_mean').reindex(index=datasets,columns=methods)
        lines=['| 数据 | '+' | '.join(LABELS[m] for m in methods)+' |',
               '| --- | '+' | '.join(['---:']*len(methods))+' |']
        for d,row in matrix.iterrows():
            lines.append('| '+d+' | '+' | '.join('N/A' if pd.isna(v) else f'{v*100:.2f}' for v in row)+' |')
        return lines
    groups={('sd_fmm' if d in ('sd_stock','sd_crypto') else 'sapienza' if d.startswith('sapienza_') else d) for d in datasets}
    counts=['| 任务 | 测试总行数 | 已知标签评价数 | 正例数 | 已知标签正例率 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for d in datasets:
        r=next(r for r in records if r['dataset']==d)
        e=r['metrics'];prevalence=e['prevalence']
        counts.append(f'| {d} | {e["evaluated"]+e["unknown_excluded"]:,} | {e["evaluated"]:,} | {e["positives"]:,} | '+
                      ('N/A' if prevalence is None else f'{prevalence*100:.4f}%')+' |')
    lines=['# 经典金融无监督异常检测：实验结果 v1','',
           f'完成 **{len(datasets)} 个任务、{len(methods)} 种方法、{len(plan["seeds"])} 个种子，共 {len(records)} 次真实运行**，覆盖 {len(groups)} 组源数据。SD-FMM 的股票和加密货币为同一来源的两个任务。本轮是经典算法核心在金融特征上的复现，不是原论文数据／网络／数值成绩的逐项复刻。','',
           f'训练每种子统一最多 {plan["train_cap"]:,} 条、校准最多 {plan["calibration_cap"]:,} 条；**测试分区全量、不平衡类别、不做 point adjustment**。标签只在最终评价打开。沿用旧数据划分，但训练上限与统一截断评分不同于此前 Kronos 实验，不能直接混排两轮成绩。','',
           '## 留出测试组成','',*counts,'',
           'AP 的随机排序参照水平与正例率有关；不能把正例率差异当作模型性能差异。所有方法在各任务上的测试对象相同。','',
           '## 主结果：固定 POT 的 F1（%）','',
           f'每格为全部 {len(plan["seeds"])} 个种子的均值；任一种子 POT 失败时该格明确 N/A，不仅平均成功种子。当前共有 **{len(failures)} 次 POT 拟合失败**，原因见 [POT 失败记录](pot_failures.json)。','',
           *table(pot,'f1'),'','## 排序能力：Average Precision（%）','',
           'AP 使用完整有真值测试子集，与阈值无关。AP 与梯形 PR-AUC 分列在 CSV，不能混称；不同数据正例率／粒度不同，不作混合平均排行榜。','',
           *table(pot,'average_precision'),'','## 预先指定的 99% 校准分位阈值：F1（%）','',
           '这是所有方法都计算的第二套操作点，不是 POT 失败时暗中替换的默认值，也不根据测试结果选择较好的阈值。离散分数的并列值和分布漂移可能导致报警比例明显高于 1%。','',
           *table(quantile,'f1'),'','![POT F1 与 AP 对照](comparison.png)','',
           '## 数据与解释限制','',
           '- Sapienza 为操纵起点块，TSTR 为整段序列，SD-FMM 为点级；粒度不同。TSTR 校准约一半为正，POT 的主要正常假设不成立；SD-FMM 已由发布者全段处理且重用公开 test 文件研究划分，不能声称原论文零样本复现。',
           '- Elliptic 未知标签仅在评价屏蔽，绝不当正常；此处使用原节点特征，不含图边。原聚合特征不能保证实时可用。',
           '- PaySim、TabFormer、AML 为合成。时间留出允许账户跨期，不等于未见账户测试。部分完整交易字段属于事后可用信息，不构成支付授权前拦截验证。',
           '- IEEE-CIS 仅评价官方训练表内部时间留出，不是 Kaggle 官方测试成绩。原始文件、模型、逐行分数和缓存没有上传。',
           '- 全部方法训练混合数据，不按标签剔除异常。OCSVM、Deep SVDD 等单类目标在污染训练下可能明显退化。经典方法的适用条件不能由算法名称保证。',
           '- 所有超参数提前固定，没有标签调参或测试最优阈值。三种子标准差反映训练／抽样随机性，不是置信区间；未作统计显著性结论。','',
           '## 完整指标与复现证据','',
           f'- [{len(records)*2} 套逐种子操作点及完整混淆矩阵](metrics.csv)：Precision、Recall、F1、FPR、AP、梯形 PR-AUC、ROC-AUC、TP/FP/FN/TN、正例率及两套阈值。',
           '- [均值与标准差](summary.csv) · [全部聚合记录、训练损失、阈值和评分 SHA256](runs.json) · [环境与代码指纹](environment.json)。',
           '- [方法来源与适配边界](../../docs/METHODS_CN.md) · [预先固定的协议](../../docs/PROTOCOL_CN.md) · [本地评分／混淆矩阵核验](verification.json)。','',
           f'- 执行代码提交：`{env["code_git_commit"]}`；代码指纹：`{env["code_fingerprint"]}`；适配神经网络设备 `{plan["device"]}`，传统方法／POT 为 CPU。','',
           '公开的是聚合证据；模型、完整逐行分数和预处理保存在本地私有运行目录。原始第三方数据不随本代码的 MIT 许可证重新授权。']
    (output/'README_CN.md').write_text('\n'.join(lines)+'\n')
    print('Exported complete runs:',len(records),'POT failures:',len(failures))


def _plot(summary,datasets,methods,destination,seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(17,7))
    for axis,metric,title in zip(axes,['f1','average_precision'],['F1 at fixed POT (%)','Average precision (%)']):
        matrix=summary.pivot(index='dataset',columns='method',values=metric+'_mean').reindex(index=datasets,columns=methods).to_numpy()*100
        im=axis.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap='Blues',vmin=0,vmax=max(1,np.nanmax(matrix)))
        axis.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=50,ha='right')
        axis.set_yticks(range(len(datasets)),datasets);axis.set_title(title)
        for i in range(len(datasets)):
            for j in range(len(methods)):
                v=matrix[i,j]
                axis.text(j,i,'N/A' if np.isnan(v) else f'{v:.1f}',ha='center',va='center',fontsize=8,
                          color='white' if np.isfinite(v) and v>im.norm.vmax*.6 else 'black')
        fig.colorbar(im,ax=axis,fraction=.035,pad=.02)
    fig.suptitle(f'Classic financial anomaly detectors: {seeds} seeds; label-blind training/calibration\nDifferent label granularities/prevalences; compare methods within each row',fontsize=12)
    fig.tight_layout();fig.savefig(destination,dpi=180);plt.close(fig)
