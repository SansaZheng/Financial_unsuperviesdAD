"""Join published aggregates, verifying matched splits/budgets; no raw rows."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def compare(old,new):
    old,new=Path(old),Path(new)
    old_runs=json.loads((old/'runs.json').read_text());new_runs=json.loads((new/'runs.json').read_text())
    lookup={(r['dataset'],r['seed']):r for r in old_runs}
    for r in new_runs:
        baseline=lookup[r['dataset'],r['seed']]
        for name in ['data_manifest_sha256','train_cap','calibration_cap','steps','risk','initial_quantile','batch_size']:
            if r['run_config'][name]!=baseline['run_config'][name]:raise ValueError(f'Mismatched {name}')
        if r['split_counts']!=baseline['split_counts']:raise ValueError('Different data split')
        if r['metrics']['evaluated']!=baseline['metrics']['evaluated']:raise ValueError('Different evaluated test rows')
    frames=[]
    for directory,group in [(old,'v1_classic'),(new,'2026_extensions')]:
        frame=pd.read_csv(directory/'metrics.csv');frame['publication_group']=group
        frame['test_feature_access']=np.where(frame.method.eq('rttad2026'),'unlabelled_test_pool_transductive','train_only_inductive')
        frames.append(frame)
    joined=pd.concat(frames,ignore_index=True);joined.to_csv(new/'combined_metrics.csv',index=False)
    combined=pd.concat([pd.read_csv(old/'summary.csv').assign(publication_group='v1_classic'),
                        pd.read_csv(new/'summary.csv').assign(publication_group='2026_extensions')],ignore_index=True)
    combined.to_csv(new/'combined_summary.csv',index=False)
    adaptation=[]
    for r in new_runs:
        if r['method']!='rttad2026':continue
        rounds=r['training']['adaptation'];updated=sum(x.get('status')=='updated' for x in rounds)
        final=rounds[-1] if rounds else {}
        adaptation.append(dict(dataset=r['dataset'],seed=r['seed'],test_pool_rows=r['training']['adaptation_pool_rows'],
            updated_rounds=updated,attempted_rounds=len(rounds),last_selection=final.get('selection',{}).get('status'),
            last_update=final.get('status'),test_labels_used=False))
    adapt=pd.DataFrame(adaptation).sort_values(['dataset','seed']);adapt.to_csv(new/'adaptation_audit.csv',index=False)
    zero=int(adapt.updated_rounds.eq(0).sum())
    from finad.report import LABELS
    datasets=list(dict.fromkeys(r['dataset'] for r in old_runs));methods=list(LABELS)
    methods=[m for m in methods if m in set(joined.method)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(20,7.5));pot=combined[combined.threshold_policy.eq('pot')]
    for ax,metric,title in zip(axes,['f1_mean','average_precision_mean'],['F1 at fixed POT (%)','Average precision (%)']):
        matrix=pot.pivot(index='dataset',columns='method',values=metric).reindex(index=datasets,columns=methods).to_numpy()*100
        im=ax.imshow(np.ma.masked_invalid(matrix),cmap='Blues',vmin=0,vmax=max(1,np.nanmax(matrix)),aspect='auto')
        ax.set_yticks(range(len(datasets)),datasets)
        ax.set_xticks(range(len(methods)),[LABELS[m]+(' *' if m=='rttad2026' else '') for m in methods],rotation=55,ha='right',fontsize=9)
        ax.set_title(title)
        for i in range(len(datasets)):
            for j in range(len(methods)):
                v=matrix[i,j];ax.text(j,i,'N/A' if np.isnan(v) else f'{v:.1f}',ha='center',va='center',fontsize=7,
                    color='white' if np.isfinite(v) and v>im.norm.vmax*.6 else 'black')
        fig.colorbar(im,ax=ax,fraction=.03,pad=.02)
    fig.suptitle('Frozen v1 baselines and 2026 core adaptations: 3 seeds; compare within each dataset\n* RTTAD uses an unlabelled test-feature pool; information budgets differ',fontsize=12)
    fig.tight_layout();fig.savefig(new/'comparison_to_v1.png',dpi=180);plt.close(fig)
    lines=['# 与原 10 种经典方法的对照','',
        f'保留原 300 次实验，新增 {len(new_runs)} 次，共 {len(old_runs)+len(new_runs)} 次记录／{len(joined)} 套阈值操作点。已核对新旧数据清单哈希、划分、测试对象和共同预算参数一致。原方法不重新拟合、不覆盖原结果。','',
        '**RGLD-ID 为归纳适配；RTTAD-TTA 读取了无标签测试特征进行自适应。后者与原方法的信息预算不同，不能把图表当作同信息预算的公平总排名。** 不做跨数据集混合平均。','',
        f'RTTAD 的 {len(adapt)} 次运行中，有 **{zero} 次实际更新轮数为 0**；详细原因见 [自适应核验](adaptation_audit.csv)。这些运行保留训练后的基础网络评分，不伪称成功自适应、不丢弃失败种子。GammaGMM 失败与 POT 拟合失败是两回事，分别记录。','',
        '![与原基线的逐任务比较](comparison_to_v1.png)','',
        '[全部逐种子对照](combined_metrics.csv) · [均值与标准差](combined_summary.csv) · [新方法独立结果](README_CN.md) · [方法与适配边界](../../docs/METHODS_2026_CN.md)','',
        '| 任务 | 种子 | 实际更新轮数 | 最后置信选择状态 |','| --- | ---: | ---: | --- |']
    for r in adaptation:lines.append(f'| {r["dataset"]} | {r["seed"]} | {r["updated_rounds"]} | {r["last_selection"]} |')
    (new/'COMPARISON_CN.md').write_text('\n'.join(lines)+'\n')
    print('Matched aggregate comparison:',len(old_runs),'+',len(new_runs),'zero adaptation:',zero)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old',type=Path,default=Path('results/v1'));p.add_argument('--new',type=Path,required=True)
    args=p.parse_args();compare(args.old,args.new)
