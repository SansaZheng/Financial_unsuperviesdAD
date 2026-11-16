"""Comparison and full-factorial figures from verified published summaries."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
def read(name):
    with (a.results/name).open() as f:return list(csv.DictReader(f))
combined=read('combined_transaction_summary.csv');summary=read('summary.csv')
tasks=['ulb','banksim','s_ffsd','sapienza_5S','sapienza_15S','sapienza_25S','sd_stock','sd_crypto','sd_synth']
methods=['iforest','lof','knn','ocsvm','pca','hbos','gmm','ae','vae','deep_svdd','cae_if_core','kronos_potratio_matched','kronos_stable_nc','train_prior_potratio','persistence_stable_nc']
names=['IF','LOF','kNN','OCSVM','PCA','HBOS','GMM','AE','VAE','Deep SVDD','CAE-IF','Kronos-M','Kronos-S','Train prior','Persistence']
def grid(ax,values,labels,title,focus):
    relative=np.zeros_like(values)
    for i,row in enumerate(values):
        low,high=np.nanmin(row),np.nanmax(row)
        relative[i]=np.where(np.isfinite(row),(row-low)/(high-low) if high>low else 0.,np.nan)
    ax.imshow(np.ma.masked_invalid(relative),cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for i in range(len(tasks)):
        for j in range(len(labels)):
            v=values[i,j];ax.text(j,i,'N/A' if np.isnan(v) else f'{v:.2f}',ha='center',va='center',fontsize=8.5,color='white' if relative[i,j]>.65 else '#18212b',fontweight='bold' if j==focus else 'normal')
    ax.add_patch(Rectangle((focus-.5,-.5),1,len(tasks),fill=False,edgecolor='#da7900',linewidth=2))
    ax.set_xticks(range(len(labels)),labels,rotation=28,ha='right');ax.set_yticks(range(len(tasks)),tasks);ax.set_title(title,loc='left')
    ax.set_xticks(np.arange(-.5,len(labels)),minor=True);ax.set_yticks(np.arange(-.5,len(tasks)),minor=True)
    ax.grid(which='minor',color='white',linewidth=.6);ax.tick_params(which='minor',bottom=False,left=False)
def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(a.results/(name+'.'+ext),dpi=170)
    plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(17,10),constrained_layout=True)
for ax,metric,title in zip(axes,['average_precision','f1'],['AP (%) | identical full test positions','F1 (%) | identical original POT policy, not the new guard']):
    values=np.full((9,15),np.nan)
    for i,d in enumerate(tasks):
        for j,m in enumerate(methods):
            r=next(x for x in combined if (x['dataset'],x['method'],x['operating_point'])==(d,m,'pot'))
            values[i,j]=float(r[metric+'_mean'])*100 if r[metric+'_mean'] else np.nan
    grid(ax,values,names,title,12)
fig.suptitle('Retrospective matched comparison | 3 seeds | orange: stable Kronos score\nColors compare within each row; use printed percentages. Controls are included.',fontsize=12)
save(fig,'comparison')
fig,axes=plt.subplots(2,1,figsize=(14,10),constrained_layout=True)
variants=['legacy','n','c','nc','prior','persist'];values=np.full((9,6),np.nan)
for i,d in enumerate(tasks):
    for j,v in enumerate(variants):
        row=next(x for x in summary if (x['dataset'],x['variant'],x['operating_point'])==(d,v,'pot'))
        values[i,j]=float(row['average_precision_mean'])*100
grid(axes[0],values,['Original','N only','C only','N+C','Train prior','Persistence N+C'],'AP (%) | normalization / cold-start ablation and backbone-free controls',3)
values=np.full((9,8),np.nan);labels=[]
for j,(v,op) in enumerate((v,op) for v in variants[:4] for op in ['pot','guarded_pot']):
    labels.append(f'N{int(v in ["n","nc"])}C{int(v in ["c","nc"])}G{int(op=="guarded_pot")}')
    for i,d in enumerate(tasks):
        row=next(x for x in summary if (x['dataset'],x['variant'],x['operating_point'])==(d,v,op))
        values[i,j]=float(row['f1_mean'])*100 if row['f1_mean'] else np.nan
grid(axes[1],values,labels,'F1 (%) | full 2 x 2 x 2 factorial; guarded policy is a separate operating point',7)
fig.suptitle('N: residual floor | C: cold-start prior mixture | G: calibration-only POT guard\nNo per-dataset configuration selection. Same predictions and full test positions.',fontsize=12)
save(fig,'ablation')
print('Saved matched and complete factorial figures',flush=True)
