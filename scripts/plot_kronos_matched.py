"""Scientific comparison heatmaps from the verified combined aggregate CSV."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
with (a.results/'combined_transaction_summary.csv').open() as f:rows=list(csv.DictReader(f))
tasks=['ulb','banksim','s_ffsd','sapienza_5S','sapienza_15S','sapienza_25S','sd_stock','sd_crypto','sd_synth']
methods=['iforest','lof','knn','ocsvm','pca','hbos','gmm','ae','vae','deep_svdd','cae_if_core','kronos_potratio_matched']
names=['IF','LOF','kNN','OCSVM','PCA','HBOS','GMM','AE','VAE','Deep SVDD','CAE-IF','KronosAD-M']
fig,axes=plt.subplots(2,1,figsize=(14,10),constrained_layout=True)
for ax,metric,title in zip(axes,['average_precision','f1'],['Average Precision (%)','F1 at the fixed POT operating point (%)']):
    values=np.full((len(tasks),len(methods)),np.nan)
    for i,d in enumerate(tasks):
        for j,m in enumerate(methods):
            r=next(r for r in rows if (r['dataset'],r['method'],r['operating_point'])==(d,m,'pot'))
            values[i,j]=float(r[metric+'_mean'])*100 if r[metric+'_mean'] else np.nan
    # Color encodes within-task relative performance; printed cells retain actual percentages.
    relative=np.zeros_like(values)
    for i,row in enumerate(values):
        lo,hi=np.nanmin(row),np.nanmax(row)
        relative[i]=np.where(np.isfinite(row),(row-lo)/(hi-lo) if hi>lo else 0,np.nan)
    ax.imshow(np.ma.masked_invalid(relative),cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for i in range(len(tasks)):
        for j in range(len(methods)):
            v=values[i,j];label='N/A' if np.isnan(v) else f'{v:.2f}'
            ax.text(j,i,label,ha='center',va='center',fontsize=9,color='white' if relative[i,j]>.65 else '#18212b',fontweight='bold' if j==len(methods)-1 else 'normal')
    ax.add_patch(Rectangle((len(methods)-1.5,-.5),1,len(tasks),fill=False,edgecolor='#d56a00',linewidth=2))
    ax.set_xticks(range(len(methods)),names,rotation=25,ha='right');ax.set_yticks(range(len(tasks)),tasks)
    ax.set_title(title,loc='left',fontsize=13)
    ax.set_xticks(np.arange(-.5,len(methods),1),minor=True);ax.set_yticks(np.arange(-.5,len(tasks),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.6);ax.tick_params(which='minor',bottom=False,left=False)
fig.suptitle('Matched financial transaction / point evaluation | 3 seeds (42, 43, 44)\nColor compares methods within each row; compare printed percentages across panels.',fontsize=13)
for suffix in ['png','pdf']:fig.savefig(a.results/f'comparison.{suffix}',dpi=180)
plt.close(fig)
print('Saved verified-metric comparison PNG and PDF')
