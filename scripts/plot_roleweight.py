"""Scientific comparison artifact; values from independently verified summary."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_recent_zero import readcsv
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();rows=readcsv(a.results/'summary.csv');ix={(r['family'],r['dataset'],r['variant'],r['operating_point']):r for r in rows}
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for col,d in enumerate(['banksim','s_ffsd']):
 for row,f in enumerate(['huge','himvh']):
  ax=axes[row,col];variants=['legacy','role','aligned','dual'] if f=='huge' else ['cm_field','cm_flat','cm_tail','m_tail'];x=np.arange(len(variants));w=.35
  for off,m,label,color in [(-w/2,'average_precision','AP','#27649d'),(w/2,'f1','POT F1','#d27827')]:
   vals=[ix[f,d,v,'pot'][m+'_mean'] for v in variants];sd=[ix[f,d,v,'pot'][m+'_std'] for v in variants]
   ax.bar(x+off,[100*v if v is not None else np.nan for v in vals],width=w,yerr=[100*v if v is not None else 0 for v in sd],label=label,color=color,capsize=3)
  ax.set_xticks(x,variants,rotation=15);ax.set_ylabel('Percent (mean ± sample SD)');ax.set_title(f.upper()+' / '+('BankSim' if d=='banksim' else 'S-FFSD'));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend();ax.set_ylim(bottom=0)
fig.suptitle('Fixed primary: HUGE dual / HIMVH cm_tail | 3 seeds | unchanged cohorts',fontsize=13)
fig.savefig(a.results/'comparison.png',dpi=180);fig.savefig(a.results/'comparison.pdf');plt.close(fig)
