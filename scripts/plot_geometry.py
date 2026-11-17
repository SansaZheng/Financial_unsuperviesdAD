"""Publication-style figure from verified numerical summaries."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_recent_zero import readcsv
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();rows=readcsv(a.results/'summary.csv');ix={(r['family'],r['dataset'],r['variant'],r['operating_point']):r for r in rows};fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
for col,d in enumerate(['banksim','s_ffsd','cardsim']):
 for row,f in enumerate(['huge','himvh']):
  ax=axes[row,col];vs=['legacy_embed','dual_embed','decoupled_embed','teacher_only','fusion'] if f=='huge' else ['cm_field','cm_tail_flat','cm_skill_tail','m_skill_tail'];labels=['Legacy','Dual','Fixed teacher','HALO only','Fusion'] if f=='huge' else ['Field','Width + tail','Skill + tail','m Skill + tail'];x=np.arange(len(vs));w=.36
  for shift,m,label,color in [(-w/2,'average_precision','AP','#25669c'),(w/2,'f1','POT F1','#cf782f')]:
   r=[ix[f,d,v,'pot'] for v in vs];ax.bar(x+shift,[100*q[m+'_mean'] if q[m+'_mean'] is not None else np.nan for q in r],w,yerr=[100*q[m+'_std'] if q[m+'_std'] is not None else 0 for q in r],color=color,label=label,capsize=3)
  ax.axhline(100*ix[f,d,vs[0],'pot']['prevalence'],color='#555',ls=':',lw=1,label='Constant AP');ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Percent (mean ± sample SD)');ax.set_title(f.upper()+' / '+dict(banksim='BankSim (retrospective)',s_ffsd='S-FFSD (retrospective)',cardsim='CardSim (independent simulation)')[d],fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
fig.suptitle('Fixed primaries: HUGE fusion / HIMVH cm_skill_tail | unchanged old cohorts + unseen synthetic source',fontsize=13);fig.savefig(a.results/'comparison.png',dpi=180);fig.savefig(a.results/'comparison.pdf');plt.close(fig)
