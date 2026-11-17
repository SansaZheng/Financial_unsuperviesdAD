"""Publication figures from verified summaries; never train or select variants."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_recent_zero import readcsv
from finad_repair.experiment import PRIMARY


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    a = p.parse_args()
    rows = readcsv(a.results/'summary.csv')
    tasks = [('huge','banksim'), ('huge','s_ffsd'),
             ('himvh','banksim'), ('himvh','s_ffsd')]
    tasks += [('sdfmm', d) for d in ['sapienza_5S','sapienza_15S','sapienza_25S',
                                    'sd_stock','sd_crypto','sd_synth']]
    labels = ['HUGE / BankSim accounts', 'HUGE / S-FFSD accounts',
              'HIMVH / BankSim transactions', 'HIMVH / S-FFSD transactions',
              'SD-FMM / Sapienza 5S', 'SD-FMM / Sapienza 15S',
              'SD-FMM / Sapienza 25S', 'SD-FMM / stock',
              'SD-FMM / crypto', 'SD-FMM / synthetic']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True,
                             layout='constrained')
    for ax, metric, title in zip(axes, ['average_precision', 'f1'],
                                  ['Average precision', 'F1 at original POT']):
        for variant, offset, color, label in [('legacy', .14, '#666666', 'Original port, rerun'),
                                             ('primary', -.14, '#007A87', 'Prespecified repair')]:
            selected = [next(r for r in rows if r['family']==f and r['dataset']==d
                             and r['variant']==('legacy' if variant=='legacy' else PRIMARY[f])
                             and r['operating_point']=='pot') for f,d in tasks]
            means = np.array([100*r[metric+'_mean'] if r[metric+'_mean'] is not None else np.nan for r in selected])
            sd = np.array([100*r[metric+'_std'] if r[metric+'_std'] is not None else np.nan for r in selected])
            ax.errorbar(means, np.arange(len(tasks))+offset, xerr=sd,
                        fmt='o', color=color, markersize=5, capsize=2,
                        elinewidth=1, label=label)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Percent; mean ± sample SD across 3 seeds')
        ax.set_xlim(-5, 105)
        ax.set_xticks(range(0, 101, 20))
        ax.grid(axis='x', alpha=.2)
        for y in [1.5,3.5]: ax.axhline(y, color='#cccccc', linewidth=.7)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(np.arange(len(labels)), labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[1].legend(loc='lower right', fontsize=9)
    fig.suptitle('Matched cohorts: fixed primary repairs versus original zero-label ports', fontsize=13)
    fig.savefig(a.results/'comparison.png', dpi=180)
    fig.savefig(a.results/'comparison.pdf')
    plt.close(fig)
    print('Exported comparison.png and comparison.pdf')


if __name__ == '__main__':
    main()
