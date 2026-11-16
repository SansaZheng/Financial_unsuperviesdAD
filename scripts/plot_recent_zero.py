"""Scientific comparison figures from verified aggregate records, no raw labels."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--report', type=Path, required=True)
    a = p.parse_args(); root = Path(__file__).resolve().parents[1]
    assert json.loads((a.report/'verification.json').read_text())['status'] == 'passed'
    def read(path):
        with path.open() as f: rows = list(csv.DictReader(f))
        for r in rows:
            for k in ['average_precision_mean', 'average_precision_std', 'f1_mean', 'f1_std', 'prevalence']:
                r[k] = float(r[k]) if r[k] else None
        return rows
    new = read(a.report/'summary.csv')
    old = read(a.report/'combined_transaction_summary.csv')+read(a.report/'combined_account_graph_summary.csv')
    old = [r for r in old if r['method'] not in ['huge_financial_inductive', 'himvh_zero_reconstruction', 'sdfmm_zero_synthetic']]
    labels = dict(banksim='BankSim', s_ffsd='S-FFSD', sapienza_5S='Sapienza 5S', sapienza_15S='Sapienza 15S', sapienza_25S='Sapienza 25S', sd_stock='SD stock', sd_crypto='SD crypto', sd_synth='SD synthetic')
    short = dict(huge_financial_inductive='HUGE', himvh_zero_reconstruction='HIMVH-Zero', sdfmm_zero_synthetic='SD-FMM-Zero')
    fig = plt.figure(figsize=(14, 10.2), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.5])
    for i, (track, tasks) in enumerate([('tabular', list(labels)), ('graph', ['banksim', 's_ffsd'])]):
        for j, (metric, title) in enumerate([('average_precision', 'AP'), ('f1', 'Original POT F1')]):
            ax = fig.add_subplot(gs[i, j]); names = []
            for position, task in enumerate(tasks):
                row = next(r for r in new if r['track'] == track and r['dataset'] == task and r['operating_point'] == 'pot')
                old_rows = [r for r in old if r['track'] == track and r['dataset'] == task and r['operating_point'] == 'pot' and r[metric+'_mean'] is not None]
                previous = max(old_rows, key=lambda r: r[metric+'_mean'])
                names.append(labels[task]+' / '+short[row['method']])
                for offset, r, color, label in [(-.18, previous, '#8a929e', 'Previous best'), (.18, row, '#2463a8', 'New method')]:
                    value, std = r[metric+'_mean'], r[metric+'_std']
                    if value is None:
                        ax.text(1, position+offset, 'N/A', va='center', fontsize=9)
                        continue
                    ax.barh(position+offset, 100*value, .31, color=color, xerr=100*std,
                            error_kw=dict(elinewidth=1, capsize=2), label=label if position == 0 else None, zorder=3)
                    ax.text(min(100*value+100*std+1, 96), position+offset, f'{100*value:.1f}', va='center', fontsize=8)
                if metric == 'average_precision':
                    ax.scatter(100*row['prevalence'], position, marker='|', s=95, c='#ac352a', zorder=5,
                               label='Constant-score AP' if position == 0 else None)
            ax.set_yticks(range(len(tasks)), names, fontsize=9)
            ax.invert_yaxis(); ax.set_xlim(0, 104); ax.set_xticks(range(0, 101, 20))
            ax.set_xlabel('Percent (%)'); ax.grid(axis='x', color='#e4e6e8', zorder=0)
            ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(('Transaction / market point' if track == 'tabular' else 'Account graph')+' | '+title, loc='left', weight='bold')
            if i == 0:
                ax.legend(loc='lower right', frameon=True, fontsize=8)
    fig.suptitle('Matched label-free financial evaluation: 3 seeds, mean ± sample SD\nPrevious best is selected separately for each task and metric; tracks are never pooled.', fontsize=12)
    fig.savefig(a.report/'comparison.png', dpi=170)
    fig.savefig(a.report/'comparison.pdf')
    plt.close(fig)
    print('PLOT EXPORTED from verified tables')


if __name__ == '__main__':
    main()
