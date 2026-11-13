"""Export the OHLCVA K-line pump-and-dump experiment (xu_pump_1h, b0r0_binance_1h).

    python scripts/evaluate_ohlcva.py --runs .local_data/runs_XU .local_data/runs_B0 \
        --cache .local_data/feature_cache --output results/ohlcva_2026

Uses the modern metric export of ``evaluate_modern.py`` unchanged and adds two non-learned
single-feature references computed from the label-blind test features (no fitting, no
threshold): the hourly amount spike against the previous 24 h (``rel_amount_24h``) and the
hourly price range (``log_range``). They show how much a learned detector adds over a rule.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_modern as em  # noqa: E402
from finad.data import Cache  # noqa: E402
from finad.experiment import write_json  # noqa: E402
from finad.modern_metrics import point_metrics, sequence_metrics  # noqa: E402
from finad.report import LABELS  # noqa: E402

REFERENCES = {'rel_amount_24h': 'Rule: amount spike vs past 24 h', 'log_range': 'Rule: hourly high/low range'}
TASKS = {'xu_pump_1h': 'Xu & Livshits: PumpOlymp signals, 921 coin×exchange series (Cryptopia, Yobit, Bittrex, Binance)',
         'b0r0_binance_1h': 'B0R0koko Binance announcements on the 58 Binance series of the same candles'}


def references(cache_root, datasets):
    out = {}
    for d in datasets:
        cache = Cache(cache_root, d)
        names = cache.manifest['features']
        x = np.asarray(cache.x('test')); y = np.asarray(cache.y('test'))
        seq = np.fromfile(cache.path/cache.manifest['sequence_id_file'], dtype=np.int32)
        for feature in REFERENCES:
            score = x[:, names.index(feature)].astype(np.float64)
            point = point_metrics(y, score, {})
            seq_metrics = sequence_metrics(y, score, seq, {}, cache.manifest['tolerance_rows'])
            out[(d, feature)] = {'average_precision': point['average_precision'], 'roc_auc': point['roc_auc'],
                                 'precision_at_n': point['precision_at_n'], 'best_f1_oracle': point['best_f1_oracle'],
                                 **{k: seq_metrics['threshold_free'][k] for k in ('vus_pr', 'vus_roc', 'pate')}}
    return out


def ranked(summary, refs, dataset, policy='pot'):
    s = summary[(summary.dataset == dataset) & (summary.threshold_policy == policy)].set_index('method')
    rows = []
    for m, r in s.iterrows():
        rows.append(dict(name=LABELS[m], kind='method', ap=r.average_precision_mean, ap_sd=r.average_precision_std,
                         roc=r.roc_auc_mean, pn=r.precision_at_n_mean, vus=r.vus_pr_mean, vus_sd=r.vus_pr_std,
                         vroc=r.vus_roc_mean, pate=r.pate_mean, mcc=r.mcc_mean, f1=r.f1_mean, prec=r.precision_mean,
                         rec=r.recall_mean, alarms=r.alarm_fraction_mean, evrec=r.event_recall_tolerance_mean,
                         epprec=r.alarm_episode_precision_tolerance_mean, aff=r.affiliation_f1_mean))
    for feature, label in REFERENCES.items():
        v = refs[(dataset, feature)]
        rows.append(dict(name=label, kind='rule', ap=v['average_precision'], ap_sd=None, roc=v['roc_auc'], pn=v['precision_at_n'],
                         vus=v['vus_pr'], vus_sd=None, vroc=v['vus_roc'], pate=v['pate']))
    return sorted(rows, key=lambda r: -r['vus'])


def pct(v, sd=None, digits=2):
    if v is None or pd.isna(v):
        return '—'
    return f'{v*100:.{digits}f}' + (f' ± {sd*100:.{digits}f}' if sd is not None and not pd.isna(sd) else '')


def write_readme(output, summary, refs, manifests, records, envs, failures):
    lines = ['# OHLCVA K线拉盘砸盘（pump-and-dump）实验', '',
             '在小时级 OHLCVA K线（开、高、低、收、成交量、成交额）上，用 12 种无监督方法、3 个种子（42/43/44）检测拉盘小时。训练、预处理与阈值校准不读取标签；标签只在最终评价打开。'
             f'共 **{len(records)} 次真实运行**，POT 阈值失败 **{len(failures)}** 次。', '',
             '## 数据与任务', '',
             '| 任务 | 说明 | 训练 / 校准 / 测试行数 | 测试正例（拉盘小时） | 测试序列（含拉盘） | 容差 τ |',
             '| --- | --- | --- | ---: | ---: | ---: |']
    for d, text in TASKS.items():
        m = manifests[d]['splits']; seqs = summary[summary.dataset == d].sequences_evaluated.dropna()
        lines.append(f'| `{d}` | {text} | {m["train"]["rows"]:,} / {m["calibration"]["rows"]:,} / {m["test"]["rows"]:,} | '
                     f'{m["test"]["positive"]} | {m["test"]["sequences"]}（{int(seqs.iloc[0]) if len(seqs) else 0}） | ±{manifests[d]["tolerance_rows"]} 小时 |')
    b = manifests['xu_pump_1h']['time_boundaries']
    lines += ['',
              f'- K线来自 Xu & Livshits（USENIX Security 2019）随附的 CryptoCompare 小时数据，2018-05-15 至 2018-10-29 UTC，BTC 计价；CryptoCompare 聚合序列 CCCAGG 与交易所序列重复且无标签，已排除。',
              f'- 按全部交易所小时K线的累计条数做时间顺序 60/20/20 划分：训练 < {b[0][:10]}，校准至 {b[1][:16]}，测试 2018-09-26 至 2018-10-29。两项任务共用同一边界。',
              '- 特征 14 维，逐序列只用当前与更早的K线：对数收益、高低价差、实体、上下影线、跳空、对数成交额，以及相对过去 24/168 小时的成交额、成交量、价差和收益波动的偏离，是否成交。不使用币种、交易所或时间特征。',
              '- 标签：信号所在小时为 1，其余为 0。0 不代表确认正常（未被 PumpOlymp 或 B0R0koko 记录的拉盘可能存在）。',
              '- **Hu et al. 事件（2019–2022）无对应K线，本次无法运行**：需要 Binance 公开K线，当前运行环境无法访问 data.binance.vision。', '',
              '## 主要结果', '']
    for d in TASKS:
        rows = ranked(summary, refs, d)
        lines += [f'### `{d}`', '',
                  '按 VUS-PR 排序；均值 ± 种子标准差。随机排序的 AUC-PR 参照等于正例率 '
                  f'（{manifests[d]["splits"]["test"]["positive"]/manifests[d]["splits"]["test"]["rows"]*100:.4f}%）。规则参照不训练、不调阈值，只给阈值无关指标。', '',
                  '| 方法 | VUS-PR | AUC-PR（合并） | AUC-ROC | Precision@n | VUS-ROC | PATE |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:
            name = f'*{r["name"]}*' if r['kind'] == 'rule' else r['name']
            lines.append(f'| {name} | {pct(r["vus"], r["vus_sd"])} | {pct(r["ap"], r["ap_sd"], 3)} | {pct(r["roc"])} | {pct(r["pn"])} | {pct(r["vroc"])} | {pct(r["pate"])} |')
        lines += ['', f'**固定 POT 阈值（校准集，无标签）**', '',
                  '| 方法 | 报警比例 | Precision | Recall | F1 | MCC×100 | 容差内事件召回 | 报警片段精度（容差内） | Affiliation-F1 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in [r for r in rows if r['kind'] == 'method']:
            lines.append(f'| {r["name"]} | {pct(r["alarms"], digits=3)} | {pct(r["prec"])} | {pct(r["rec"])} | {pct(r["f1"])} | {pct(r["mcc"])} | '
                         f'{pct(r["evrec"])} | {pct(r["epprec"])} | {pct(r["aff"])} |')
        lines.append('')
    lines += ['![AUC-PR、VUS-PR、Affiliation-F1 与容差事件 F1](comparison.png)', '',
              '## 指标', '',
              '合并视角：在全部测试K线上计算 AUC-PR（Average Precision）、AUC-ROC、Precision@n（n＝拉盘小时数）以及固定阈值下的 Precision／Recall／F1／MCC。'
              '序列视角（TSB-AD 协议，NeurIPS 2024）：每个含拉盘的币×交易所测试序列单独计算 VUS-PR／VUS-ROC（滑动窗口 2τ）、PATE（前后缓冲 τ）与 Affiliation-F1 后宏平均；'
              '容差内事件召回为 ±τ 小时内至少一次报警的拉盘比例，报警片段精度为落在某拉盘 ±τ 小时内的报警片段比例。实现与验证见 [`finad/modern_metrics.py`](../../finad/modern_metrics.py)。', '',
              '## 解释限制', '',
              '- `b0r0_binance_1h` 测试期只有 10 个拉盘小时（58 条序列），种子方差与抽样误差很大，只能作补充参考。',
              '- 2018 年的低流动性币种中，大量K线成交量为 0；拉盘小时与“沉寂后首次成交”在特征上相近，这是主要误报来源之一。',
              '- 标签只到小时粒度；分钟级的拉盘与砸盘被合并进同一根K线。',
              '- RTTAD-TTA 在最多 8192 条无标签测试特征上自适应（transductive），其余方法为归纳设定。三种子标准差反映训练／抽样随机性，不是置信区间。', '',
              '## 文件', '',
              '- [metrics.csv](metrics.csv)：逐种子、逐阈值策略的全部指标 · [summary.csv](summary.csv)：均值与标准差 · [references.json](references.json)：规则参照',
              '- [modern_metrics_runs.json](modern_metrics_runs.json) · [runs.json](runs.json) · [splits.json](splits.json) · [environment.json](environment.json) · [verification.json](verification.json)',
              f'- 执行代码提交：`{envs[0]["code_git_commit"]}`；代码指纹：`{envs[0]["code_fingerprint"]}`；{envs[0]["platform"]}，CPU。', '',
              '复现：', '', '```bash',
              'pip install rdata statsmodels==0.15.0 && pip install --no-deps TSB-AD==1.5 PATE==0.1.1',
              'python scripts/prepare_repo_data.py --output .local_data/feature_cache --datasets xu_pump_1h b0r0_binance_1h',
              'python -m finad benchmark --cache .local_data/feature_cache --output .local_data/runs_ohlcva --datasets xu_pump_1h b0r0_binance_1h \\',
              '  --methods iforest lof knn ocsvm pca hbos gmm ae vae deep_svdd rgld2026 rttad2026 --device cpu',
              'python scripts/evaluate_ohlcva.py --runs .local_data/runs_ohlcva --cache .local_data/feature_cache --output results/ohlcva_2026',
              '```']
    (output/'README_CN.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    captured = {}
    em.readme = lambda output, frame, summary, records, datasets, temporal, methods, seeds, manifests, failures, envs: \
        captured.update(output=output, summary=summary, records=records, manifests=manifests, failures=failures, envs=envs)
    em.export(args.runs, args.cache, args.output, args.workers)
    refs = references(args.cache, list(TASKS))
    write_json(args.output/'references.json', [dict(dataset=d, reference=f, **v) for (d, f), v in refs.items()])
    write_readme(captured['output'], captured['summary'], refs, captured['manifests'], captured['records'], captured['envs'], captured['failures'])
    print('README written')
