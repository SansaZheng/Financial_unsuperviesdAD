"""Re-score completed benchmark runs with the modern metric suite and export aggregate evidence.

    python scripts/evaluate_modern.py --runs .local_data/runs_A .local_data/runs_B \
        --cache .local_data/feature_cache --output results/data_2026

Every run directory must hold a complete sweep (``environment.json`` plan fully realised).
The saved test scores of each run are evaluated once; the per-run result is cached next to
the scores as ``modern_metrics.json`` keyed by the score file digest. Only aggregates,
per-seed metric rows and provenance are written to ``--output``; models and row-level
scores stay private.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finad.data import Cache, sha256  # noqa: E402
from finad.experiment import write_json  # noqa: E402
from finad.modern_metrics import point_metrics, sequence_metrics  # noqa: E402
from finad.report import LABELS, OPERATING, RANKING  # noqa: E402

POINT_FREE = ['average_precision', 'roc_auc', 'pr_auc_trapezoid', 'best_f1_oracle', 'precision_at_n']
POINT_OP = OPERATING+['mcc', 'tp', 'fp', 'fn', 'tn']
SEQ_FREE = ['seq_auc_pr', 'seq_auc_roc', 'vus_pr', 'vus_roc', 'pate']
SEQ_OP = ['seq_f1', 'pa_f1', 'pak_f1_area', 'event_f1', 'range_f1', 'affiliation_precision', 'affiliation_recall',
          'affiliation_f1', 'pate_f1', 'sequences_without_alarm', 'event_recall', 'event_recall_tolerance',
          'alarm_episodes', 'alarm_episode_precision_tolerance', 'event_f1_tolerance']
MEAN_ALWAYS = POINT_FREE+SEQ_FREE


def load_scores(run):
    result = json.loads((run/'result.json').read_text())
    path = run/result['test_scores_file']
    if sha256(path) != result['test_scores_sha256']:
        raise ValueError(f'Score digest mismatch in {run}')
    if path.suffix == '.npz':
        with np.load(path) as z:
            return result, z['scores']
    return result, np.load(path)


def evaluate_run(job):
    run, cache_root = Path(job[0]), Path(job[1])
    result, scores = load_scores(run)
    cached = run/'modern_metrics.json'
    if cached.exists():
        previous = json.loads(cached.read_text())
        if previous['test_scores_sha256'] == result['test_scores_sha256'] and previous.get('version') == 2:
            return previous
    cache = Cache(cache_root, result['dataset'])
    y = np.asarray(cache.y('test'))
    out = {'version': 2, 'dataset': result['dataset'], 'method': result['method'], 'seed': result['seed'],
           'test_scores_sha256': result['test_scores_sha256'], 'thresholds': result['thresholds'],
           'point': point_metrics(y, scores, result['thresholds']), 'sequence': None}
    if cache.manifest.get('temporal'):
        n = cache.manifest['splits']['test']['rows']
        seq = np.fromfile(cache.path/cache.manifest['sequence_id_file'], dtype=np.int32)
        if len(seq) != n:
            raise ValueError('Sequence id file length differs from manifest')
        out['sequence'] = sequence_metrics(y, scores, seq, result['thresholds'], cache.manifest['tolerance_rows'])
    write_json(cached, out)
    print(f"evaluated {result['dataset']} {result['method']} seed={result['seed']}", flush=True)
    return out


def collect(run_dirs):
    envs, records = [], []
    for root in run_dirs:
        root = Path(root)
        env = json.loads((root/'environment.json').read_text()); plan = env['configuration']
        expected = {(d, m, s) for d in plan['datasets'] for m in plan['methods'] for s in plan['seeds']}
        found = {}
        for path in sorted(root.glob('*/*/result.json')):
            r = json.loads(path.read_text())
            if r['status'] != 'complete':
                raise ValueError(f'Incomplete run {path}')
            found[(r['dataset'], r['method'], r['seed'])] = (path.parent, r)
        if set(found) != expected:
            raise ValueError(f'Sweep {root} incomplete: missing={expected-set(found)} unexpected={set(found)-expected}')
        envs.append(env); records.extend(found.values())
    keys = [(r['dataset'], r['method'], r['seed']) for _, r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate runs across sweep directories')
    return envs, records


def flatten(modern, record):
    rows = []
    thresholds = record['thresholds']
    for policy, value in thresholds.items():
        row = {'dataset': modern['dataset'], 'method': modern['method'], 'seed': modern['seed'], 'threshold_policy': policy,
               'threshold_status': value['status'], 'threshold': value['threshold'], 'threshold_failure': value.get('reason'),
               'train_rows_used': record['train_rows_used'], 'calibration_rows_used': record['calibration_rows_used'],
               'total_seconds': record['total_seconds']}
        point = modern['point']
        row.update({k: point[k] for k in ('evaluated', 'unknown_excluded', 'positives', 'normal', 'prevalence')})
        row.update({k: point[k] for k in POINT_FREE})
        op = point['operating_points'][policy]
        row.update(op if op is not None else {k: None for k in POINT_OP})
        seq = modern['sequence']
        if seq is None:
            row.update({'sequences_evaluated': None, 'tolerance_rows': None, **{k: None for k in SEQ_FREE+SEQ_OP}})
        else:
            row.update({'sequences_evaluated': seq['sequences_evaluated'], 'tolerance_rows': seq['tolerance_rows']})
            free = seq['threshold_free'] or {}
            row.update({'seq_auc_pr': free.get('auc_pr'), 'seq_auc_roc': free.get('auc_roc'), 'vus_pr': free.get('vus_pr'),
                        'vus_roc': free.get('vus_roc'), 'pate': free.get('pate')})
            sop = seq['operating_points'][policy]
            if sop is None:
                row.update({k: None for k in SEQ_OP})
            else:
                row.update({'seq_f1': sop['f1'], **{k: sop[k] for k in ('pa_f1', 'pak_f1_area', 'event_f1', 'range_f1',
                            'affiliation_precision', 'affiliation_recall', 'affiliation_f1', 'pate_f1', 'sequences_without_alarm')}})
                pooled = sop['pooled_events']
                row.update({k: pooled[k] for k in ('event_recall', 'event_recall_tolerance', 'alarm_episodes',
                                                    'alarm_episode_precision_tolerance', 'event_f1_tolerance')})
        rows.append(row)
    return rows


def summarise(frame):
    numeric = [c for c in POINT_FREE+POINT_OP+SEQ_FREE+SEQ_OP if c in frame]
    summaries = []
    for (d, m, policy), g in frame.groupby(['dataset', 'method', 'threshold_policy'], sort=False):
        all_valid = bool(g.f1.notna().all())
        row = dict(dataset=d, method=m, threshold_policy=policy, seeds=len(g), valid_operating_seeds=int(g.f1.notna().sum()),
                   evaluated_per_seed=int(g.evaluated.iloc[0]), positives_per_seed=int(g.positives.iloc[0]),
                   unknown_excluded_per_seed=int(g.unknown_excluded.iloc[0]),
                   sequences_evaluated=None if pd.isna(g.sequences_evaluated.iloc[0]) else int(g.sequences_evaluated.iloc[0]),
                   tolerance_rows=None if pd.isna(g.tolerance_rows.iloc[0]) else int(g.tolerance_rows.iloc[0]))
        for metric in numeric:
            eligible = (metric in MEAN_ALWAYS or all_valid) and g[metric].notna().all()
            row[metric+'_mean'] = float(g[metric].mean()) if eligible else None
            row[metric+'_std'] = float(g[metric].std(ddof=0)) if eligible else None
        summaries.append(row)
    return pd.DataFrame(summaries)


def table(summary, datasets, methods, metric, scale=100, fmt='{:.2f}'):
    matrix = summary.pivot(index='dataset', columns='method', values=metric+'_mean').reindex(index=datasets, columns=methods)
    lines = ['| 任务 | '+' | '.join(LABELS[m] for m in methods)+' |', '| --- | '+' | '.join(['---:']*len(methods))+' |']
    for d, row in matrix.iterrows():
        best = row.max()
        cells = []
        for v in row:
            if pd.isna(v):
                cells.append('N/A')
            else:
                text = fmt.format(v*scale)
                cells.append(f'**{text}**' if v == best else text)
        lines.append('| '+d+' | '+' | '.join(cells)+' |')
    return lines


def plot(summary, datasets, temporal, methods, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    # Sequential magnitude: one hue, light -> dark.
    cmap = LinearSegmentedColormap.from_list('blue_ramp', ['#eef3f8', '#a9c3d8', '#4f82a8', '#1f3b57'])
    panels = [('average_precision', 'pot', 'AUC-PR, pooled test split (%)', datasets),
              ('vus_pr', 'pot', 'VUS-PR, per-sequence macro (%)', temporal),
              ('affiliation_f1', 'pot', 'Affiliation-F1 at fixed POT (%)', temporal),
              ('event_f1_tolerance', 'pot', 'Event-F1 within tolerance at POT (%)', temporal)]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for axis, (metric, policy, title, rows) in zip(axes.ravel(), panels):
        data = summary[summary.threshold_policy.eq(policy)]
        matrix = data.pivot(index='dataset', columns='method', values=metric+'_mean').reindex(index=rows, columns=methods).to_numpy()*100
        image = axis.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap=cmap, vmin=0, vmax=100)
        axis.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=45, ha='right', fontsize=9)
        axis.set_yticks(range(len(rows)), rows, fontsize=9); axis.set_title(title, fontsize=11, loc='left')
        for i in range(len(rows)):
            for j in range(len(methods)):
                v = matrix[i, j]
                axis.text(j, i, 'N/A' if np.isnan(v) else f'{v:.1f}', ha='center', va='center', fontsize=8,
                          color='white' if np.isfinite(v) and v > 55 else '#1f1f1f')
        for spine in axis.spines.values():
            spine.set_visible(False)
        fig.colorbar(image, ax=axis, fraction=.03, pad=.02)
    fig.suptitle('Label-blind training/calibration, 3 seeds; compare methods within a row, not across rows', fontsize=12)
    fig.tight_layout(); fig.savefig(destination, dpi=170); plt.close(fig)


def export(run_dirs, cache_root, output, workers):
    envs, records = collect(run_dirs)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    jobs = [(str(path), str(cache_root)) for path, _ in records]
    with Pool(workers) as pool:
        modern = pool.map(evaluate_run, jobs, chunksize=1)
    by_key = {(m['dataset'], m['method'], m['seed']): m for m in modern}
    rows = []
    for path, record in records:
        rows.extend(flatten(by_key[(record['dataset'], record['method'], record['seed'])], record))
    frame = pd.DataFrame(rows).sort_values(['dataset', 'method', 'seed', 'threshold_policy'])
    frame.to_csv(output/'metrics.csv', index=False)
    summary = summarise(frame); summary.to_csv(output/'summary.csv', index=False)
    datasets = [d for env in envs for d in env['configuration']['datasets']]
    methods = envs[0]['configuration']['methods']; seeds = envs[0]['configuration']['seeds']
    for env in envs:
        if env['configuration']['methods'] != methods or env['configuration']['seeds'] != seeds:
            raise ValueError('Sweeps use different method/seed plans')
    manifests = {d: json.loads((Path(cache_root)/d/'manifest.json').read_text()) for d in datasets}
    temporal = [d for d in datasets if manifests[d].get('temporal')]
    # Aggregate evidence only; no workstation paths.
    import importlib.metadata
    for env in envs:
        env['configuration']['cache'] = '/path/to/private/feature_cache'
        env['configuration']['output'] = '.local_data/runs_' + ('_'.join(env['configuration']['datasets']))
        env['evaluation_packages'] = {name: importlib.metadata.version(name) for name in ('TSB-AD', 'PATE', 'statsmodels', 'scikit-learn')}
    write_json(output/'environment.json', envs)
    write_json(output/'runs.json', [r for _, r in records])
    write_json(output/'modern_metrics_runs.json', modern)
    write_json(output/'splits.json', manifests)
    write_json(output/'cache_files_sha256.json', json.loads((Path(cache_root)/'cache_files_sha256.json').read_text()))
    failures = [dict(dataset=r['dataset'], method=r['method'], seed=r['seed'], reason=r['thresholds']['pot']['reason'])
                for _, r in records if r['thresholds']['pot']['status'] != 'ok']
    write_json(output/'pot_failures.json', failures)
    plot(summary, datasets, temporal, methods, output/'comparison.png')
    readme(output, frame, summary, records, datasets, temporal, methods, seeds, manifests, failures, envs)
    print('Exported runs:', len(records), 'POT failures:', len(failures))


def readme(output, frame, summary, records, datasets, temporal, methods, seeds, manifests, failures, envs):
    pot = summary[summary.threshold_policy.eq('pot')]; quantile = summary[summary.threshold_policy.eq('quantile')]
    counts = ['| 任务 | 粒度 | 测试总行数 | 已知标签评价数 | 正例数 | 正例率 | 评价序列数 | 容差 τ（行） |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for d in datasets:
        r = next(r for _, r in records if r['dataset'] == d); e = r['metrics']; m = manifests[d]
        kind = '序列点级' if m.get('temporal') else '交易级'
        seq = frame[frame.dataset.eq(d)].sequences_evaluated.dropna()
        counts.append(f'| {d} | {kind} | {e["evaluated"]+e["unknown_excluded"]:,} | {e["evaluated"]:,} | {e["positives"]:,} | '
                      f'{e["prevalence"]*100:.4f}% | {int(seq.iloc[0]) if len(seq) else "—"} | {m.get("tolerance_rows", "—")} |')
    lines = ['# 仓库 data/ 数据的无监督异常检测实验（现代评价指标）', '',
             f'对本仓库 [`data/raw`](../../data/raw) 中可直接使用的 **{len(datasets)} 个任务** 运行全部 **{len(methods)} 种方法**（10 种经典方法 + RGLD-ID、RTTAD-TTA），'
             f'**{len(seeds)} 个种子（{"/".join(map(str, seeds))}），共 {len(records)} 次真实运行**。训练、预处理和阈值校准不读取标签；标签只在最终评价打开。',
             '',
             '评价指标分两个视角：',
             '',
             '1. **交易级／点级（全量测试集合并）**：AUC-PR（Average Precision）、AUC-ROC、Precision@n（n＝正例数）、固定阈值下的 Precision／Recall／F1／FPR／**MCC**；另给出标签选优的 best-F1 作为上限参照（不是方法的操作点）。',
             '2. **序列视角（TSB-AD 协议，NeurIPS 2024）**：对每条含异常的测试序列单独评价后做宏平均。阈值无关：AUC-PR、AUC-ROC、**VUS-PR／VUS-ROC**（Paparrizos et al., VLDB 2022）、**PATE**（Ghorbani et al., KDD 2024）；固定阈值：标准 F1、PA-F1、**PA%K-F1 面积**（Kim et al., AAAI 2022）、事件级 F1、范围 F1（Tatbul et al., NeurIPS 2018）、**Affiliation P/R/F1**（Huet et al., KDD 2022）、PATE-F1；另有合并的事件召回（事件内／容差内）与报警片段精度。',
             '',
             f'VUS 的滑动窗口取 2τ（两侧缓冲 0..τ），PATE 的前／后缓冲取 τ；τ 在看标签前按数据说明固定（Sapienza：发布者说明标注块最多早于公告 120 s，即 120 s／块长；SD-FMM：160 步窗口的 5%）。VUS、PATE 与 Affiliation 等由官方实现（TSB-AD 1.5、PATE 0.1.1）计算，PA%K、MCC 与合并事件视角为本仓库实现，见 [`finad/modern_metrics.py`](../../finad/modern_metrics.py) 及其测试。',
             '',
             '## 测试组成', '', *counts, '',
             'ulb、sapienza_15S、sd_stock、sd_crypto 沿用 v1 的划分与特征定义（缓存文件哈希与 v1 记录逐一比对：24 个文件中 23 个字节一致，仅 sd_crypto/train.x.f32 不同；校准／测试文件全部一致）。banksim、s_ffsd、sapienza_5S／25S、sd_synth 为本轮按同一规则新增。S-FFSD 的标签 2 作为未知保留在训练／校准中，仅在评价时屏蔽。sd_synth 是发布者的合成异常序列，不是真实操纵。',
             '',
             '## 交易级／点级：AUC-PR（%，全量测试集合并，3 种子均值）', '',
             'AP 的随机参照水平等于正例率；不同任务的正例率差异不能当作方法差异，只在同一行内比较方法。', '',
             *table(pot, datasets, methods, 'average_precision'), '',
             '## 交易级／点级：AUC-ROC（%）', '', *table(pot, datasets, methods, 'roc_auc'), '',
             '## 固定 POT 阈值：F1（%）与 MCC（×100）', '',
             f'任一种子 POT 失败时该格为 N/A（本轮 **{len(failures)} 次 POT 失败**，见 [pot_failures.json](pot_failures.json)）。', '',
             '**F1**', '', *table(pot, datasets, methods, 'f1'), '', '**MCC**', '', *table(pot, datasets, methods, 'mcc'), '',
             '## 序列视角（TSB-AD 协议）：VUS-PR（%）', '',
             '每条含异常测试序列单独评价后宏平均；Sapienza 每个 pump 事件一条序列（测试 79 条，其中 2 条无标注块被跳过），SD-FMM 每个 160 步窗口一条序列。', '',
             *table(pot, temporal, methods, 'vus_pr'), '',
             '**VUS-ROC（%）**', '', *table(pot, temporal, methods, 'vus_roc'), '',
             '**PATE（%）**', '', *table(pot, temporal, methods, 'pate'), '',
             '**序列级 AUC-PR（%）**', '', *table(pot, temporal, methods, 'seq_auc_pr'), '',
             '## 序列视角、固定 POT 阈值', '',
             '**Affiliation-F1（%）**：随机报警的 Affiliation 指标也有不低的基线（约 0.5–0.7），应与 Affiliation-Precision／Recall 及其它指标合看。', '',
             *table(pot, temporal, methods, 'affiliation_f1'), '',
             '**事件级 F1（%，TSB-AD 定义：事件内任一报警即命中，精度按点计）**', '', *table(pot, temporal, methods, 'event_f1'), '',
             '**PA%K-F1 面积（%，K=0…100 步长 10）**', '', *table(pot, temporal, methods, 'pak_f1_area'), '',
             '**PATE-F1（%）**', '', *table(pot, temporal, methods, 'pate_f1'), '',
             '**合并事件召回（容差内，%）**：所有测试事件中，事件区间前后 τ 行内至少出现一次报警的比例。', '',
             *table(pot, temporal, methods, 'event_recall_tolerance'), '',
             '**报警片段精度（容差内，%）**：所有报警片段中落在某真实事件 ±τ 内的比例。', '',
             *table(pot, temporal, methods, 'alarm_episode_precision_tolerance'), '',
             '## 预先指定的 99% 校准分位阈值', '',
             '**F1（%）**', '', *table(quantile, datasets, methods, 'f1'), '',
             '**Affiliation-F1（%）**', '', *table(quantile, temporal, methods, 'affiliation_f1'), '',
             '![AUC-PR、VUS-PR、Affiliation-F1 与容差事件 F1](comparison.png)', '',
             '## 解释限制', '',
             '- 交易级任务（ulb、banksim、s_ffsd）没有序列结构，只报告点级指标；序列视角指标只对 Sapienza 与 SD-FMM 任务定义。',
             '- Sapienza 的行在时间上不是等间距（无成交的块缺失），按行计的容差只是时间容差的近似；每个事件只有一个标注块，PA-F1 与标准 F1 在此几乎等价。',
             '- SD-FMM 加密货币序列的起始点几乎都在第 100 步附近，发布者已对整段序列做过归一化，不能视作可部署的实时实验；股票测试仅 6 条序列，方差大。',
             '- RTTAD-TTA 在最多 8192 条无标签测试特征上自适应，是 transductive 设定；其余方法为归纳设定，信息预算不同。',
             '- 所有超参数预先固定，没有标签调参或测试最优阈值；best-F1 仅作上限参照。三种子标准差反映训练／抽样随机性，不是置信区间。',
             '- 未纳入：SynthAML 只有告警表没有特征；ADMA21 只有异常区间没有行情；CardSim 只有模拟器代码。',
             '', '## 文件', '',
             f'- [metrics.csv](metrics.csv)：{len(frame)} 行逐种子、逐阈值策略的全部指标（点级 + 序列视角）。',
             '- [summary.csv](summary.csv)：按任务×方法×阈值策略的均值与标准差。',
             '- [modern_metrics_runs.json](modern_metrics_runs.json)：每次运行的完整现代指标记录（含跳过的序列数、无报警序列数）。',
             '- [runs.json](runs.json)：基准运行记录（训练信息、阈值、评分文件 SHA256）· [environment.json](environment.json)：环境与代码指纹 · [splits.json](splits.json)：数据分区与特征定义 · [cache_files_sha256.json](cache_files_sha256.json)：输入缓存哈希。',
             f'- 执行代码提交：`{envs[0]["code_git_commit"]}`；代码指纹：`{envs[0]["code_fingerprint"]}`；Python {envs[0]["python"]}，{envs[0]["platform"]}，CPU。',
             '', '模型、逐行分数与预处理保存在本地私有运行目录，不随聚合报告上传。原始第三方数据保留各自许可。']
    (output/'README_CN.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    export(args.runs, args.cache, args.output, args.workers)
