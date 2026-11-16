"""Build a Chinese matched report only from independently verified runs."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = dict(iforest='Isolation Forest', lof='LOF', knn='kNN', ocsvm='One-Class SVM', pca='PCA-SPE', hbos='HBOS',
             gmm='GMM', ae='AE', vae='VAE', deep_svdd='Deep SVDD', cae_if_core='CAE-IF核心适配',
             kronos_potratio_matched='KronosAD-M', kronos_stable_nc='KronosAD-S（N+C）',
             train_prior_potratio='训练先验对照', persistence_stable_nc='持续值残差对照（N+C）',
             ebgad_release_zero='EB-GAD发布版零标签', audit_hcl_zero_adapted='Audit-HCL零标签适配',
             huge_financial_inductive='HUGE金融归纳端口', himvh_zero_reconstruction='HIMVH-Zero', sdfmm_zero_synthetic='SD-FMM-Zero')
DATASETS = dict(ulb='ULB信用卡', banksim='BankSim', s_ffsd='S-FFSD', sapienza_5S='Sapienza 5S', sapienza_15S='Sapienza 15S',
                sapienza_25S='Sapienza 25S', sd_stock='SD-FMM股票', sd_crypto='SD-FMM加密货币', sd_synth='SD-FMM合成')
METRICS = ['average_precision', 'pr_auc_trapezoid', 'roc_auc', 'precision', 'recall', 'f1', 'fpr']


def savecsv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, keys, lineterminator='\n'); w.writeheader(); w.writerows(rows)


def readcsv(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list(r):
            if k.endswith(('_mean', '_std')) or k == 'prevalence':
                r[k] = float(r[k]) if r[k] else None
            if k in ['seeds', 'threshold_successes', 'evaluated', 'unknown_excluded', 'positives', 'normal']:
                r[k] = int(r[k])
    return rows


def fmt(v):
    return 'N/A' if v is None else f'{100*v:.2f}'


def ms(row, name):
    return 'N/A' if row[name+'_mean'] is None else fmt(row[name+'_mean'])+' ± '+fmt(row[name+'_std'])


def winner(rows, track, task, metric, op):
    candidates = [r for r in rows if r['track'] == track and r['dataset'] == task and r['operating_point'] == op and r[metric+'_mean'] is not None]
    best = max(r[metric+'_mean'] for r in candidates)
    return [r for r in candidates if abs(r[metric+'_mean']-best) < 1e-12]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--update-main-report', action='store_true')
    a = p.parse_args(); verification = json.loads((a.runs/'verification.json').read_text())
    assert verification['status'] == 'passed' and verification['runs'] == 30
    records = json.loads((a.runs/'records.json').read_text()); manifest = json.loads((a.runs/'run_manifest.json').read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    for name in ['run_manifest.json', 'verification.json', 'complete.json', 'input_files_sha256.json']:
        shutil.copyfile(a.runs/name, a.output/name)
    (a.output/'per_seed.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
    rows = []; tails = []; details = []
    for r in records:
        path = a.runs/r['track']/r['dataset']/r['method']/str(r['seed'])
        thresholds = json.loads((path/'thresholds.json').read_text())
        with np.load(path/'scores.npz', allow_pickle=False) as z:
            cal, test = z['calibration'], z['test']
        detail = {k: r[k] for k in ['track', 'dataset', 'method', 'seed', 'seconds', 'peak_cuda_gb']}
        detail.update(r['model_info']); details.append(detail)
        tails.append(dict(track=r['track'], dataset=r['dataset'], method=r['method'], seed=r['seed'],
            calibration_min=float(cal.min()), calibration_max=float(cal.max()), calibration_std=float(cal.std()),
            test_min=float(test.min()), test_max=float(test.max()), test_std=float(test.std()),
            pot_status=thresholds['pot']['status'], pot_shape=thresholds['pot'].get('shape'),
            pot_threshold=thresholds['pot'].get('threshold'), pot_reason=thresholds['pot'].get('reason'),
            pot_calibration_alarm_fraction=float(np.mean(cal >= thresholds['pot']['threshold'])) if thresholds['pot']['status'] == 'ok' else None,
            q99_threshold=thresholds['quantile']['threshold'], q99_calibration_alarm_fraction=float(np.mean(cal >= thresholds['quantile']['threshold']))))
        for op, value in r['metrics']['operating_points'].items():
            row = {k: r[k] for k in ['track', 'dataset', 'method', 'seed']}
            row.update(operating_point=op, status='ok' if value is not None else 'failed',
                       **{k: v for k, v in r['metrics'].items() if k != 'operating_points'})
            row.update({k: value[k] if value else None for k in ['precision', 'recall', 'f1', 'fpr', 'tp', 'fp', 'fn', 'tn', 'alarm_fraction']})
            rows.append(row)
    assert len(rows) == 60
    savecsv(a.output/'metrics.csv', rows); savecsv(a.output/'threshold_audit.csv', tails)
    savecsv(a.output/'training_resources.csv', details)
    groups = sorted({(r['track'], r['dataset'], r['method'], r['operating_point']) for r in rows})
    summary = []
    for track, task, method, op in groups:
        g = [r for r in rows if (r['track'], r['dataset'], r['method'], r['operating_point']) == (track, task, method, op)]
        assert len(g) == 3
        row = dict(track=track, dataset=task, method=method, operating_point=op, seeds=3,
                   threshold_successes=sum(r['status'] == 'ok' for r in g), source_commit=manifest['git_commit'], source_path='finad_recent/experiment.py')
        for m in METRICS:
            values = [r[m] for r in g if r[m] is not None]
            row[m+'_mean'] = float(np.mean(values)) if len(values) == 3 else None
            row[m+'_std'] = float(np.std(values, ddof=1)) if len(values) == 3 else None
        for k in ['evaluated', 'unknown_excluded', 'positives', 'normal', 'prevalence']:
            assert len({r[k] for r in g}) == 1
            row[k] = g[0][k]
        summary.append(row)
    savecsv(a.output/'summary.csv', summary)
    old_tab = readcsv(ROOT/'results/kronos_stable_20261116/combined_transaction_summary.csv')
    old_graph = readcsv(ROOT/'results/comparison_20261114/account_graph_summary.csv')
    combined = old_tab+old_graph+summary
    assert len(combined) == 314
    for r in summary:
        old = next(x for x in old_tab+old_graph if (x['track'], x['dataset'], x['operating_point']) == (r['track'], r['dataset'], r['operating_point']))
        for k in ['evaluated', 'unknown_excluded', 'positives', 'normal', 'prevalence']:
            assert r[k] == old[k]
    savecsv(a.output/'combined_transaction_summary.csv', [r for r in combined if r['track'] == 'tabular'])
    savecsv(a.output/'combined_account_graph_summary.csv', [r for r in combined if r['track'] == 'graph'])
    best = []
    for track, tasks in [('tabular', DATASETS), ('graph', ['banksim', 's_ffsd'])]:
        for task in tasks:
            for metric, op in [('average_precision', 'pot'), ('roc_auc', 'pot'), ('f1', 'pot'), ('f1', 'quantile')]:
                for r in winner(combined, track, task, metric, op):
                    best.append(dict(track=track, dataset=task, metric=metric, operating_point=op, method=r['method'], mean=r[metric+'_mean'], std=r[metric+'_std']))
    savecsv(a.output/'best_methods.csv', best)
    gains = []
    for r in summary:
        if r['operating_point'] != 'pot': continue
        track, task = r['track'], r['dataset']
        previous = winner(old_tab+old_graph, track, task, 'average_precision', 'pot')[0]
        gains.append(dict(track=track, dataset=task, method=r['method'], ap=r['average_precision_mean'],
                          previous_best_method=previous['method'], previous_best_ap=previous['average_precision_mean'],
                          ap_delta=r['average_precision_mean']-previous['average_precision_mean']))
    savecsv(a.output/'against_previous_best.csv', gains)
    index = {(r['track'], r['dataset'], r['method'], r['operating_point']): r for r in combined}
    def table(track, task):
        lines = ['| 方法 | AP ↑ | ROC-AUC ↑ | 原POT F1 ↑ | q99 F1 ↑ | 原POT FPR ↓ |', '|---|---:|---:|---:|---:|---:|']
        candidates = [r for r in combined if r['track'] == track and r['dataset'] == task and r['operating_point'] == 'pot']
        winners = {m: {r['method'] for r in winner(combined, track, task, m, 'pot')} for m in ['average_precision', 'roc_auc', 'f1']}
        for r in candidates:
            method = r['method']; q = index[track, task, method, 'quantile']
            def cell(m):
                text = ms(r, m)
                return '**'+text+'**' if method in winners.get(m, set()) else text
            lines.append('| '+NAMES[method]+' | '+' | '.join([cell('average_precision'), cell('roc_auc'), cell('f1'), ms(q, 'f1'), ms(r, 'fpr')])+' |')
        return '\n'.join(lines)
    text = ['# HUGE、HIMVH-Zero、SD-FMM-Zero：零标签金融公平对比', '', '日期：2026-11-17。全部30次正式运行、60条操作点记录已通过独立核验。数值为百分数，均值±三个种子的样本标准差。**新零标签适配成绩不等于原论文监督成绩，账户图与交易／行情任务分别排名。**', '',
            '## 先看结论', '',
            '| 轨道 | 数据集 | 本轮方法 | AP | 上轮最佳AP方法 | 上轮最佳AP | 差值（百分点） |', '|---|---|---|---:|---|---:|---:|']
    for g in gains:
        text.append(f'| {"账户图" if g["track"] == "graph" else "交易／行情"} | {DATASETS[g["dataset"]]} | {NAMES[g["method"]]} | {fmt(g["ap"])} | {NAMES[g["previous_best_method"]]} | {fmt(g["previous_best_ap"])} | {100*g["ap_delta"]:+.2f} |')
    text += ['', 'AP比较说明排序能力，不依赖POT。上表没有将更有利的阈值当作排序改善。只有相同轨道、数据集和操作点才能比较；不跨任务平均，也不声称统计显著。', '',
             '## 方法与公平性', '',
             '- **HUGE金融归纳端口**保留作者HALO、MLP/GCN、排序与非对称对齐及负邻居余弦评分；使用真实付款图，训练后将权重迁移到校准／测试图。改变了原作者单图转导设定、图特征与归一化；排序抽样512，固定300更新。',
             '- **HIMVH-Zero**独立实现论文多视图超图、新颖度及调制融合；不使用标签差异或标签传播，监督分类改成去噪重构。BankSim、S-FFSD各4个原生视图，window4、hidden256、3层，固定200更新。使用各阶段离线关系上下文，不能解释成在线实时系统。',
             '- **SD-FMM-Zero**保留作者160长patch Transformer与13长对比支路；不使用真实标注异常片段，改用无标签小波扰动自产目标，不以验证F1挑模型。固定400更新，末行因果评分；使用160步历史，信息范围大于逐行方法与Kronos32步。',
             '- 沿用原训练抽样、校准抽样、全量测试、共同缩放器及未知标签处理。81个交易缓存文件和20个正式图文件逐一核验，所有新运行与旧AE基线索引／测试目标／缩放器一致。图仅评价原付款账户；不捏造资产序列或账户边。',
             '- 原POT（q=.85、risk=.01、至少20峰）和独立q99分别报告，POT失败记N/A，不采用Kronos专用护栏替换共同POT。指标为异常类、未经点调整；AP与梯形PR-AUC分列保存。',
             '- 训练完全不使用真实标签，但源训练窗口可能含真实异常。相同划分不代表上下文、预训练、参数量和FLOPs完全一致。旧测试集已被先前研究查看，属于固定旧测试集上的回顾性扩展，不是新盲测。', '',
             'ULB缺少原生账户关系与资产序列，本轮三方法均N/A。HUGE不用于逐笔交易排行；HIMVH不用于没有真实交易视图的行情序列；SD-FMM不把ULB/BankSim/S-FFSD全局行序强行当成单资产。N/A表示任务不兼容，不能当作0分或训练失败。完整设计、上游提交、公式歧义与适配见[固定协议](../../docs/recent_zero_protocol_CN.md)。', '',
             '## 交易／行情轨道：完整对比', '']
    for task in DATASETS:
        r = next(x for x in combined if x['track'] == 'tabular' and x['dataset'] == task)
        text += [f'### {DATASETS[task]}', '', f'已知测试{r["evaluated"]}行，异常{r["positives"]}行，占比{100*r["prevalence"]:.4f}%；未知排除{r["unknown_excluded"]}行。', '', table('tabular', task), '']
    text += ['## 账户图轨道：完整对比', '']
    for task in ['banksim', 's_ffsd']:
        r = next(x for x in combined if x['track'] == 'graph' and x['dataset'] == task)
        text += [f'### {DATASETS[task]}账户图', '', f'已知测试{r["evaluated"]}个付款账户，异常占比{100*r["prevalence"]:.2f}%；未知排除{r["unknown_excluded"]}。账户标签是阶段内交易聚合代理，不等于司法确认账户。', '', table('graph', task), '']
    text += ['## 各任务的最佳方法', '', '| 轨道 | 数据集 | AP最佳 | 原POT F1最佳 | q99 F1最佳 |', '|---|---|---|---|---|']
    for track, tasks in [('tabular', DATASETS), ('graph', ['banksim', 's_ffsd'])]:
        for task in tasks:
            cells = []
            for m, op in [('average_precision', 'pot'), ('f1', 'pot'), ('f1', 'quantile')]:
                wins = winner(combined, track, task, m, op)
                cells.append('／'.join(NAMES[r['method']] for r in wins)+' '+fmt(wins[0][m+'_mean']))
            text.append('| '+('账户图' if track == 'graph' else '交易／行情')+' | '+DATASETS[task]+' | '+' | '.join(cells)+' |')
    text += ['', '## 如何解释新方法的不足', '',
             '以下是由目标、数据与分数诊断支持的解释或机制假设；本轮没有进行逐项消融，不能当作已证明的因果结论。', '',
             '1. **零标签目标发生了实质改变。** HIMVH原本依靠分类损失、标签差异／传播学习欺诈，零标签重构更容易关注常见交易特征；异常也可能被准确重构。其原论文监督高分不能预期直接保留。SD-FMM原发布代码使用标注目标和按标签片段训练，本轮只学习泛化小波扰动；合成容易识别不代表真实欺诈也相似。',
             '2. **HUGE的图假设可能与角色图不一致。** 图边连接付款方和收款方，两端角色和交易统计天然不同，HALO可能主要反映业务角色而非欺诈。归纳跨阶段还存在分布变化。该机制解释需角色／特征消融验证，不能仅凭一次成绩认定。',
             '3. **异常不是统一的“稀有离群点”。** S-FFSD已知测试交易异常占比约38.87%，已知测试付款账户异常占比约87.48%。这些与原图欺诈方法假定正常占多数的情景不同；未知样本参与无标签训练和校准，但最终只评估已知样本，也会改变被评价子集的分布。',
             '4. **POT与排序能力是两个问题。** 固定1%风险及尾部分布拟合未必对应测试异常比例，校准到测试的分数漂移可造成低召回或误报；AP较高、F1较低并不矛盾。下面公布阈值成功率和实际校准报警率，不能只看拟合成功。',
             '5. **训练预算和适配范围有限。** 200/300/400更新是固定预算，不是原论文完整epoch或超参搜索；本轮不能证明方法经过充分调优仍劣于经典基线，也不会用测试标签补调后再报告同一次实验。', '',
             '### 阈值与资源诊断', '', '| 任务／方法 | 原POT成功 | 校准POT报警率范围 | 校准q99报警率范围 | 平均运行秒数 | 最大CUDA分配GB |', '|---|---:|---:|---:|---:|---:|']
    for g in gains:
        group = [r for r in records if (r['track'], r['dataset'], r['method']) == (g['track'], g['dataset'], g['method'])]
        tail = [r for r in tails if (r['track'], r['dataset'], r['method']) == (g['track'], g['dataset'], g['method'])]
        rates = [r['pot_calibration_alarm_fraction'] for r in tail if r['pot_calibration_alarm_fraction'] is not None]
        qr = [r['q99_calibration_alarm_fraction'] for r in tail]
        interval = f'{100*min(rates):.3f}–{100*max(rates):.3f}%' if rates else 'N/A'
        peak = max(r['peak_cuda_gb'] or 0 for r in group)
        text.append(f'| {DATASETS[g["dataset"]]}／{NAMES[g["method"]]}／{g["track"]} | {len(rates)}/3 | {interval} | {100*min(qr):.3f}–{100*max(qr):.3f}% | {np.mean([r["seconds"] for r in group]):.1f} | {peak:.2f} |')
    text += ['', '运行秒数包含本轮各自的数据加载、拟合、重载检查和全量评分；不是等硬件、等信息量的速度排行。训练首末损失及分数范围／标准差、GPD形状、阈值见CSV。', '',
             '## 复核与文件', '', f'正式科学源码：`{manifest["git_commit"]}`。本轮未按金融测试结果选择任何超参数或分数方向。', '',
             '- [逐种子完整记录](per_seed.json)、[60条操作点指标](metrics.csv)、[方法汇总](summary.csv)。',
             '- [交易全方法汇总](combined_transaction_summary.csv)、[账户图全方法汇总](combined_account_graph_summary.csv)、[最佳方法](best_methods.csv)。',
             '- [阈值审计](threshold_audit.csv)、[训练与资源](training_resources.csv)、[独立核验](verification.json)、[输入摘要](input_files_sha256.json)、[运行环境](run_manifest.json)。',
             '- 大型模型和分数保存在CQ输出及本地实验归档，不将二进制模型混入Git。复核命令如下；需完整分数／检查点和上一轮正式输出。', '',
             '```bash', 'pip install -e ".[recent2026,test]"',
             'PYTHONPATH=. python scripts/verify_recent_zero.py --runs /path/to/new/results \\',
             '  --data /path/to/new/inputs/tabular --graphs /path/to/new/inputs/graphs \\',
             '  --old-runs /path/to/j-1114-2e8178/results',
             'PYTHONPATH=. python scripts/report_recent_zero.py --runs /path/to/new/results \\',
             '  --output results/recent_zero_20261117', '```', '',
             '## Plain English', '',
             'We evaluated a financial inductive HUGE port and explicit label-free adaptations of HIMVH and SD-FMM. All 30 seeded runs use the previous evaluation cohorts and shared preprocessing. Transaction/market-point and payer-account graph tasks are separate. HIMVH reconstruction and SD-FMM synthetic corruption change the original supervised objectives; these results do not reproduce the papers’ supervised claims. Exact input fingerprints, checkpoints, scores and independently replayed metrics are retained. Different history, graph and pretraining budgets remain disclosed limitations.', '']
    (a.output/'README_CN.md').write_text('\n'.join(text))
    section = ['## 12. HUGE、HIMVH-Zero、SD-FMM-Zero 同划分零标签扩展（2026-11-17）', '',
               '全部30次正式运行和60条操作点指标已完成并独立核验。HUGE在2组账户图运行；HIMVH-Zero在2组交易数据运行；SD-FMM-Zero在6组行情时序运行，各3种子。原81个交易缓存及20个正式图输入、抽样／测试对象／共同缩放器均核对，未知标签只在评价排除。', '',
               '**HIMVH与SD-FMM是显式零标签适配，不能称原论文监督结果复现。** HUGE为跨阶段归纳端口；上下文信息、预训练和计算量不同，不声称完全等信息预算。本轮仍用共同原POT与独立q99，POT失败N/A。前述第2节保留11月16日15方法历史表，最新排名以本节链接为准。', '',
               '[最新完整中文报告、交易与账户图对比、各数据集最佳方法及原因分析](../results/recent_zero_20261117/README_CN.md)。', '',
               '| 轨道 | 数据集 | 新方法 | AP | 旧最佳AP | 差值（百分点） |', '|---|---|---|---:|---:|---:|']
    for g in gains:
        section.append(f'| {g["track"]} | {DATASETS[g["dataset"]]} | {NAMES[g["method"]]} | {fmt(g["ap"])} | {NAMES[g["previous_best_method"]]} {fmt(g["previous_best_ap"])} | {100*g["ap_delta"]:+.2f} |')
    (a.output/'main_report_section_CN.md').write_text('\n'.join(section)+'\n')
    if a.update_main_report:
        report = ROOT/'docs/financial_ad_comparison_CN.md'; original = report.read_text()
        marker = '\n## 12. HUGE、HIMVH-Zero、SD-FMM-Zero'
        if marker in original:
            original = original.split(marker)[0]
        notice = '**2026-11-17最新扩展：** [HUGE、HIMVH-Zero、SD-FMM-Zero完整中文报告](../results/recent_zero_20261117/README_CN.md)。最新最佳方法见第12节及该报告；下文11月16日和此前表格保留为历史结果。\n\n'
        if notice not in original:
            original = original.replace('# 金融无监督异常检测综合对比报告\n\n', '# 金融无监督异常检测综合对比报告\n\n'+notice, 1)
        report.write_text(original.rstrip()+'\n\n'+'\n'.join(section)+'\n')
    print('REPORT EXPORTED: 30 runs, 60 operating records, 314 combined summaries')


if __name__ == '__main__':
    main()
