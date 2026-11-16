"""Fixed label-free fit -> checkpoint reload -> scores -> thresholds -> labels."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import scipy
import sklearn
import torch
import pywt
from threadpoolctl import threadpool_limits
from finad.preprocessing import Preprocessor
from finad.thresholds import calibrate
from finad.evaluation import metrics
from finad_zero.graph_data import load_phase
from finad_zero.experiment import code_commit
from .data import ROOT, PHASES, TEMPORAL, GRAPH, sha, write, verify_inputs, transaction_keys
from .huge import HUGE, bidirected_union
from .himvh import HIMVHZero
from .sdfmm import SDFMMZero, sequence_starts


def load_tabular(data, out, seed):
    meta = json.loads((data/'manifest.json').read_text()); p = len(meta['features'])
    raw = {s: np.memmap(data/f'{s}.x.f32', mode='r', dtype='float32', shape=(meta['splits'][s]['rows'], p)) for s in PHASES}
    ti = np.sort(np.random.default_rng(seed).choice(len(raw['train']), min(8192, len(raw['train'])), replace=False))
    ci = np.sort(np.random.default_rng(seed+100).choice(len(raw['calibration']), min(100000, len(raw['calibration'])), replace=False))
    pre = Preprocessor().fit(raw['train'][ti]); pre.save(out/'preprocessor.npz')
    np.savez(out/'indices.npz', train=ti, calibration=ci, test=np.arange(len(raw['test'])))
    x = {s: pre.transform(raw[s]) for s in PHASES}
    return meta, x, ti, ci


def tabular_run(data, out, task, method, seed, device, args):
    meta, x, ti, ci = load_tabular(data, out, seed)
    if method == 'himvh_zero_reconstruction':
        views = {}
        for phase in PHASES:
            views[phase], n = transaction_keys(task, phase, ti if phase == 'train' else None)
            assert n == (len(ti) if phase == 'train' else len(x[phase]))
        model = HIMVHZero(x['train'].shape[1], 4, seed, device, args.himvh_steps).fit(x['train'][ti], views['train'])
        model.save(out/'model.pt'); loaded = HIMVHZero.load(out/'model.pt', device)
        # Check on the training graph before any target labels are opened.
        a = model.score(x['train'][ti], views['train']); b = loaded.score(x['train'][ti], views['train'])
        cs = model.score(x['calibration'], views['calibration'])[ci]
        ts = model.score(x['test'], views['test'])
        detail = dict(phase_hyperedges={s: [len(e) for e in views[s]] for s in PHASES},
                      information_budget='offline phase-specific native transaction views, no cross-partition edges')
    elif method == 'sdfmm_zero_synthetic':
        starts = {s: sequence_starts(np.fromfile(data/f'{s}.seq.i32', dtype=np.int32)) for s in PHASES}
        model = SDFMMZero(x['train'].shape[1], seed, device, args.sd_steps).fit(x['train'], starts['train'], ti)
        model.save(out/'model.pt'); loaded = SDFMMZero.load(out/'model.pt', device)
        probe = np.arange(min(257, len(x['test'])))
        a = model.score(x['test'], starts['test'], probe); b = loaded.score(x['test'], starts['test'], probe)
        cs = model.score(x['calibration'], starts['calibration'], ci)
        ts = model.score(x['test'], starts['test'], np.arange(len(x['test'])))
        detail = dict(information_budget='160-row past/current context within phase and sequence; train target indices unchanged')
    else:
        raise ValueError(method)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    checks = dict(checkpoint_max_abs=float(np.max(np.abs(a-b))), source_manifest=meta,
                  test_target_sha256=sha(data/'test.y.i8'), **detail)
    return finish(out, cs, ts, lambda: np.fromfile(data/'test.y.i8', dtype=np.int8)), model.info, checks


def graph_run(data, out, seed, device, args):
    raw = {s: load_phase(data, s) for s in PHASES}
    roles = {s: raw[s][0][0, :, 0] > .5 for s in PHASES}
    pre = Preprocessor().fit(raw['train'][0].reshape(-1, 10)); pre.save(out/'preprocessor.npz')
    flat = {}
    edges = {}
    for s in PHASES:
        v, e, _ = raw[s]
        flat[s] = pre.transform(v.reshape(-1, 10)).reshape(v.shape).transpose(1, 0, 2).reshape(v.shape[1], -1)
        edges[s] = bidirected_union(e, v.shape[1])
    eligible = np.flatnonzero(roles['train'])
    ti = np.sort(np.random.default_rng(seed).choice(eligible, min(8192, len(eligible)), replace=False))
    np.savez(out/'indices.npz', train=ti, calibration=np.flatnonzero(roles['calibration']), test=np.flatnonzero(roles['test']))
    model = HUGE(40, seed, device, args.huge_steps).fit(flat['train'], edges['train'])
    model.save(out/'model.pt'); loaded = HUGE.load(out/'model.pt', device)
    a = model.score(flat['calibration'], edges['calibration']); b = loaded.score(flat['calibration'], edges['calibration'])
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    cs = a[roles['calibration']]
    ts = model.score(flat['test'], edges['test'])[roles['test']]
    checks = dict(checkpoint_max_abs=float(np.max(np.abs(a-b))), source_manifest=json.loads((data/'manifest.json').read_text()),
                  test_target_sha256=sha(data/'test.y.npy'),
                  information_budget='all train graph nodes; fixed trained weights transfer to independent phase graphs; evaluated payers unchanged',
                  undirected_edges={s: int(edges[s].shape[1]) for s in PHASES})
    return finish(out, cs, ts, lambda: np.load(data/'test.y.npy', allow_pickle=False)[roles['test']]), model.info, checks


def finish(out, calibration, test, read_test_labels):
    if not np.isfinite(calibration).all() or not np.isfinite(test).all():
        raise FloatingPointError('Nonfinite saved scores')
    np.savez_compressed(out/'scores.npz', calibration=calibration, test=test)
    thresholds = calibrate(calibration); write(out/'thresholds.json', thresholds)
    # The only label access in the execution path, after model, reload, scores
    # and calibration thresholds have been durably recorded.
    write(out/'label_access_gate.json', dict(checkpoint_sha256=sha(out/'model.pt'), scores_sha256=sha(out/'scores.npz'),
          thresholds_sha256=sha(out/'thresholds.json'), status='scores_and_thresholds_saved_before_labels'))
    return metrics(read_test_labels(), test, thresholds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='cpu')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--huge-steps', type=int, default=300); p.add_argument('--himvh-steps', type=int, default=200)
    p.add_argument('--sd-steps', type=int, default=400)
    p.add_argument('--methods', nargs='+', default=['huge_financial_inductive', 'himvh_zero_reconstruction', 'sdfmm_zero_synthetic'])
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if min(args.huge_steps, args.himvh_steps, args.sd_steps) < 1:
        raise ValueError('Positive fixed training budgets required')
    torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    write(args.output/'input_files_sha256.json', verify_inputs(args.data, args.graphs))
    scientific = list((ROOT/'finad_recent').glob('*.py'))+[ROOT/'finad'/f for f in ['preprocessing.py', 'thresholds.py', 'evaluation.py']]
    manifest = dict(git_commit=code_commit(), source_sha256={str(f.relative_to(ROOT)): sha(f) for f in sorted(scientific)},
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                    torch=torch.__version__, pywavelets=pywt.__version__, device=args.device, seeds=args.seeds,
                    budgets=dict(huge=args.huge_steps, himvh=args.himvh_steps, sdfmm=args.sd_steps), methods=args.methods)
    path = args.output/'run_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Resume source/configuration differs')
    write(path, manifest)
    jobs = [(track, task, method, seed) for method in args.methods
            for track, task in ([('graph', t) for t in GRAPH] if method == 'huge_financial_inductive' else
                                [('tabular', t) for t in (GRAPH if method == 'himvh_zero_reconstruction' else TEMPORAL)])
            for seed in args.seeds]
    records = []
    for i, (track, task, method, seed) in enumerate(jobs, 1):
        out = args.output/track/task/method/str(seed); out.mkdir(parents=True, exist_ok=True)
        if (out/'result.json').exists():
            records.append(json.loads((out/'result.json').read_text())); continue
        print(f'RUN {i}/{len(jobs)} {track}/{task}/{method}/{seed}', flush=True)
        start = time.monotonic()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        with threadpool_limits(limits=4):
            if track == 'graph':
                result, info, checks = graph_run(args.graphs/task, out, seed, args.device, args)
            else:
                result, info, checks = tabular_run(args.data/task, out, task, method, seed, args.device, args)
        record = dict(track=track, dataset=task, method=method, seed=seed, seconds=time.monotonic()-start,
                      metrics=result, model_info=info, checks=checks, checkpoint_sha256=sha(out/'model.pt'),
                      score_sha256=sha(out/'scores.npz'),
                      peak_cuda_gb=torch.cuda.max_memory_allocated()/2**30 if args.device.startswith('cuda') else None)
        write(out/'result.json', record); records.append(record)
        write(args.output/'progress.json', dict(finished=len(records), expected=len(jobs), failed=0))
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        # No numerical financial outcomes are printed before the complete run.
        print(f'DONE {i}/{len(jobs)} {task}/{method}/{seed} seconds={record["seconds"]:.1f}', flush=True)
    write(args.output/'records.json', records)
    write(args.output/'complete.json', dict(finished=len(records), expected=len(jobs), failed=0))


if __name__ == '__main__':
    main()
