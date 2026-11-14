"""Full-test matched Kronos-Proxy benchmark. No label-based fitting or selection."""
import argparse
import gc
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import subprocess
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from finad.preprocessing import Preprocessor
from finad.thresholds import calibrate
from finad.evaluation import metrics
from .assets import fetch_assets, load_predictor, sha
from .adapter import LOOKBACK, EPSILON, sequence_starts, history, stamp_table, forecast, residual_scores, batch_seed

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('ulb', 'banksim', 's_ffsd', 'sapienza_5S', 'sapienza_15S', 'sapienza_25S', 'sd_stock', 'sd_crypto', 'sd_synth')
METHOD = 'kronos_proxy_matched'


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def source_manifest():
    declared = os.environ.get('FINAD_CODE_COMMIT')
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = declared
    if not commit or not re.fullmatch('[a-f0-9]{40}', commit) or declared and commit != declared:
        raise ValueError('A matching complete source commit is required')
    paths = sorted((ROOT/'finad_kronos').glob('*.py'))+[ROOT/'finad'/f for f in ('preprocessing.py', 'thresholds.py', 'evaluation.py')]+[ROOT/'scripts/prepare_repo_data.py']
    return dict(git_commit=commit, source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths})


def verify_inputs(data, tasks):
    golden = json.loads((ROOT/'results/zero_label_20261114/input_cache_files_sha256.json').read_text())
    checked = {}
    for name, record in golden.items():
        if name.split('/')[0] not in tasks:
            continue
        path = data/name
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'Input differs from the paired baseline: {name}')
        checked[name] = record
    if not checked or {p.split('/')[0] for p in checked} != set(tasks):
        raise ValueError('Missing golden input records')
    return checked


def load_values(data, seed, out):
    meta = json.loads((data/'manifest.json').read_text())
    width = len(meta['features'])
    raw = {s: np.memmap(data/f'{s}.x.f32', dtype='float32', mode='r', shape=(meta['splits'][s]['rows'], width)) for s in ('train', 'calibration', 'test')}
    ti = np.sort(np.random.default_rng(seed).choice(len(raw['train']), min(8192, len(raw['train'])), replace=False))
    ci = np.sort(np.random.default_rng(seed+100).choice(len(raw['calibration']), min(100000, len(raw['calibration'])), replace=False))
    pre = Preprocessor().fit(raw['train'][ti]); pre.save(out/'preprocessor.npz')
    reloaded = Preprocessor.load(out/'preprocessor.npz')
    np.testing.assert_array_equal(pre.transform(raw['train'][ti[:17]]), reloaded.transform(raw['train'][ti[:17]]))
    values = np.concatenate([pre.transform(raw[s]) for s in ('train', 'calibration', 'test')])
    nt, nc = len(raw['train']), len(raw['calibration'])
    if meta.get('temporal'):
        # Group ids from different partitions must never share a history.
        starts = np.concatenate([sequence_starts(np.fromfile(data/f'{s}.seq.i32', dtype=np.int32))+offset
                                 for s, offset in [('train', 0), ('calibration', nt), ('test', nt+nc)]])
    else:
        starts = np.zeros(len(values), dtype=np.int64)
    targets = {'calibration': nt+ci, 'test': nt+nc+np.arange(len(raw['test']))}
    np.savez(out/'indices.npz', train=ti, calibration=ci, test=np.arange(len(raw['test'])))
    counts = {s: dict(scored=len(t), cold_start=int(np.count_nonzero(t-starts[t] < LOOKBACK)),
                      no_history=int(np.count_nonzero(t == starts[t]))) for s, t in targets.items()}
    return values, starts, targets, meta, counts


def score_phase(predictor, values, starts, targets, out, phase, dataset, seed, batch_size):
    width = values.shape[1]
    work = out/'work'; work.mkdir(exist_ok=True)
    pred_path, score_path, progress_path = work/f'{phase}.predictions.npy', work/f'{phase}.scores.npy', work/f'{phase}.progress.json'
    exists = progress_path.exists()
    predictions = np.lib.format.open_memmap(pred_path, mode='r+' if exists else 'w+', dtype=np.float32, shape=(len(targets), width))
    scores = np.lib.format.open_memmap(score_path, mode='r+' if exists else 'w+', dtype=np.float64, shape=(len(targets),))
    cursor = json.loads(progress_path.read_text())['completed_forecasts'] if exists else 0
    if not exists:
        scores[:] = 0
    local = targets-starts[targets]
    stamps = stamp_table(int(local.max()))
    total = len(targets)*width
    started = time.monotonic()
    for start in range(cursor, total, batch_size):
        flat = np.arange(start, min(start+batch_size, total)); ri, fi = flat//width, flat % width
        h, _ = history(values, targets[ri], fi, starts)
        pred, std = forecast(predictor, h, local[ri], batch_seed(dataset, seed, phase, start//batch_size), stamps)
        predictions[ri, fi] = pred
        error = residual_scores(values[targets[ri], fi], pred, std)
        np.maximum.at(scores, ri, error)
        done = int(flat[-1]+1)
        if start == cursor or (start//batch_size) % 32 == 31 or done == total:
            predictions.flush(); scores.flush()
            write(progress_path, dict(completed_forecasts=done, total_forecasts=total))
            print(f'PROGRESS {dataset} seed={seed} {phase} {done}/{total} feature forecasts {time.monotonic()-started:.1f}s', flush=True)
    if not np.isfinite(predictions).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid completed forecasts or scores')
    return predictions, scores


def run_one(data, output, predictor, dataset, seed, batch_size):
    out = output/dataset/str(seed); out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists():
        return
    started = time.monotonic()
    values, starts, targets, meta, cold = load_values(data/dataset, seed, out)
    predictions, scores = {}, {}
    for phase in ('calibration', 'test'):
        predictions[phase], scores[phase] = score_phase(predictor, values, starts, targets[phase], out, phase, dataset, seed, batch_size)
    np.savez_compressed(out/'forecasts.npz', **predictions)
    np.savez_compressed(out/'scores.npz', **scores)
    thresholds = calibrate(scores['calibration']); write(out/'thresholds.json', thresholds)
    info = dict(method=METHOD, frozen_weights=True, representation='one engineered feature copied into OHLC slots; volume/amount zero placeholders',
                observed_features='the shared train-only clipped baseline feature space', lookback=LOOKBACK,
                sample_count=1, temperature=1., top_p=.9, top_k=0, batch_size=batch_size, epsilon=EPSILON,
                scalar_score='max_i ((z_ti - predicted_z_ti)/(past_32_std_i + 1e-6))**2',
                history='strictly past; stream history crosses chronological splits; event/case histories reset',
                cold_start='left-pad with training medians (zero in shared centered space), never omit a test row',
                extra_information='frozen pretraining and preceding observed feature rows; no future rows or labels', cold_start_counts=cold)
    write(out/'model_spec.json', info)
    # Scores, forecast archive, configuration and both thresholds exist BEFORE this label read.
    y = np.fromfile(data/dataset/'test.y.i8', dtype=np.int8)
    result = dict(track='tabular', dataset=dataset, method=METHOD, seed=seed, seconds=time.monotonic()-started,
                  device=predictor.device, metrics=metrics(y, scores['test'], thresholds), model_info=info,
                  checks=dict(test_target_sha256=sha(data/dataset/'test.y.i8'), source_manifest=meta,
                              score_sha256=sha(out/'scores.npz'), forecasts_sha256=sha(out/'forecasts.npz'),
                              train_preprocessing_reloaded=True, full_test_scored=len(y)==len(scores['test'])))
    write(out/'result.json', result)
    del values, starts, predictions, scores
    gc.collect()
    # Full forecast archives are retained; remove only their regenerable uncompressed work copies.
    import shutil
    shutil.rmtree(out/'work')
    print(f'FINISHED {dataset} seed={seed}', flush=True)


def worker(device, tasks, options):
    torch.set_num_threads(options['threads'])
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    predictor = load_predictor(options['assets'], device)
    while True:
        try:
            dataset, seed = tasks.get_nowait()
        except queue.Empty:
            return
        try:
            with threadpool_limits(limits=options['threads']):
                run_one(options['data'], options['output'], predictor, dataset, seed, options['batch_size'])
        except Exception as error:
            out = options['output']/dataset/str(seed); out.mkdir(parents=True, exist_ok=True)
            write(out/'failure.json', dict(dataset=dataset, seed=seed, error=str(error), traceback=traceback.format_exc()))
            print(traceback.format_exc(), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--batch-size', type=int, default=1024); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', default='cuda'); a = p.parse_args()
    if a.batch_size < 1 or len(set(a.tasks)) != len(a.tasks) or len(set(a.seeds)) != len(a.seeds):
        raise ValueError('Invalid batching or duplicate task/seed')
    a.output.mkdir(parents=True, exist_ok=True)
    inputs = verify_inputs(a.data, a.tasks); write(a.output/'input_files_sha256.json', inputs)
    receipt = fetch_assets(a.assets); write(a.output/'assets.json', receipt)
    manifest = dict(**source_manifest(), tasks=a.tasks, seeds=a.seeds, method=METHOD, batch_size=a.batch_size,
                    lookback=LOOKBACK, epsilon=EPSILON, policies=['pot q=.85 risk=.01 min_peaks=20', '99% calibration quantile'],
                    python=__import__('platform').python_version(), packages={k: importlib.metadata.version(k) for k in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'huggingface-hub', 'safetensors', 'einops']})
    saved = a.output/'run_manifest.json'
    if saved.exists() and json.loads(saved.read_text()) != manifest:
        raise ValueError('Resume source/configuration/environment mismatch')
    write(saved, manifest)
    devices = [f'cuda:{i}' for i in range(min(a.workers, torch.cuda.device_count()))] if a.device=='cuda' else [a.device]
    if not devices:
        raise ValueError('No GPU available')
    context = mp.get_context('spawn'); tasks = context.Queue()
    # Long tasks first; scheduling does not alter per-dataset/seed/phase/batch RNG keys.
    jobs = []
    for d in a.tasks:
        meta = json.loads((a.data/d/'manifest.json').read_text())
        cost = len(meta['features'])*(min(100000, meta['splits']['calibration']['rows'])+meta['splits']['test']['rows'])
        jobs.extend((cost, d, s) for s in a.seeds)
    for _, d, s in sorted(jobs, reverse=True):
        tasks.put((d, s))
    options = dict(data=a.data, output=a.output, assets=a.assets, batch_size=a.batch_size, threads=max(1, 4//len(devices)))
    processes = [context.Process(target=worker, args=(device, tasks, options)) for device in devices]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    records, failures = [], []
    for _, d, s in jobs:
        result, failed = a.output/d/str(s)/'result.json', a.output/d/str(s)/'failure.json'
        if result.exists():
            records.append(json.loads(result.read_text()))
        elif failed.exists():
            failures.append(json.loads(failed.read_text()))
        else:
            failures.append(dict(dataset=d, seed=s, error='worker exited without a result', worker_exit_codes=[x.exitcode for x in processes]))
    write(a.output/'records.json', records); write(a.output/'failures.json', failures)
    write(a.output/'complete.json', dict(finished=len(records), failed=len(failures), expected=len(jobs)))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
