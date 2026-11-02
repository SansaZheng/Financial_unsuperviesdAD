"""Frozen, label-blind fitting/calibration and full held-out evaluation."""
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .classical import ClassicalDetector, METHODS as CLASSICAL
from .data import Cache, DATASETS, sample, sha256
from .evaluation import metrics
from .neural import NeuralDetector, METHODS as NEURAL
from .preprocessing import Preprocessor
from .thresholds import calibrate

METHODS = CLASSICAL+NEURAL


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def code_fingerprint():
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        if path.name not in ('report.py', 'verify.py'):
            digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def environment(args):
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'joblib', 'threadpoolctl']
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        commit = None
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={name: importlib.metadata.version(name) for name in packages},
                code_git_commit=commit, code_fingerprint=code_fingerprint(),
                configuration={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                training_labels_used=False, calibration_labels_used=False)


def run_one(args, dataset, method, seed, fingerprint):
    cache = Cache(args.cache, dataset)
    destination = args.output / dataset / f'{method}_seed{seed}'
    config = dict(dataset=dataset, method=method, seed=seed, train_cap=args.train_cap,
                  calibration_cap=args.calibration_cap, steps=args.steps, pretrain_steps=args.pretrain_steps,
                  device=args.device if method in NEURAL else 'cpu', batch_size=args.batch_size,
                  risk=args.risk, initial_quantile=args.initial_quantile,
                  data_manifest_sha256=sha256(cache.path/'manifest.json'), code_fingerprint=fingerprint)
    result_path = destination/'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['run_config'] != config:
            raise ValueError(f'Different configuration already exists in {destination}; use a new output')
        return result
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'config.json', config)
    start = time.monotonic()
    print(f'START {dataset} {method} seed={seed}', flush=True)
    raw_train, indices = sample(cache.x('train'), args.train_cap, seed)
    prep = Preprocessor().fit(raw_train)
    train = prep.transform(raw_train); del raw_train
    prep.save(destination/'preprocessor.npz'); np.save(destination/'train_indices.npy', indices)
    write_json(destination/'features.json', {'features': cache.features})
    model = (ClassicalDetector(method, seed) if method in CLASSICAL else
             NeuralDetector(method, seed, args.device, args.steps, args.pretrain_steps)).fit(train)
    extension = 'joblib' if method in CLASSICAL else 'pt'
    model.save(destination/f'model.{extension}')
    info = model.info
    train_seconds = time.monotonic()-start
    del train; gc.collect()
    raw_cal, cal_indices = sample(cache.x('calibration'), args.calibration_cap, seed+100)
    cal_scores = np.concatenate([model.score(prep.transform(raw_cal[i:i+args.batch_size]))
                               for i in range(0, len(raw_cal), args.batch_size)])
    if not np.isfinite(cal_scores).all():
        raise ValueError('Nonfinite calibration scores')
    thresholds = calibrate(cal_scores, args.initial_quantile, args.risk)
    write_json(destination/'thresholds.json', thresholds)
    np.save(destination/'calibration_scores.npy', cal_scores)
    np.save(destination/'calibration_indices.npy', cal_indices)
    del raw_cal, cal_scores
    test = cache.x('test')
    test_scores = np.lib.format.open_memmap(destination/'test_scores.npy', mode='w+',
                                          dtype=np.float64, shape=(len(test),))
    for i in range(0, len(test), args.batch_size):
        test_scores[i:i+args.batch_size] = model.score(prep.transform(test[i:i+args.batch_size]))
    test_scores.flush()
    # This is the first target-file read. Training/calibration targets are never opened.
    evaluation = metrics(cache.y('test'), test_scores, thresholds)
    result = dict(status='complete', dataset=dataset, method=method, seed=seed,
                  run_config=config, train_rows_used=len(indices), calibration_rows_used=len(cal_indices),
                  split_counts=cache.manifest['splits'], data_protocol=cache.manifest['protocol'],
                  training=info, train_seconds=train_seconds, total_seconds=time.monotonic()-start,
                  thresholds=thresholds, metrics=evaluation,
                  test_scores_sha256=sha256(destination/'test_scores.npy'))
    write_json(result_path, result)
    op = evaluation['operating_points']['pot']
    f1 = op['f1'] if op is not None else None
    print(f'DONE {dataset} {method} seed={seed} F1_POT={f1} AP={evaluation["average_precision"]} seconds={result["total_seconds"]:.1f}', flush=True)
    del model, test_scores, test; gc.collect()
    if args.device == 'mps':
        torch.mps.empty_cache()
    return result


def benchmark(args):
    if any(len(value) != len(set(value)) for value in (args.datasets, args.methods, args.seeds)):
        raise ValueError('Datasets, methods and seeds must not contain duplicates')
    if min(args.train_cap, args.calibration_cap, args.batch_size, args.steps) < 1 or args.pretrain_steps < 1:
        raise ValueError('Caps, training steps and batch sizes must be positive')
    if not 0 < args.risk < 1-args.initial_quantile < 1:
        raise ValueError('Invalid risk/quantile')
    args.output.mkdir(parents=True, exist_ok=True); args.output.chmod(0o700)
    (args.output/'.gitignore').write_text('*\n')
    env = environment(args)
    # Preserve the full sweep definition when resuming exactly the same plan.
    if (args.output/'environment.json').exists():
        old = json.loads((args.output/'environment.json').read_text())
        if old['configuration'] != env['configuration'] or old['code_fingerprint'] != env['code_fingerprint']:
            raise ValueError('Output belongs to a different sweep; use a new output directory')
    else:
        write_json(args.output/'environment.json', env)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for dataset in args.datasets:
            for method in args.methods:
                for seed in args.seeds:
                    # Explicit failures stop the sweep; already-complete runs remain resumable.
                    run_one(args, dataset, method, seed, env['code_fingerprint'])
