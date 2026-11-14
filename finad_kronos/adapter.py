"""Causal histories and fixed max-residual scoring; never accepts target labels."""
import hashlib
import numpy as np
import pandas as pd
import torch

LOOKBACK = 32
EPSILON = 1e-6


def sequence_starts(sequence):
    sequence = np.asarray(sequence)
    if sequence.ndim != 1 or not len(sequence):
        raise ValueError('A nonempty sequence vector is required')
    changes = np.r_[0, np.flatnonzero(sequence[1:] != sequence[:-1])+1]
    if len(np.unique(sequence[changes])) != len(changes):
        raise ValueError('Each sequence must occupy a single contiguous block')
    return np.repeat(changes, np.diff(np.r_[changes, len(sequence)]))


def history(values, targets, features, starts, lookback=LOOKBACK):
    targets, features = np.asarray(targets), np.asarray(features)
    indexes = targets[:, None]-np.arange(lookback, 0, -1)
    valid = indexes >= starts[targets, None]
    # Left fill is zero in the shared train-median-centered feature space.
    gathered = values[np.maximum(indexes, 0), features[:, None]]
    return np.where(valid, gathered, 0).astype(np.float64), valid.sum(axis=1)


def stamp_table(max_position, lookback=LOOKBACK):
    times = pd.DatetimeIndex(pd.Timestamp('2000-01-01')+pd.to_timedelta(np.arange(-lookback, max_position+1), unit='min'))
    return np.column_stack([times.minute, times.hour, times.dayofweek, times.day, times.month]).astype(np.float32)


def proxy_input(histories):
    histories = np.asarray(histories, dtype=np.float64)
    mean, std = histories.mean(axis=1), histories.std(axis=1, ddof=0)
    z = np.clip((histories-mean[:, None])/(std[:, None]+1e-5), -5, 5)
    x = np.zeros((*z.shape, 6), dtype=np.float32)
    x[:, :, :4] = z[:, :, None]
    return x, mean, std


def residual_scores(observed, predicted, history_std):
    error = np.square((np.asarray(observed, dtype=np.float64)-predicted)/(history_std+EPSILON))
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite Kronos residual scores')
    return error


def batch_seed(dataset, seed, phase, batch_number):
    key = f'financial-kronos-matched-v1:{dataset}:{seed}:{phase}:{batch_number}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)


def forecast(predictor, histories, local_positions, random_seed, stamps=None):
    x, mean, std = proxy_input(histories)
    positions = np.asarray(local_positions)
    stamps = stamp_table(int(positions.max())) if stamps is None else stamps
    xs = stamps[positions[:, None]-np.arange(LOOKBACK, 0, -1)+LOOKBACK]
    ys = stamps[positions[:, None]+LOOKBACK]
    device = torch.device(predictor.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.random.default_generator.manual_seed(random_seed)
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(random_seed)
        generated = predictor.generate(x, xs, ys, 1, 1., 0, .9, 1, False)
    prediction = np.asarray(generated)[:, 0, 3].astype(np.float64)*(std+1e-5)+mean
    # Stored float32 predictions are also the values used for the scored experiment.
    prediction = prediction.astype(np.float32)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite Kronos forecasts')
    return prediction, std
