"""Reload a locally saved detector and score numeric CSV chunks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .classical import ClassicalDetector, METHODS as CLASSICAL
from .neural import NeuralDetector
from .preprocessing import Preprocessor
from .detectors import load_detector, extension


def predict(run, csv, output, device='cpu', chunksize=8192):
    run, output = Path(run), Path(output)
    config = json.loads((run/'config.json').read_text())
    features = json.loads((run/'features.json').read_text())['features']
    thresholds = json.loads((run/'thresholds.json').read_text())
    prep = Preprocessor.load(run/'preprocessor.npz')
    method = config['method']
    model = load_detector(method,run/f'model.{extension(method)}',device)
    # Refuse overwrite. Output contains scores and row positions, not source personal identifiers.
    with output.open('x') as destination:
        offset, header = 0, True
        for frame in pd.read_csv(csv, usecols=features, chunksize=chunksize):
            score = model.score(prep.transform(frame[features].to_numpy(dtype=np.float32)))
            result = pd.DataFrame({'row': np.arange(offset, offset+len(frame)), 'anomaly_score': score})
            for name, value in thresholds.items():
                result[f'{name}_is_anomaly'] = score >= value['threshold'] if value['status'] == 'ok' else pd.NA
            result.to_csv(destination, index=False, header=header)
            offset += len(frame); header = False
    return offset
