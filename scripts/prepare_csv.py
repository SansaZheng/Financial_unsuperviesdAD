"""Import explicitly pre-split numeric CSV files without using labels to split."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, required=True); p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--test', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', required=True); p.add_argument('--features', nargs='+', required=True)
    p.add_argument('--target', default='is_anomaly')
    args = p.parse_args()
    if args.target in args.features or len(set(args.features)) != len(args.features):
        p.error('Target cannot be a feature; feature names must be unique')
    root = args.output/args.name; root.mkdir(parents=True, exist_ok=False); root.chmod(0o700)
    manifest = {'dataset':args.name,'features':args.features,'protocol':'User-provided explicit train/calibration/test files; no label-based splitting by this tool','splits':{}}
    for split in ('train','calibration','test'):
        counts = dict(rows=0,positive=0,normal=0,unknown=0)
        with (root/f'{split}.x.f32').open('wb') as fx, (root/f'{split}.y.i8').open('wb') as fy:
            for frame in pd.read_csv(getattr(args,split),chunksize=100000):
                x = frame[args.features].to_numpy(dtype=np.float32)
                if np.isinf(x).any():
                    raise ValueError('Infinite features')
                raw = frame[args.target].to_numpy() if args.target in frame else np.full(len(frame),-1)
                if not np.isin(raw,[-1,0,1]).all():
                    raise ValueError('Use binary 0/1 targets or -1=unknown; no submission placeholders')
                y = raw.astype(np.int8)
                x.tofile(fx); y.tofile(fy); counts['rows'] += len(frame)
                for value,key in [(1,'positive'),(0,'normal'),(-1,'unknown')]:
                    counts[key] += int((y==value).sum())
        if not counts['rows']:
            raise ValueError('Empty split')
        manifest['splits'][split] = counts
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared cache:',root)
