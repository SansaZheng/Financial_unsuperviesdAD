import argparse
from pathlib import Path

from .data import DATASETS


def main():
    parser = argparse.ArgumentParser(description='Classic and 2026 financial anomaly detectors')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('benchmark', help='label-blind fit/calibration and full held-out metrics')
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument('--methods', nargs='+', default=['iforest','lof','knn','ocsvm','pca','hbos','gmm','ae','vae','deep_svdd'])
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    p.add_argument('--train-cap', type=int, default=8192)
    p.add_argument('--calibration-cap', type=int, default=100000)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--pretrain-steps', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--risk', type=float, default=.01)
    p.add_argument('--initial-quantile', type=float, default=.85)
    p.add_argument('--device', choices=['cpu','mps','cuda'], default='cpu')
    p = sub.add_parser('predict', help='score new CSV using a trusted local checkpoint')
    p.add_argument('--run', type=Path, required=True); p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--device', default='cpu')
    p = sub.add_parser('report', help='export aggregate evidence only')
    p.add_argument('--runs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('verify', help='recompute saved confusion matrices and score digests')
    p.add_argument('--runs', type=Path, required=True); p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'benchmark':
        from .experiment import benchmark, METHODS
        if not set(args.methods) <= set(METHODS):
            parser.error('Unknown method')
        benchmark(args)
    elif args.command == 'predict':
        from .predict import predict
        print('Scored rows:', predict(args.run, args.csv, args.output, args.device))
    elif args.command == 'report':
        from .report import export
        export(args.runs, args.output)
    else:
        from .verify import verify
        from .experiment import write_json
        result = verify(args.runs, args.cache); write_json(args.output, result); print(result)


if __name__ == '__main__':
    main()
