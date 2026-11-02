"""Build the nine-group cache using the pinned, independently licensed data loader.

The new detectors do not import Kronos; only this optional preparation wrapper
uses its existing publisher-file readers. No restricted raw data is uploaded.
"""
import argparse
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT = '2c0e70eb9461081c49a4ef4cfd62f4560f345477'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ieee-dir', type=Path)
    p.add_argument('--datasets', nargs='+', default=['sapienza_15S','tstr','sd_stock','sd_crypto','ulb',
                                                  'elliptic','paysim','tabformer','ibm_aml','ieee_cis'])
    args = p.parse_args()
    source = args.source_repo.resolve()
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if head != SOURCE_COMMIT:
        raise SystemExit(f'Use data-loader revision {SOURCE_COMMIT}; current source is {head}. Nothing was changed.')
    if 'ieee_cis' in args.datasets and args.ieee_dir is None:
        p.error('IEEE-CIS requires locally authorized --ieee-dir; otherwise omit it from --datasets')
    command = [sys.executable, '-m', 'benchmarks.data', '--cache', str(args.output.resolve()),
               '--datasets', *args.datasets]
    if args.ieee_dir:
        command += ['--ieee-dir', str(args.ieee_dir.resolve())]
    subprocess.run(command, cwd=source, check=True)
