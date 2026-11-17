#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
export PIP_CACHE_DIR="$CQ_OUTPUT_DIR/pip-cache" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cleanup() {
  task_status=$?
  python - <<'PY'
import os
from pathlib import Path
p=Path(os.environ['CQ_OUTPUT_DIR'])/'pytest'
if p.exists():
 for f in p.rglob('*'):
  if f.is_symlink(): f.unlink()
PY
  if [[ "$task_status" != 0 ]]; then
    for f in tests.txt progress.log; do if [[ -f "$CQ_OUTPUT_DIR/$f" ]]; then tail -n 80 "$CQ_OUTPUT_DIR/$f"; fi; done
  fi
  return "$task_status"
}
trap cleanup EXIT
python -m pip install -q 'numpy==2.3.5' 'scipy==1.15.3' 'pandas==2.3.3' 'scikit-learn==1.7.2' 'PyWavelets==1.8.0' 'pytest>=8' 'joblib>=1.3' 'threadpoolctl>=3' 'pythresh==1.1.1' 'numba>=0.63'
mode="${1:-env}"
if [[ "$mode" == env ]]; then
  python -m pytest -q -p no:cacheprovider --basetemp "$CQ_OUTPUT_DIR/pytest" > "$CQ_OUTPUT_DIR/tests.txt" 2>&1
  tail -n 40 "$CQ_OUTPUT_DIR/tests.txt"
else
  python - <<'STAGE' > "$CQ_OUTPUT_DIR/progress.log" 2>&1
import hashlib,json,os,tarfile
from pathlib import Path
base=Path(os.environ['CQ_OUTPUT_DIR'])/'inputs';base.mkdir(parents=True)
for name,subdir in [('tabular',''),('graph','graphs')]:
 m=json.loads(Path('docs/recent_zero_'+name+'_archive.json').read_text());p=Path(m['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
 dest=base/subdir;dest.mkdir(exist_ok=True)
 with tarfile.open(p) as archive:archive.extractall(dest,filter='data')
print('Staged101 exact prior input files',flush=True)
STAGE
  if [[ "$mode" == smoke ]]; then
    python scripts/repair_smoke.py --data "$CQ_OUTPUT_DIR/inputs/tabular" --graphs "$CQ_OUTPUT_DIR/inputs/graphs" --output "$CQ_OUTPUT_DIR/qualification" >> "$CQ_OUTPUT_DIR/progress.log" 2>&1
  else
    python -m finad_repair.experiment --family "$mode" --device cuda --data "$CQ_OUTPUT_DIR/inputs/tabular" --graphs "$CQ_OUTPUT_DIR/inputs/graphs" --output "$CQ_OUTPUT_DIR/results" >> "$CQ_OUTPUT_DIR/progress.log" 2>&1
  fi
fi
