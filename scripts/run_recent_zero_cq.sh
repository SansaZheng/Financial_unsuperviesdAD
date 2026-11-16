#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD" PIP_CACHE_DIR="$CQ_OUTPUT_DIR/pip-cache" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
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
    for name in tests.txt progress.log; do
      if [[ -f "$CQ_OUTPUT_DIR/$name" ]]; then tail -n 80 "$CQ_OUTPUT_DIR/$name"; fi
    done
  fi
  return "$task_status"
}
trap cleanup EXIT
python -m pip install -q 'numpy==2.3.5' 'scipy==1.15.3' 'pandas>=2.2,<3' 'scikit-learn==1.7.2' 'PyWavelets==1.8.0' 'pytest>=8' 'joblib>=1.3' 'threadpoolctl>=3'
mode="${1:-env}"
if [[ "$mode" == env ]]; then
  python -m pytest -q -p no:cacheprovider --basetemp "$CQ_OUTPUT_DIR/pytest" tests/test_recent.py > "$CQ_OUTPUT_DIR/tests.txt" 2>&1
  python -c 'import json,torch,numpy,scipy,pywt; print(json.dumps(dict(status="passed",torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,pywavelets=pywt.__version__)))' > "$CQ_OUTPUT_DIR/environment.json"
elif [[ "$mode" == smoke ]]; then
  python scripts/recent_zero_smoke.py --device cuda --resource-profile --output "$CQ_OUTPUT_DIR/qualification" > "$CQ_OUTPUT_DIR/progress.log" 2>&1
else
  python scripts/prepare_repo_data.py --output "$CQ_OUTPUT_DIR/inputs/tabular" --datasets ulb banksim s_ffsd sapienza_5S sapienza_15S sapienza_25S sd_stock sd_crypto sd_synth > "$CQ_OUTPUT_DIR/progress.log" 2>&1
  python -m finad_zero.graph_data --output "$CQ_OUTPUT_DIR/inputs/graphs" >> "$CQ_OUTPUT_DIR/progress.log" 2>&1
  python -m finad_recent.experiment --device cuda --data "$CQ_OUTPUT_DIR/inputs/tabular" --graphs "$CQ_OUTPUT_DIR/inputs/graphs" --output "$CQ_OUTPUT_DIR/results" >> "$CQ_OUTPUT_DIR/progress.log" 2>&1
fi
