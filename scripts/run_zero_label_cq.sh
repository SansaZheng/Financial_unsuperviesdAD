#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" PIP_CACHE_DIR="$CQ_OUTPUT_DIR/pip-cache"
# pytest tmp_path creates '*current' symlinks; CQ correctly rejects links in
# collected archives. Remove only links generated in this job's pytest scratch.
cleanup_test_links() {
  task_status=$?
  python - <<'CLEAN'
import os
from pathlib import Path
root=Path(os.environ['CQ_OUTPUT_DIR'])/'pytest'
if root.exists():
    for path in root.rglob('*'):
        if path.is_symlink(): path.unlink()
CLEAN
  if [[ "$task_status" != 0 ]]; then
    for diagnostic in tests.txt progress.log; do
      if [[ -f "$CQ_OUTPUT_DIR/$diagnostic" ]]; then
        echo "DIAGNOSTIC $diagnostic"
        tail -n 100 "$CQ_OUTPUT_DIR/$diagnostic"
      fi
    done
  fi
  return "$task_status"
}
trap cleanup_test_links EXIT
python -m pip install -q 'numpy>=2,<3' 'scipy>=1.15,<2' 'pandas>=2.2,<3' 'scikit-learn==1.7.2' 'torch-geometric==2.6.1' 'pytest>=8' 'joblib>=1.3' 'threadpoolctl>=3'
mode="${1:-smoke}"
if [[ "$mode" == env ]]; then
  python -m pytest -q -p no:cacheprovider --basetemp "$CQ_OUTPUT_DIR/pytest" tests/test_zero_label.py > "$CQ_OUTPUT_DIR/tests.txt" 2>&1
  python -c 'import json,sys,torch,torch_geometric,numpy,sklearn; print(json.dumps(dict(status="passed",python=sys.version,torch=torch.__version__,pyg=torch_geometric.__version__,numpy=numpy.__version__,sklearn=sklearn.__version__)))' > "$CQ_OUTPUT_DIR/env.json"
elif [[ "$mode" == smoke ]]; then
  python -m pytest -q -p no:cacheprovider --basetemp "$CQ_OUTPUT_DIR/pytest" tests/test_zero_label.py > "$CQ_OUTPUT_DIR/tests.txt" 2>&1
  python scripts/zero_label_smoke.py --device cuda --output "$CQ_OUTPUT_DIR/smoke" > "$CQ_OUTPUT_DIR/progress.log" 2>&1
else
  python scripts/prepare_repo_data.py --output "$CQ_OUTPUT_DIR/work/tabular" --datasets ulb banksim s_ffsd sapienza_5S sapienza_15S sapienza_25S sd_stock sd_crypto sd_synth > "$CQ_OUTPUT_DIR/progress.log" 2>&1
  python -m finad_zero.graph_data --output "$CQ_OUTPUT_DIR/work/graphs" >> "$CQ_OUTPUT_DIR/progress.log" 2>&1
  python -m finad_zero.experiment --device cuda --data "$CQ_OUTPUT_DIR/work/tabular" --graph-data "$CQ_OUTPUT_DIR/work/graphs" --output "$CQ_OUTPUT_DIR/results" --cache "$CQ_OUTPUT_DIR/work/eb_cache" --upstream "$CQ_OUTPUT_DIR/work/upstream" >> "$CQ_OUTPUT_DIR/progress.log" 2>&1
  python scripts/verify_zero_label.py --results "$CQ_OUTPUT_DIR/results" --data "$CQ_OUTPUT_DIR/work/tabular" --graph-data "$CQ_OUTPUT_DIR/work/graphs" >> "$CQ_OUTPUT_DIR/progress.log" 2>&1
fi
