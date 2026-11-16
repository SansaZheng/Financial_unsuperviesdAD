#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
cleanup_pytest_aliases() {
  python - <<'PY'
import os
from pathlib import Path
folder=Path(os.environ['CQ_OUTPUT_DIR'])/'pytest'
if folder.exists():
    for path in folder.rglob('*'):
        if path.is_symlink():path.unlink()
PY
}
trap cleanup_pytest_aliases EXIT
python -m pip install -q 'numpy==2.1.3' 'scipy==1.16.3' 'pandas==2.3.3' 'scikit-learn==1.7.2' 'pytest>=8'
python -m pytest -q -p no:cacheprovider --basetemp "$CQ_OUTPUT_DIR/pytest" tests/test_kronos_stable.py > "$CQ_OUTPUT_DIR/tests.txt" 2>&1
cat "$CQ_OUTPUT_DIR/tests.txt"
python scripts/kronos_stable_smoke.py --output "$CQ_OUTPUT_DIR/qualification" > "$CQ_OUTPUT_DIR/progress.log" 2>&1
cat "$CQ_OUTPUT_DIR/progress.log"
