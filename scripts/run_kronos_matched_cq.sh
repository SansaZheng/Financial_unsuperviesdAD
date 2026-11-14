#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PIP_CACHE_DIR="$CQ_OUTPUT_DIR/pip-cache"
mode="${1:-smoke}"
shift || true
finalize() {
  run_status=$?
  python - <<'FINALIZE'
import hashlib,json,os,shutil,tarfile
from pathlib import Path
out=Path(os.environ['CQ_OUTPUT_DIR'])
tests=out/'pytest'
if tests.exists():
    for p in tests.rglob('*'):
        if p.is_symlink():p.unlink()
tab=out/'work/tabular'
if tab.exists():
    archive=out/'prepared_inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:tf.add(tab,arcname='tabular')
    (out/'input_archive.json').write_text(json.dumps(dict(file=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))+'\n')
    shutil.rmtree(tab)
if (out/'pip-cache').exists():shutil.rmtree(out/'pip-cache')
for name in ['tests.txt','progress.log']:
    p=out/name
    if p.exists():print('DIAGNOSTIC '+name+'\n'+p.read_text(errors='replace')[-12000:],flush=True)
print('OUTPUT_BYTES',sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),flush=True)
FINALIZE
  return "$run_status"
}
trap finalize EXIT
python -m pip install -q 'numpy==2.1.3' 'scipy==1.16.3' 'pandas==2.3.3' 'scikit-learn==1.7.2' 'huggingface-hub>=0.25,<1' 'safetensors>=0.4' 'einops>=0.8' 'tqdm>=4.65' 'pytest>=8' 'joblib>=1.3' 'threadpoolctl>=3'
if [[ "$mode" == smoke ]]; then
  python -m pytest -q -p no:cacheprovider --basetemp "$CQ_OUTPUT_DIR/pytest" tests/test_kronos_matched.py > "$CQ_OUTPUT_DIR/tests.txt" 2>&1
  python scripts/prepare_repo_data.py --output "$CQ_OUTPUT_DIR/work/tabular" --datasets banksim > "$CQ_OUTPUT_DIR/progress.log" 2>&1
  python scripts/kronos_matched_smoke.py --data "$CQ_OUTPUT_DIR/work/tabular" --assets "$CQ_OUTPUT_DIR/assets" --output "$CQ_OUTPUT_DIR/smoke" >> "$CQ_OUTPUT_DIR/progress.log" 2>&1
else
  python scripts/prepare_repo_data.py --output "$CQ_OUTPUT_DIR/work/tabular" --datasets "$@" > "$CQ_OUTPUT_DIR/progress.log" 2>&1
  python -m finad_kronos.experiment --data "$CQ_OUTPUT_DIR/work/tabular" --output "$CQ_OUTPUT_DIR/results" --assets "$CQ_OUTPUT_DIR/assets" --tasks "$@" --workers 2 >> "$CQ_OUTPUT_DIR/progress.log" 2>&1
  python scripts/verify_kronos_matched.py --results "$CQ_OUTPUT_DIR/results" --data "$CQ_OUTPUT_DIR/work/tabular" >> "$CQ_OUTPUT_DIR/progress.log" 2>&1
fi
