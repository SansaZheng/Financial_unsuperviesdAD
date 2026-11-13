"""Verify immutable source/input artifacts and extract the exact prepared inputs.

This is a post-experiment integrity audit, not model training. It hashes target
files as bytes but does not interpret training/calibration labels.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

p=argparse.ArgumentParser();p.add_argument('--job-output',type=Path,required=True)
p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);args=p.parse_args()
root=args.job_output;results=root/'results'
manifest=json.loads((results/'run_manifest.json').read_text());commit=manifest['git_commit']
assert re.fullmatch('[a-f0-9]{40}',commit)
assert json.loads((results/'complete.json').read_text())==dict(finished=144,failed=0,expected=144)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(2**20):h.update(b)
    return h.hexdigest()
sources={}
for name,expected in manifest['source_sha256'].items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    content=subprocess.check_output(['git','-C',str(args.repo),'show',commit+':'+name])
    actual=hashlib.sha256(content).hexdigest();assert actual==expected,(name,actual,expected)
    sources[name]=actual
archive_info=json.loads((results/'input_archive.json').read_text());archive=root/archive_info['file']
assert archive.parent==root and archive.stat().st_size==archive_info['bytes']
assert digest(archive)==archive_info['sha256']
exact=root/'exact_inputs';exact.mkdir(exist_ok=True)
with tarfile.open(archive,'r:gz') as tf:
    members=tf.getmembers();assert sum(m.size for m in members)<1024**3
    for member in members:
        parts=Path(member.name).parts
        assert parts and parts[0]=='tabular' and '..' not in parts and not Path(member.name).is_absolute()
        assert member.isfile() or member.isdir()
    tf.extractall(exact,filter='data')
cache=json.loads((results/'input_cache_files_sha256.json').read_text());input_files={}
for name,meta in cache.items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    path=exact/'tabular'/name
    assert path.stat().st_size==meta['bytes'] and digest(path)==meta['sha256'],name
    input_files[name]=meta
# Verify the archive pinned by the wrapper without importing or executing it.
eb_source=subprocess.check_output(['git','-C',str(args.repo),'show',commit+':finad_zero/ebgad.py']).decode()
constants={}
for statement in ast.parse(eb_source).body:
    if isinstance(statement,ast.Assign) and isinstance(statement.value,ast.Constant):
        for target in statement.targets:
            if isinstance(target,ast.Name):constants[target.id]=statement.value.value
upstream=root/'work/upstream/ebgad.tar.gz'
assert digest(upstream)==constants['TAR_SHA256']
graph_files={str(path.relative_to(root/'work/graphs')):dict(bytes=path.stat().st_size,sha256=digest(path))
             for path in sorted((root/'work/graphs').rglob('*')) if path.is_file()}
result=dict(status='passed',benchmark_commit=commit,source_files=sources,input_archive=archive_info,
            prepared_input_files=len(input_files),prepared_input_bytes=sum(v['bytes'] for v in input_files.values()),
            upstream_commit=constants['COMMIT'],upstream_archive_sha256=constants['TAR_SHA256'],
            graph_files=graph_files)
(results/'artifact_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print('ZERO_LABEL_SOURCE_AND_INPUT_ARTIFACTS_VERIFIED',flush=True)
print('Exact prepared inputs:',exact/'tabular',flush=True)
