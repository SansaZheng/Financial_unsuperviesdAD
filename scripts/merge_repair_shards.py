"""Read-only, lossless materialization of three independent family seed jobs."""
import argparse
import json
import os
from pathlib import Path
import shutil
from finad_recent.data import sha,write
from finad_repair.experiment import planned


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shards',type=Path,nargs=3,required=True,help='three whole CQ output roots')
    p.add_argument('--inputs',type=Path,required=True,help='paired, immutable input root')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();records=[];provenance=[];manifests=[];golden=None;seen=set();family=None
    for job in a.shards:
        run=job/'results';manifest=json.loads((run/'run_manifest.json').read_text())
        if family is None:family=manifest['family']
        assert family in ['sdfmm','himvh'] and manifest['family']==family and len(manifest['seeds'])==1
        seed=manifest['seeds'][0];assert seed in [42,43,44] and seed not in seen;seen.add(seed)
        n=len(planned([family],[seed]))
        assert json.loads((run/'complete.json').read_text())==dict(finished=n,expected=n,failed=0)
        rows=json.loads((run/'records.json').read_text())
        assert {(r['family'],r['dataset'],r['variant'],r['seed']) for r in rows}==set(planned([family],[seed]))
        fingerprint=json.loads((run/'input_files_sha256.json').read_text())
        if golden is None:golden=fingerprint
        assert fingerprint==golden
        records.extend(rows);manifests.append(manifest)
        provenance.append(dict(seed=seed,job_directory=job.name,manifest=manifest,
                               manifest_sha256=sha(run/'run_manifest.json'),
                               records_sha256=sha(run/'records.json'),complete_sha256=sha(run/'complete.json')))
    total=len(planned([family]));assert seen=={42,43,44} and len(records)==total
    for m in manifests[1:]:
        assert {k:v for k,v in m.items() if k!='seeds'}=={k:v for k,v in manifests[0].items() if k!='seeds'}
    a.output.mkdir(parents=True,exist_ok=False);dest=a.output/'results';dest.mkdir()
    for job in a.shards:
        shutil.copytree(job/'results/tabular',dest/'tabular',dirs_exist_ok=True,copy_function=os.link)
    (a.output/'inputs').symlink_to(a.inputs.resolve(),target_is_directory=True)
    manifest=manifests[0].copy();manifest['seeds']=[42,43,44]
    manifest['composed_from_independent_seed_shards']=True
    write(dest/'run_manifest.json',manifest);write(dest/'records.json',records)
    write(dest/'complete.json',dict(finished=total,expected=total,failed=0))
    write(dest/'input_files_sha256.json',golden);write(dest/'shard_provenance.json',provenance)
    print(f'Materialized{total} unique configurations; per-run bytes unchanged; original shard manifests preserved')


if __name__=='__main__':main()
