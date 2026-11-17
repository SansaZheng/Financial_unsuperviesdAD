"""Train-observation/synthetic-only audit: never load any target array."""
import argparse
import json
from pathlib import Path
import numpy as np
from finad.preprocessing import Preprocessor
from finad_recent.data import transaction_keys, write
from finad_recent.himvh import hyperedges, discrepancies
from finad_recent.sdfmm import contexts, sequence_starts, synthesize
from finad_repair.sdfmm import coherent, transform_windows
from finad_repair.hypergraph import NativeContext


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    audit=dict(real_training_labels_used=False,calibration_labels_used=False,test_labels_used=False,
               score_direction='Original higher-risk directions retained; no outcome-dependent sign flips')
    task='sapienza_15S';d=a.data/task;meta=json.loads((d/'manifest.json').read_text());n=meta['splits']['train']['rows'];width=len(meta['features'])
    raw=np.memmap(d/'train.x.f32',mode='r',dtype='float32',shape=(n,width));ti=np.sort(np.random.default_rng(42).choice(n,8192,replace=False))
    pre=Preprocessor().fit(raw[ti]);starts=sequence_starts(np.fromfile(d/'train.seq.i32',dtype=np.int32));rows=ti[(ti-starts[ti])>=160][:64]
    original=contexts(raw,starts,rows);normalized=transform_windows(original,pre)
    old,_,_,_=synthesize(normalized,np.random.default_rng(42));inverted=old*pre.scale+pre.center
    new=coherent(original,rows-starts[rows],task,np.random.default_rng(42))
    def clock_error(w):return np.abs(w[:,-13:,8]**2+w[:,-13:,9]**2-1)
    audit['sdfmm']=dict(source='train only,64 sampled warm windows',original_clock_violation_fraction=float(np.mean(clock_error(original)>1e-3)),
                        legacy_clock_violation_fraction=float(np.mean(clock_error(inverted)>1e-3)),
                        repaired_clock_violation_fraction=float(np.mean(clock_error(new)>1e-3)),
                        repaired_clock_coordinates_unchanged=bool(np.array_equal(original[:,:,8:],new[:,:,8:])),
                        legacy_synthetic_positive_fraction=13/320,endpoint_target_positive_fraction=.5,
                        endpoint_gradient_dilution_factor=160)
    x=np.random.default_rng(8).normal(size=(12,3)).astype(np.float32);changed=x.copy();changed[0,0]+=20
    views=[hyperedges(np.arange(12)//4),hyperedges(np.arange(12)%3)]*2
    olda,oldb=discrepancies(x,views),discrepancies(changed,views);olda[0,0]=oldb[0,0]=0
    x[0,0]=changed[0,0]=0;ctx=NativeContext(x,views)
    audit['himvh']=dict(clean_aux_hidden_target_influence_max=float(np.max(np.abs(olda-oldb))),
                        repaired_masked_aux_influence_max=float(np.max(np.abs(ctx.enhanced(x,np.arange(12))-ctx.enhanced(changed,np.arange(12))))))
    meta=json.loads((a.data/'banksim/manifest.json').read_text());n=meta['splits']['train']['rows'];ti=np.sort(np.random.default_rng(42).choice(n,8192,replace=False))
    selected,_=transaction_keys('banksim','train',ti);native,_=transaction_keys('banksim','train')
    audit['himvh']['banksim_train_hyperedges_selected']=[len(e) for e in selected]
    audit['himvh']['banksim_train_hyperedges_native']=[len(e) for e in native]
    audit['interpretation']='Confirmed adaptation counterexamples, not proof of each cause of observed financial AP; controlled ablations required. No original score-orientation bug established.'
    write(a.output,audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
