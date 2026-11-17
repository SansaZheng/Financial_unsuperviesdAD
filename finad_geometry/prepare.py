"""Generate one previously unseen CardSim world with predeclared parameters."""
import argparse,hashlib,json,sys,tarfile,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from finad_recent.data import PHASES,sha,write
from finad_recent.himvh import hyperedges
from finad_zero.graph_data import make_phase,aggregate_targets,FEATURES
from scripts.prepare_repo_data import cumulative_time_boundaries,temporal_split,signed_log,hash_onehot,fixed_onehot

SEED=2026111707

def prepare(output):
    output.mkdir(parents=True,exist_ok=False);up=output/'upstream';up.mkdir()
    archive=Path('data/raw/cardsim/feds-2025017-python.zip')
    with zipfile.ZipFile(archive) as z:z.extractall(up)
    sys.path.insert(0,str(up/'cardsim'))
    from cardsim import Cardsim
    sim=Cardsim(seed=SEED,payer_payee_factor=1,dcpc_folder=str(up/'cardsim/cardsim/dcpc'),log_level='WARNING')
    d=sim.simulate(n_payers=1024,n_days=90)
    if d is None:raise RuntimeError('Official simulator failed; never substitute fabricated data')
    d=d.sort_values('date_time',kind='stable').reset_index(drop=True);d.to_csv(output/'raw_transactions.csv.gz',index=False,compression='gzip')
    t=pd.to_datetime(d.date_time).astype('int64').to_numpy();bounds=cumulative_time_boundaries(t);split=temporal_split(t,*bounds)
    x=np.column_stack([signed_log(d.amount),signed_log(d.distance),fixed_onehot(d.credit_card.astype(int).astype(str),['0','1']),fixed_onehot(d.remote.astype(int).astype(str),['0','1']),hash_onehot(d.payee_id,16),np.sin(2*np.pi*d.time_seconds/86400),np.cos(2*np.pi*d.time_seconds/86400)]).astype('f4')
    names=['log_amount','log_distance']+['card_0','card_1','remote_0','remote_1']+[f'payee_hash_{i}' for i in range(16)]+['clock_sin','clock_cos'];assert x.shape[1]==24
    root=output/'inputs';tab=root/'tabular/cardsim';graph=root/'graphs/cardsim';tab.mkdir(parents=True);graph.mkdir(parents=True)
    meta=dict(dataset='cardsim',features=names,time_boundaries=list(map(int,bounds)),splits={},simulator_seed=SEED,raw_sha256=sha(output/'raw_transactions.csv.gz'),generator_zip_sha256=sha(archive),independent_of_previous_runs=True,synthetic=True)
    gm=dict(dataset='cardsim',features=FEATURES,splits={},raw_sha256=meta['raw_sha256'],time_boundaries=meta['time_boundaries'])
    for k,s in enumerate(PHASES):
        mask=split==k;f=d.loc[mask];xx=x[mask];xx.tofile(tab/f'{s}.x.f32');y=f.fraud.to_numpy(dtype='i1');y.tofile(tab/f'{s}.y.i8');np.flatnonzero(mask).astype('i8').tofile(tab/f'{s}.rows.i64')
        views=[hyperedges(f[c].astype(str).to_numpy(),4) for c in ['payer_id','payee_id','credit_card','remote']];np.savez_compressed(tab/f'{s}.views.npz',**{f'e{i}':v for i,v in enumerate(views)})
        meta['splits'][s]=dict(rows=len(f))
        sources=sorted(f.payer_id.unique(),key=lambda a:hashlib.sha256(('financial-zero-20261114:'+str(a)).encode()).hexdigest())[:4096];gf=f.loc[f.payer_id.isin(sources)];tt=pd.to_datetime(gf.date_time).astype('int64').to_numpy()
        gx,edges,active,src,dst,ids,sb=make_phase(gf.payer_id,gf.payee_id,tt,gf.amount.to_numpy());np.savez_compressed(graph/f'{s}.graph.npz',x=gx,active=active,**{f'e{i}':e for i,e in enumerate(edges)});(graph/f'{s}.ids.json').write_text(json.dumps(ids));np.save(graph/f'{s}.y.npy',aggregate_targets(src,dst,gf.fraud.to_numpy(),len(ids)),allow_pickle=False)
        gm['splits'][s]=dict(nodes=len(ids),edges=sum(e.shape[1] for e in edges),evaluated_source_accounts=len(sources),snapshot_boundaries=list(map(int,sb)))
    write(tab/'manifest.json',meta);write(graph/'manifest.json',gm)
    files={str(f.relative_to(root)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(root.rglob('*')) if f.is_file()};write(output/'input_files_sha256.json',files)
    with tarfile.open(output/'cardsim_inputs.tar.gz','w:gz') as a:
        for p in sorted(root.iterdir()):a.add(p,arcname=p.name)
    write(output/'generation.json',dict(status='complete',seed=SEED,n_payers=1024,n_days=90,payer_payee_factor=1,rows=len(d),input_files=len(files),generator_zip_sha256=sha(archive),archive_sha256=sha(output/'cardsim_inputs.tar.gz'),archive_bytes=(output/'cardsim_inputs.tar.gz').stat().st_size,raw_sha256=meta['raw_sha256'],source='https://www.federalreserve.gov/econres/feds/cardsim-a-bayesian-simulator-for-payment-card-fraud-detection-research.htm',labels_used_for_learning=False,outcomes_inspected=False))
    import shutil;shutil.rmtree(up)
    print('GENERATED independent CardSim world; feature-only row count',len(d),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output)
