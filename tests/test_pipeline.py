from argparse import Namespace
import json
import numpy as np
import pandas as pd
import pytest

from finad.experiment import benchmark
from finad.predict import predict
from finad.report import collect, export
from finad.verify import verify


def test_full_pipeline_without_train_or_calibration_labels(tmp_path):
    root=tmp_path/'cache'/'toy';root.mkdir(parents=True)
    rng=np.random.default_rng(42)
    manifest={'features':['f1','f2','f3'],'protocol':'invented fixtures only','splits':{}}
    test=None
    for split,n in [('train',80),('calibration',400),('test',30)]:
        x=rng.normal(size=(n,3)).astype(np.float32);x.tofile(root/f'{split}.x.f32')
        manifest['splits'][split]={'rows':n}
        if split=='test':
            test=x
            np.array([0]*20+[1]*10,dtype=np.int8).tofile(root/'test.y.i8')
    (root/'manifest.json').write_text(json.dumps(manifest))
    args=Namespace(command='benchmark',cache=tmp_path/'cache',output=tmp_path/'runs',datasets=['toy'],
                   methods=['iforest'],seeds=[42],train_cap=64,calibration_cap=400,steps=2,
                   pretrain_steps=2,batch_size=32,risk=.01,initial_quantile=.85,device='cpu')
    benchmark(args)
    run=args.output/'toy'/'iforest_seed42'
    result=json.loads((run/'result.json').read_text())
    assert result['metrics']['evaluated']==30
    original=np.load(run/'test_scores.npy')
    pd.DataFrame(test,columns=manifest['features']).to_csv(tmp_path/'query.csv',index=False)
    predict(run,tmp_path/'query.csv',tmp_path/'prediction.csv',chunksize=7)
    np.testing.assert_allclose(pd.read_csv(tmp_path/'prediction.csv').anomaly_score,original)
    audit=verify(args.output,args.cache)
    assert audit['verified_runs']==1
    export(args.output,tmp_path/'report')
    assert len(pd.read_csv(tmp_path/'report'/'metrics.csv'))==2
    assert json.loads((tmp_path/'report'/'environment.json').read_text())['configuration']['cache']=='/path/to/private/feature_cache'
    benchmark(args)  # Exact resumption is safe.
    args.train_cap=63
    with pytest.raises(ValueError,match='different sweep'):
        benchmark(args)


def test_report_refuses_missing_experiments(tmp_path):
    (tmp_path/'environment.json').write_text(json.dumps({'configuration':{
        'datasets':['toy'],'methods':['iforest'],'seeds':[42,43,44]}}))
    with pytest.raises(ValueError,match='Incomplete/duplicate sweep'):
        collect(tmp_path)
