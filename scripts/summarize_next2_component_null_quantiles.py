"""Complete score/max quantiles for the existing 25 CV0 null diagnostics."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    out=ROOT/'artifacts/next2/component_null_quantiles';out.mkdir(parents=True,exist_ok=True)
    quantiles=(.5,.9,.95,.99,.999);rows=[];sources={}
    for folder in sorted((ROOT/'artifacts/next2/component_stress_cv0').iterdir()):
        if not (folder/'RESULTS.json').exists():continue
        summary=json.loads((folder/'RESULTS.json').read_text());assert summary['status']=='PASS'
        for bg in ('N0','N1','N2','N3'):
            base=folder/'long_null'/bg;receipt=json.loads((base/'RESULTS.json').read_text());path=base/'prediction.npy'
            assert receipt['status']=='PASS' and sha256(path)==receipt['files_sha256'][path.name]
            sources[str((base/'RESULTS.json').relative_to(ROOT))]=sha256(base/'RESULTS.json')
            p=np.load(path,mmap_mode='r',allow_pickle=False);assert p.shape==(512,16384)
            for horizon in (1024,4096,16384):
                prefix=p[:,:horizon];maxima=prefix.max(axis=1)
                pq=np.quantile(prefix,quantiles);mq=np.quantile(maxima,quantiles)
                rows.append({'candidate':folder.name,'regime':bg,'horizon':horizon,'fitted_fold':0,
                    'artifact_kind':'existing full-CV fold-0 model','median_max_score':float(np.median(maxima)),
                    **{'score_quantile_'+str(q):float(v) for q,v in zip(quantiles,pq)},
                    **{'series_max_quantile_'+str(q):float(v) for q,v in zip(quantiles,mq)}})
        print('NEXT2 COMPONENT NULL QUANTILES',folder.name,flush=True)
    assert len(rows)==25*4*3
    frame=pd.DataFrame(rows);frame.to_csv(out/'QUANTILES.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'script_sha256':sha256(Path(__file__)),
        'source_receipts_sha256':sources,'table_sha256':sha256(out/'QUANTILES.csv'),'new_fits':0,'seal_rows':0,
        'interpretation':'Pooled point-score quantiles and across-series prefix-maximum quantiles are different quantities. CV0 fitted models only; do not mix these with all-DEV fits.'})
    checkpoint('Completed all component null score and maximum quantiles',['Integrate long-null diagnostic tables','Keep fitted-model identities explicit'])


if __name__=='__main__':main()
