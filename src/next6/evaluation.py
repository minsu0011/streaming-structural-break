"""Fixed evaluation-only helpers. No learner selection or parameter mutation."""
import csv
import numpy as np
from src.next6.common import BASE,read_json,digest,save
from src.next5.references import primary

METHODS=('M0','BASE_ONLY','M1','M2','M3','OWN','RANDOM')

def csv_write(name,rows):
    rows=list(rows);p=BASE/name;p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def frozen_predictions(data,meta):
    for fold in range(5):
        assert (BASE/f'fold_{fold}/OUTER_PREDICTIONS_FROZEN.json').exists()
    p0,_,receipt=primary('full','PRIMARY',meta,data.guard)
    out={'M0':p0,'AMPLITUDE_0.9_M0':.9*p0};receipts={'M0':receipt}
    for method in (*METHODS[1:],'M4'):
        p=np.full(len(p0),np.nan);parts=[]
        for fold in range(5):
            d=BASE/f'fold_{fold}/outer/{method}'
            if not (d/'DONE.json').exists():
                assert method=='M4';continue
            r=read_json(d/'DONE.json');assert digest(d/'predictions.npz')==r['prediction_sha256']
            with np.load(d/'predictions.npz',allow_pickle=False) as z:
                data.guard.admit(np.unique(z['dataset_id']),purpose='NEXT6 fixed outer evaluation')
                rows=z['global_row'];assert np.array_equal(meta['dataset_id'][rows],z['dataset_id'])
                assert np.all(meta['fold'][rows]==fold) and np.isnan(p[rows]).all()
                p[rows]=z['prediction']
            parts.append(r)
        if parts:out[method]=p;receipts[method]=parts
        if method!='M4':assert np.isfinite(p).all()
    return out,receipts
