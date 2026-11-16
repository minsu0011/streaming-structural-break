"""Scorer parity on a fixed development prefix, and causal feature gate refresh."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.loader import load_training
from src.data.labels import labels_from_tau
from src.features.streaming import replay,feature_version
from src.models.baselines import OfficialEWMA
from src.scoring.ts_auc import score_frames
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json,sha256,utc_now
from scripts.verify_scorer import main as synthetic_parity,load_official,official_score

def main():
    synthetic_parity()
    split=FrozenSplit(ROOT/'artifacts/validation')
    pieces=[];prefix_errors=[];rng=np.random.default_rng(20260908)
    for number,series in enumerate(load_training(ROOT/'data/raw',split=split)):
        h=series.historical.astype(np.float32);o=series.online.astype(np.float32)
        if number<24:
            features=np.array(list(replay(h,o)))
            length=min(50,len(o))
            perturbed=np.r_[o[:length],np.full(17,1000,dtype=np.float32)]
            alternative=np.array(list(replay(h,perturbed)))
            prefix_errors.append(float(np.max(abs(features[:length]-alternative[:length]))))
            if not np.isfinite(features).all():raise AssertionError('Nonfinite real feature')
        detector=OfficialEWMA(h)
        pred=np.array([detector.predict_one(point) for point in o],dtype=np.float32)
        index=pd.MultiIndex.from_arrays([np.full(len(o),series.dataset_id),series.online_index],names=['id','time'])
        pieces.append(pd.DataFrame({'target':labels_from_tau(len(o),series.tau),'prediction':pred},index=index))
        if number==127:break
    frame=pd.concat(pieces).sort_index();official=load_official();comparisons=[]
    for kind,values in [('official_ewma',frame.prediction.to_numpy()),('constant',np.full(len(frame),.5,dtype=np.float32)),('seeded_ties',rng.integers(0,8,len(frame)).astype(np.float32)/7)]:
        p=pd.DataFrame({'prediction':values},index=frame.index);y=frame[['target']]
        ours=score_frames(p,y);reference=official_score(official,p,y)
        comparisons.append({'prediction':kind,'ours':ours,'official':reference,'abs_error':abs(ours-reference)})
    maximum=max(r['abs_error'] for r in comparisons)
    report={'status':'PASS' if maximum<1e-12 else 'FAIL','timestamp':utc_now(),'data_scope':'FIRST_128_FROZEN_DEVELOPMENT_IDS',
        'rows':len(frame),'series':len(pieces),'comparisons':comparisons,'max_abs_error':maximum,'tolerance':1e-12,
        'official_source_sha256':sha256(ROOT/'vendor/official/scoring.py'),'split_sha256':split.digest,
        'selection_rule':'ascending original ID; fixed count, no outcome selection','LOCAL_FINAL_SEAL_OPENED':False,'reduced_scored_uses':0}
    write_json(ROOT/'artifacts/validation/SCORER_PARITY_REAL.json',report);write_json(ROOT/'SCORER_PARITY_REAL.json',report)
    prefix={'status':'PASS' if max(prefix_errors)==0 else 'FAIL','feature_hash':feature_version(),'feature_max_abs_diff':max(prefix_errors),
        'suffix_length_changed':True,'suffix_values_changed':True,'series':len(prefix_errors),'scope':'REAL_DEVELOPMENT'}
    write_json(ROOT/'artifacts/validation/PREFIX_INVARIANCE.json',prefix)
    write_json(ROOT/'REDUCED_TEST_USAGE_LEDGER.json',{'max_scored_uses':2,'scored_uses':0,'entries':[],
        'file_byte_hash_and_parquet_footer_only_so_far':True,'reduced_target_rows_selected':0,'tuning_from_reduced_forbidden':True})
    print(report,flush=True);print(prefix,flush=True)
    if report['status']!='PASS' or prefix['status']!='PASS':raise RuntimeError('Real admission parity failed')

if __name__=='__main__':main()
