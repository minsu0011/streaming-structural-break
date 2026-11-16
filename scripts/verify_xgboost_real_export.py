"""Exact fixed-model latency experiment; no refit and no held-out diagnostic data."""
import sys,json,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.compact_xgboost import export_xgboost,implementation_hash
from src.models.learners import ModelBundle
from src.features.streaming import FEATURE_NAMES,StreamingFeatureState
from src.features.cache import read_fold
from src.validation.splits import FrozenSplit
from src.data.loader import load_training
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json,utc_now,sha256
from scripts.verify_engineering import benchmark

def main():
    eid='M5_xgboost_ABCD_S0_439e0c7e5382'
    oof=ROOT/'data/processed/oof'/eid
    manifest=json.loads((oof/'MANIFEST.json').read_text(encoding='utf-8'))
    cache=ROOT/manifest['cache_directory'];split=FrozenSplit(ROOT/'artifacts/validation')
    sample=next(load_training(ROOT/'data/raw',split=split));h=sample.historical.astype(np.float32)
    points=np.tile(sample.online.astype(np.float32),max(1,3000//len(sample.online)+1))[:3000]
    results=[]
    for fold in range(5):
        model=ModelBundle.load(ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib')
        frame=read_fold(cache,fold,split=split)
        features=frame.loc[:,FEATURE_NAMES].to_numpy(dtype=np.float32);x=features[:,model.columns]
        native=model.predict(features).astype(np.float32);tree=export_xgboost(model.estimator);compact=tree.predict(x)
        first=ts_auc(frame.target,native,frame.time_online);second=ts_auc(frame.target,compact,frame.time_online)
        native_model=model
        def original():
            state=StreamingFeatureState(h)
            return lambda point:native_model.predict_one(state.update_and_get(point))
        native_time=benchmark(original,points[:1000],repeats=3)
        def exported():
            state=StreamingFeatureState(h)
            return lambda point:tree.predict_one(state.update_and_get(point)[model.columns])
        compact_time=benchmark(exported,points,repeats=3)
        results.append({'fold':fold,'rows':len(frame),'native_ts_auc':first,'compact_ts_auc':second,'ts_auc_abs_diff':abs(first-second),
            'prediction_max_abs_diff':float(np.max(abs(native-compact))),'different_float32_rows':int(np.count_nonzero(native!=compact)),
            'native_latency':native_time,'compact_latency':compact_time,'native_model_sha256':sha256(ROOT/'artifacts/models/real'/eid/f'fold_{fold}.joblib')})
        print(json.dumps({k:v for k,v in results[-1].items() if k not in ('native_latency','compact_latency','native_model_sha256')}),flush=True)
    maximum=max(r['prediction_max_abs_diff'] for r in results);metric=max(r['ts_auc_abs_diff'] for r in results)
    report={'status':'PASS' if maximum<=2e-7 and metric<=1e-8 else 'FAIL','timestamp':utc_now(),'source_candidate':eid,
        'scope':'FIXED_REAL_DEV_MODELS_NO_REFIT','exporter_hash':implementation_hash(),'exporter_source':'src/models/compact_xgboost.py',
        'source_documentation':'https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html',
        'prediction_tolerance':2e-7,'metric_tolerance':1e-8,'max_prediction_abs_diff':maximum,'max_ts_auc_abs_diff':metric,
        'folds':results,'native_median_us':float(np.median([r['native_latency']['throughput_median_us_per_point'] for r in results])),
        'compact_median_us':float(np.median([r['compact_latency']['throughput_median_us_per_point'] for r in results])),
        'concurrent_research_running':True,'final_isolated_timing_required':True,'production_model_changed':False,'seal_opened':False,'reduced_uses':0}
    write_json(ROOT/'artifacts/real/XGBOOST_EXPORT_REAL.json',report);print(report['status'],flush=True)

if __name__=='__main__':main()
