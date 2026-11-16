"""Independent real-series live/OOF checks for an explicit AR-order experiment."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.ar_order_calibrated import AROrderCalibratedBundle
from src.validation.splits import FrozenSplit
from src.data.loader import load_training
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);parser.add_argument('--series-per-fold',type=int,default=16);args=parser.parse_args()
    report_path=ROOT/'artifacts/real'/f'{args.candidate}.json';candidate=json.loads(report_path.read_text())
    if candidate['status']!='COMPLETE':raise RuntimeError('Complete CV result required')
    split=FrozenSplit(ROOT/'artifacts/validation');oof=ROOT/'data/processed/oof'/args.candidate;manifest=json.loads((oof/'MANIFEST.json').read_text())
    if manifest['split_sha256']!=split.digest:raise RuntimeError('OOF split mismatch')
    models=[AROrderCalibratedBundle.load(ROOT/'artifacts/models/real'/args.candidate/f'fold_{fold}.joblib') for fold in range(5)]
    predictions=[np.load(oof/f'fold_{fold}.npy',allow_pickle=False) for fold in range(5)]
    positions=[]
    for fold in range(5):
        frame=pd.read_parquet(ROOT/manifest['cache_directory']/f'fold_{fold}.parquet',columns=['dataset_id','time_online'])
        positions.append(frame.groupby('dataset_id',sort=False).indices)
    counts=[0]*5;points=[0]*5
    for item in load_training(ROOT/'data/raw',split=split):
        fold=split.assignment(item.dataset_id)[1]
        if counts[fold]>=args.series_per_fold:continue
        stream=models[fold].make_state(item.historical.astype(np.float32))
        actual=np.array([models[fold].predict_one(stream.update_and_get(p)) for p in item.online.astype(np.float32)],dtype=np.float32)
        np.testing.assert_array_equal(actual,predictions[fold][positions[fold][item.dataset_id]])
        counts[fold]+=1;points[fold]+=len(actual)
        if min(counts)>=args.series_per_fold:break
    report={'status':'PASS','timestamp':utc_now(),'candidate':args.candidate,'fold_series':counts,'fold_points':points,
        'live_cached_prediction_max_abs_diff':0.,'bitwise_equal':True,'full_development_replay':False,
        'selection':'first ascending DEV series in each original validation fold; no target-based probe choice',
        'oof_manifest_sha256':sha256(oof/'MANIFEST.json'),'split_sha256':split.digest,
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    path=ROOT/'artifacts/real/contracts'/f'{args.candidate}_live_oof.json';write_json(path,report)
    candidate['live_cached_prediction_max_abs_diff']=0.;candidate['independent_live_validation']=str(path.relative_to(ROOT));write_json(report_path,candidate)
    print('AR ORDER LIVE PASS',args.candidate,sum(counts),'series',sum(points),'points',flush=True)


if __name__=='__main__':main()
