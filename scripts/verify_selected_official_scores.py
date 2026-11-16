"""Run the unchanged official scorer on complete existing DEV OOF folds."""
import argparse,contextlib,io,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.verify_scorer import load_official,official_score
from src.scoring.ts_auc import score_frames
from src.validation.splits import FrozenSplit,assert_development_ids
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',action='append',required=True);args=parser.parse_args()
    split=FrozenSplit(ROOT/'artifacts/validation');official=load_official()
    state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text())
    for eid in args.candidate:
        result=next(r for r in state['results'] if r['experiment_id']==eid)
        oof=ROOT/'data/processed/oof'/eid;manifest=json.loads((oof/'MANIFEST.json').read_text());cache=ROOT/manifest['cache_directory']
        cache_manifest=json.loads((cache/'MANIFEST.json').read_text())
        identity={str(k):sha256(oof/f'fold_{k}.npy') for k in range(5)}
        path=ROOT/'artifacts/real/scorer_full_oof'/eid/'RESULTS.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old['status']!='PASS' or old['oof_file_sha256']!=identity or old['official_source_sha256']!=sha256(ROOT/'vendor/official/scoring.py') or old['local_source_sha256']!=sha256(ROOT/'src/scoring/ts_auc.py'):
                raise RuntimeError('Existing full-score evidence identity differs')
            print('REUSE FULL OFFICIAL PARITY',eid,flush=True);continue
        if manifest['split_sha256']!=split.digest or cache_manifest['split_sha256']!=split.digest:raise AssertionError('OOF split differs')
        rows=[]
        for fold in range(5):
            frame_path=cache/f'fold_{fold}.parquet'
            if sha256(frame_path)!=cache_manifest['fold_sha256'][str(fold)]:raise AssertionError('Feature cache bytes changed')
            frame=pd.read_parquet(frame_path,columns=['dataset_id','time_online','target']);assert_development_ids(frame.dataset_id.unique(),split)
            expected=set(split.frame.loc[(split.frame.role=='DEVELOPMENT_POOL')&(split.frame.fold==fold),'dataset_id'])
            if set(frame.dataset_id.unique())!=expected:raise AssertionError('Incomplete fold coverage')
            index=pd.MultiIndex.from_arrays([frame.dataset_id.to_numpy(),frame.time_online.to_numpy()],names=['id','time'])
            target=pd.DataFrame({'target':frame.target.to_numpy()},index=index).sort_index()
            prediction=pd.DataFrame({'prediction':np.load(oof/f'fold_{fold}.npy',allow_pickle=False)},index=index).sort_index()
            saved_loaders=official._load_prediction,official._load_y_test
            try:
                official._load_prediction=lambda _:prediction.copy();official._load_y_test=lambda _:target.copy()
                with contextlib.redirect_stdout(io.StringIO()):official.check('unused','unused')
            finally:official._load_prediction,official._load_y_test=saved_loaders
            local=score_frames(prediction,target);reference=official_score(official,prediction,target)
            maximum=max(abs(local-reference),abs(local-result['fold_scores'][fold]))
            if maximum>=1e-12:raise AssertionError('Complete official/local/saved DEV score parity failed')
            rows.append({'fold':fold,'rows':len(prediction),'official_ts_auc':reference,'local_ts_auc':local,'saved_ts_auc':result['fold_scores'][fold],
                'max_abs_diff':maximum,'official_index_dtype_finiteness_check':'PASS'})
            print('FULL OFFICIAL PARITY',eid,fold,len(prediction),maximum,flush=True)
        report={'status':'PASS','timestamp':utc_now(),'candidate':eid,'folds':rows,'total_rows':sum(r['rows'] for r in rows),
            'max_abs_diff':max(r['max_abs_diff'] for r in rows),'official_source_sha256':sha256(ROOT/'vendor/official/scoring.py'),
            'local_source_sha256':sha256(ROOT/'src/scoring/ts_auc.py'),'script_sha256':sha256(__file__),'oof_file_sha256':identity,'split_sha256':split.digest,
            'scope':'Entire saved DEV OOF, same frozen folds and float32 probabilities. Only official file loaders are replaced with in-memory frames; original check/score logic is unchanged. No new model, metric, reduced data or seal performance.',
            'index_convention':'Canonical sorted (id,time_online) index; official per-series cumcount yields exactly those online coordinates.',
            'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
        write_json(path,report)


if __name__=='__main__':main()
