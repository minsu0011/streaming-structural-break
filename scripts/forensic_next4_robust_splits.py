"""Fixed SCALE slices within already-completed ALT/H fits, without refitting."""
from pathlib import Path
import sys
import argparse
import hashlib
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.robustness import frozen_predictions
from src.next4.diagnostics import scale_directions
from src.next.data import prepare_dev
from src.next.bootstrap import PreparedAUC,paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args()
    pp=BASE/'diagnostic_policy/ALT_SCALE_DIAGNOSTIC_POLICY.json';policy=read_json(pp)
    assert args.candidate in policy['candidates']
    for name,sha in policy['sources_sha256'].items():assert digest(ROOT/name)==sha
    source=BASE/'robustness'/args.candidate;done=read_json(source/'RESULTS.json');assert done['status']=='PASS'
    out=BASE/'robustness_forensic'/args.candidate;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(pp),'robustness_result_sha256':digest(source/'RESULTS.json'),'new_fits':0},immutable=True)
    data=prepare_dev(ROOT);metadata=data.rows();set_num_threads(2)
    mp=ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    assert digest(mp)==read_json(mp.parent/'REGIME_ASSIGNMENT_LOCK.json')['files_sha256'][mp.name]
    series=pd.read_parquet(mp).set_index('dataset_id');data.guard.admit(series.index.to_numpy(),purpose='NEXT4 ALT/H SCALE diagnostic')
    directions,direction_proof=scale_directions(data,series);fold_rows=[];summaries=[]
    for split in policy['splits']:
        new={scope:[] for scope in policy['scopes']};old={scope:[] for scope in policy['scopes']}
        support={scope:[] for scope in policy['scopes']}
        for fold in range(5):
            directory=source/split;receipt=read_json(directory/f'fold_{fold}.json');assert receipt['status']=='PASS'
            for name in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy'):
                assert digest(directory/name)==receipt['files_sha256'][name]
            rows=np.load(directory/f'fold_{fold}_global_rows.npy',allow_pickle=False)
            p=np.load(directory/f'fold_{fold}.npy',allow_pickle=False);baseline,proof=frozen_predictions(data,split,fold,rows)
            ids,y,t=(metadata[k][rows] for k in ('dataset_id','target','time_online'));meta=series.reindex(ids);null=meta.tau.to_numpy()<0
            code,counts=draw_group_multiplicities(ids,2000,policy['bootstrap_seed']+fold)
            masks={'SCALE':(meta.signature.to_numpy()=='SCALE_DOMINANT')|null,
                'observed_scale_decrease':np.array([directions.get(int(sid))=='decrease' for sid in ids])|null,
                'observed_scale_increase':np.array([directions.get(int(sid))=='increase' for sid in ids])|null}
            for scope,mask in masks.items():
                cache=BASE/'robustness_forensic/frozen_BL80'/split;cache.mkdir(parents=True,exist_ok=True)
                name=f'fold_{fold}_{scope}.npy';identity={'policy_sha256':digest(pp),'rows_sha256':hashlib.sha256(rows[mask].tobytes()).hexdigest(),
                    'counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),'split':split,'fold':fold,'scope':scope}
                if (cache/(name+'.json')).exists():
                    prior=read_json(cache/(name+'.json'));assert prior['identity']==identity and digest(cache/name)==prior['sha256']
                    a=np.load(cache/name,allow_pickle=False)
                else:
                    a=PreparedAUC.create(y[mask],baseline['BL80'][mask],t[mask],code[mask],counts.shape[1]).evaluate(counts) if mask.any() else np.full(2001,.5)
                    np.save(cache/name,a,allow_pickle=False)
                    atomic_json(cache/(name+'.json'),{'identity':identity,'sha256':digest(cache/name),'reference_sources_sha256':proof},immutable=True)
                b=PreparedAUC.create(y[mask],p[mask],t[mask],code[mask],counts.shape[1]).evaluate(counts) if mask.any() else np.full(2001,.5)
                new[scope].append(b);old[scope].append(a)
                positive_series=len(np.unique(ids[mask&(y==1)]));negative_series=len(np.unique(ids[mask&null]))
                support[scope].append((positive_series,negative_series))
                np.save(out/f'{split}_fold_{fold}_{scope}.npy',b,allow_pickle=False)
                fold_rows.append({'candidate':args.candidate,'split':split,'fold':fold,'scope':scope,'ts_auc':b[0],
                    'reference_ts_auc':a[0],'delta':b[0]-a[0],'positive_series':positive_series,'null_series':negative_series,
                    'scored_rows':int(mask.sum()),'empty_scope':not bool(mask.any())})
            progress('SCALE replication on existing ALT/H fits',args.candidate+' '+split+' fold '+str(fold)+'; no refit')
            print('NEXT4 ALT SCALE',args.candidate,split,fold,flush=True)
        for scope in policy['scopes']:
            a=np.asarray(old[scope]).T;b=np.asarray(new[scope]).T;positives=sum(s[0] for s in support[scope]);nulls=sum(s[1] for s in support[scope])
            summaries.append({'candidate':args.candidate,'split':split,'scope':scope,'candidate_mean':float(b[0].mean()),
                'reference_mean':float(a[0].mean()),'positive_series':positives,'null_series':nulls,
                'supported_positive_series':positives>=100,'folds_with_positive_series':sum(s[0]>0 for s in support[scope]),
                **paired_summary(a,b)})
    pd.DataFrame(fold_rows).to_csv(out/'FOLD_SCALE.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'PAIRED_SCALE.csv',index=False);data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':args.candidate,'new_fits':0,'seal_rows':0,
        'summaries':summaries,'directional_definition':direction_proof,'original_primary_gate_overridden':False,
        'scope':'Existing DEV signatures inside already-completed fixed ALT/H predictions. No new data, fitted model, split, label definition or seed. Alternative-split agreement does not replace the primary paired CI.',
        'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    print('NEXT4 ALT/H SCALE DIAGNOSTICS COMPLETE',args.candidate,flush=True)


if __name__=='__main__':main()
