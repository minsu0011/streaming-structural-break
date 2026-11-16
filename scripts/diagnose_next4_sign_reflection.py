"""One frozen input-orientation stress on original held-out models; never a recipe."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import csv
import hashlib
from pathlib import Path
import sys
import numpy as np
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress

CANDIDATES=('B1_SPLIT_ENERGY_BRANCHES','B2_LOG_ENERGY_SEMITAIL','D2_DISCRETE_SCALE_MIXTURE')
POLICY=BASE/'diagnostic_policy/SIGN_REFLECTION_POLICY.json'


def sources():
    names=['scripts/diagnose_next4_sign_reflection.py','src/next4/model.py','src/next3/frozen.py',
        'src/next4/frozen.py','src/next2/shared_blend_runtime.py','src/next2/counterexamples.py',
        'src/next2/blend.py','src/next/bootstrap.py','src/validation/group_bootstrap.py','src/scoring/ts_auc.py']
    return {name:digest(ROOT/name) for name in names}


def register():
    from src.next4.scheduler import register_job
    model_files=[]
    for fold in range(5):
        model_files.extend([ROOT/'artifacts/next/experiments/J102_SERIAL_VARIANCE_512/full'/f'fold_{fold}.joblib',
            ROOT/'artifacts/next2/experiments/C205_DEPENDENCE/full'/f'fold_{fold}.joblib'])
        model_files.extend(BASE/'experiments'/cid/'full'/f'fold_{fold}_expert.joblib' for cid in CANDIDATES)
    atomic_json(POLICY,{'created_utc':utc_now(),'sources_sha256':sources(),
        'candidate_ids':list(CANDIDATES),'reference_models':['J102','C205','BL80'],
        'model_files_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in model_files},
        'transformation':'Exactly H -> -H and each current online point -> -point, once; no other transformations.',
        'population':'Original 8000 DEV series with original five held-out fitted models, unchanged labels/tau and original fold assignment.',
        'hypothesis':'An invertible sign reflection preserves structural-change locations. Measure dependence of frozen learned scores on observation orientation.',
        'selection_context':'Post-hoc mechanistic follow-up after original real-CV and B2 memory results, prospectively frozen before any reflected prediction.',
        'score_models':['J102','C205','BL80',*CANDIDATES],
        'paired_models':['BL80',*CANDIDATES],
        'multiplicities':'Exactly original 2000 whole-series bootstrap counts per fold, original seed 20260908 + fold; no new random draws.',
        'inference_inputs':'H and current online scalar only. IDs, labels, tau and lengths are evaluation coordinates only.',
        'new_fits':0,'new_candidate_identities':0,'promotion_authority':False,
        'forbidden_followups':['No mirrored ensemble','No sign-augmentation fit','No selection based on reflected scores','No altered blend weights'],
        'stop_rule':'One transformation and this fixed model list; report the complete results regardless of direction.'},immutable=True)
    register_job({'id':'SR001_SIGN_REFLECTION','kind':'validation','priority':180,
        'depends_on_success':[],'argv':['scripts/diagnose_next4_sign_reflection.py','--run'],
        'description':'Fixed sign reflection of original DEV with original held-out models; no fitting or new recipe',
        'retry_safe':False,'policy_sha256':digest(POLICY)})
    progress('Input-orientation stress registered','Original five folds, three fixed existing candidates, one reflection and unchanged bootstrap multiplicities; no selection authority')
    print('NEXT4 SIGN REFLECTION REGISTERED',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    assert args.register!=args.run
    if args.register:return register()
    from src.next.data import prepare_dev
    from src.next3.frozen import fold_artifact,J102,C205
    from src.next4.frozen import fold_predictions
    from src.next4.model import FeatureModel
    from src.next2.blend import FixedBlendBundle,combine,source_hash as blend_hash
    from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
    from src.next2.counterexamples import predict
    from src.next.bootstrap import PreparedAUC,paired_summary
    from src.validation.group_bootstrap import draw_group_multiplicities
    from src.scoring.ts_auc import ts_auc
    policy=read_json(POLICY);assert policy['sources_sha256']==sources()
    for name,sha in policy['model_files_sha256'].items():assert digest(ROOT/name)==sha
    out=BASE/'sign_reflection';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(POLICY)},immutable=True)
    set_num_threads(2);data=prepare_dev(ROOT);meta=data.rows();score_rows=[];point_rows=[];proofs=[]
    original_draws={key:[] for key in policy['paired_models']};reflected_draws={key:[] for key in policy['paired_models']}
    checked=0
    for fold in range(5):
        rows=np.flatnonzero(meta['fold']==fold);original,proof=fold_predictions(data,fold,'full',rows)
        j,_,jr,_=fold_artifact(data,J102,fold,'full');c,_,cr,_=fold_artifact(data,C205,fold,'full')
        np.testing.assert_array_equal(jr,rows);np.testing.assert_array_equal(cr,rows)
        prepared=PreparedSharedBlendPredictor(FixedBlendBundle(j,c,.8,implementation_sha256=blend_hash()),resident=False)
        models={cid:FeatureModel.load(BASE/'experiments'/cid/'full'/f'fold_{fold}_expert.joblib') for cid in CANDIDATES}
        for cid in CANDIDATES:
            directory=BASE/'experiments'/cid/'full';r=read_json(directory/f'fold_{fold}.json')
            for name in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy'):
                assert digest(directory/name)==r['files_sha256'][name]
            np.testing.assert_array_equal(np.load(directory/f'fold_{fold}_global_rows.npy',allow_pickle=False),rows)
            original[cid]=np.load(directory/f'fold_{fold}.npy',allow_pickle=False)
        reflected={key:np.empty(len(rows),dtype=np.float32) for key in policy['score_models']}
        series_indices=np.flatnonzero(data.series['fold']==fold);cursor=0
        for ordinal,index in enumerate(series_indices):
            sid=int(data.ids[index]);h,o,_=data.get(sid);end=cursor+len(o)
            assert np.all(meta['dataset_id'][rows[cursor:end]]==sid)
            hr=-np.asarray(h);orr=-np.asarray(o)
            p={'J102':predict(prepared.primary,hr,orr),'C205':predict(prepared.complement,hr,orr)}
            p['BL80']=combine(p['J102'],p['C205'],.8)
            for cid,model in models.items():
                state=model.make_state(hr);expert=np.asarray(model.predict(state.replay(orr)),dtype=np.float32)
                p[cid]=combine(expert,p['C205'],.8) if cid.startswith('B') else combine(p['BL80'],expert,.9)
                if ordinal<2:
                    state=model.make_state(hr)
                    scalar=np.asarray([model.predict_one(state.update(float(v))) for v in orr],dtype=np.float32)
                    np.testing.assert_array_equal(scalar,expert);checked+=1
            for key,values in p.items():reflected[key][cursor:end]=values
            if ordinal<2:
                jp=predict(prepared.primary,h,o);cp=predict(prepared.complement,h,o);bp=combine(jp,cp,.8)
                for key,values in [('J102',jp),('C205',cp),('BL80',bp)]:
                    np.testing.assert_array_equal(values,original[key][cursor:end]);checked+=1
                for cid,model in models.items():
                    ep=np.asarray(model.predict(model.make_state(h).replay(o)),dtype=np.float32)
                    values=combine(ep,cp,.8) if cid.startswith('B') else combine(bp,ep,.9)
                    np.testing.assert_array_equal(values,original[cid][cursor:end]);checked+=1
            cursor=end
            if (ordinal+1)%100==0:
                progress('Fixed sign-reflection replay',f'Original fold {fold}: {ordinal+1}/{len(series_indices)} series; no fitting')
                print('NEXT4 REFLECTION REPLAY',fold,ordinal+1,len(series_indices),flush=True)
        assert cursor==len(rows)
        ids,y,t=(meta[k][rows] for k in ('dataset_id','target','time_online'))
        code,counts=draw_group_multiplicities(ids,2000,20260908+fold)
        old=read_json(BASE/'bootstrap/frozen_BL80'/f'fold_{fold}.json')
        assert old['identity']['counts_sha256']==hashlib.sha256(counts.tobytes()).hexdigest()
        proofs.append({'fold':fold,'original_multiplicities_sha256':old['identity']['counts_sha256'],'reference_proof':proof})
        for key,values in reflected.items():
            assert np.isfinite(values).all();np.save(out/f'fold_{fold}_{key}_REFLECTED.npy',values,allow_pickle=False)
            delta=np.asarray(values,dtype=np.float64)-original[key]
            point_rows.append({'model':key,'fold':fold,'series':len(series_indices),'points':len(values),
                'mean_absolute_score_change':float(np.abs(delta).mean()),'maximum_absolute_score_change':float(np.abs(delta).max()),
                'mean_signed_score_change':float(delta.mean()),'exactly_equal_fraction':float((delta==0).mean())})
            before=ts_auc(y,original[key],t);after=ts_auc(y,values,t)
            score_rows.append({'model':key,'fold':fold,'original_ts_auc':before,'reflected_ts_auc':after,
                'reflection_delta':after-before,'reflected_bl80_ts_auc':ts_auc(y,reflected['BL80'],t),
                'new_fit':False,'promotion_authority':False})
            if key in original_draws:
                prior=BASE/'bootstrap'/('frozen_BL80' if key=='BL80' else key)/f'fold_{fold}.npy'
                receipt=read_json(prior.with_suffix('.json')) if key=='BL80' else read_json(prior.parent/'RESULTS.json')
                expected=receipt['sha256'] if key=='BL80' else receipt['files_sha256'][prior.name]
                assert digest(prior)==expected;original_draws[key].append(np.load(prior,allow_pickle=False))
                draws=PreparedAUC.create(y,values,t,code,counts.shape[1]).evaluate(counts)
                assert abs(float(draws[0])-after)<=1e-14;reflected_draws[key].append(draws)
            print('NEXT4 REFLECTION SCORE',key,fold,before,after,flush=True)
        progress('Fixed reflection statistics complete','Fold '+str(fold)+' original group counts; one unchanged reflection')
    summaries={};rbl=np.asarray(reflected_draws['BL80']).T
    for key in original_draws:
        before=np.asarray(original_draws[key]).T;after=np.asarray(reflected_draws[key]).T
        summaries[key]={'reflected_minus_original':paired_summary(before,after),
            'candidate_minus_reflected_BL80':paired_summary(rbl,after)}
        np.savez_compressed(out/(key+'_AGGREGATE_DRAWS.npz'),original=before,reflected=after)
    for name,records in [('FOLD_SCORES.csv',score_rows),('POINT_SENSITIVITY.csv',point_rows)]:
        with (out/name).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'policy_sha256':digest(POLICY),
        'summaries':summaries,'series':8000,'online_points':len(meta['target']),'folds':5,
        'scalar_and_original_cached_parity_checks':checked,'new_fits':0,'new_candidate_identities':0,
        'new_random_seeds_or_bootstrap_counts':0,'seal_rows':0,'promotion_authority':False,
        'interpretation':'A post-hoc orientation stress of fixed original models, not an additional deployable recipe or independent validation dataset. Conditional paired intervals reuse original counts and do not adjust for adaptive DEV use.',
        'proofs':proofs,'source_sha256':sources(),'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    print('NEXT4 FIXED SIGN REFLECTION COMPLETE',flush=True)


if __name__=='__main__':main()
