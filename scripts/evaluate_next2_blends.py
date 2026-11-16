"""Evaluate only prequalified complements on the four prospectively fixed weights."""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.next.candidate_io import model_class as old_model_class
from src.next2.io import model_class
from src.next2.blend import FixedBlendBundle,combine,WEIGHTS,source_hash
from scripts.run_next2_research import export
from src.next2.session import checkpoint
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--complement',required=True)
    args=parser.parse_args()
    data=prepare_dev(ROOT)
    c,comp_report,metadata,cpred,csources=load_full_oof(data,args.complement)
    if comp_report['status']!='COMPLEMENT_ELIGIBLE_FIXED_BLEND':
        raise RuntimeError('Complement must independently pass full standalone/correlation admission')
    pc,primary,pm,ppred,psources=load_full_oof(data,'J102_SERIAL_VARIANCE_512')
    for k in metadata:
        np.testing.assert_array_equal(metadata[k],pm[k])
    recipe={'created_utc':utc_now(),'primary':'J102_SERIAL_VARIANCE_512','complement':args.complement,
        'weights':list(WEIGHTS),'combination_sha256':source_hash(),'source_sha256':{'primary':psources,'complement':csources},
        'complement_nomination_sha256':sha256(ROOT/f'artifacts/next2/experiments/{args.complement}/RESULTS.json'),
        'policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),'meta_fit':False,
        'sampling':'Same original held-out series in both components; no in-sample primary predictions.'}
    nomination=ROOT/'artifacts/next2/blend_nominations'/args.complement/'POLICY.json'
    if nomination.exists():
        prior=json.loads(nomination.read_text(encoding='utf-8'))
        for key in recipe:
            if key!='created_utc':
                assert prior[key]==recipe[key]
    else:
        write_json(nomination,recipe)
    for weight in WEIGHTS:
        eid='BL2_'+args.complement+'_'+str(int(round(weight*100)))
        out=ROOT/'artifacts/next2/experiments'/eid
        out.mkdir(parents=True,exist_ok=True)
        if (out/'RESULTS.json').exists():
            print('BLEND RETAINED',eid,flush=True)
            continue
        candidate={'experiment_id':eid,'parent_id':'J102_SERIAL_VARIANCE_512','generation':'NEXT2_BLEND_V1',
            'phase':'FIXED_COMPLEMENT_BLEND','hypothesis':'Independent conditional '+c['counterexample_target']+' information may improve primary rankings.',
            'change':f'Fixed primary/complement weight {weight}/{1-weight}',
            'counterexample_target':c['counterexample_target'],'engine_config':pc['engine_config'],
            'base':'none','groups':[],'learner':'FIXED_BLEND','weighting':'NO_META_FIT','params':{},
            'blend':{'primary':'J102_SERIAL_VARIANCE_512','complement':args.complement,'primary_weight':weight,
                     'implementation_sha256':source_hash()},'nomination_sha256':sha256(nomination)}
        folder=out/'full'
        folder.mkdir(exist_ok=True)
        write_json(folder/'POLICY.json',{'candidate':candidate,'policy_sha256':recipe['policy_sha256'],
            'component_sources':recipe['source_sha256'],'script_sha256':sha256(Path(__file__))})
        prediction=combine(ppred,cpred,weight)
        scores,cv=[],[]
        for fold in range(5):
            rows=np.flatnonzero(metadata['fold']==fold)
            score=ts_auc(metadata['target'][rows],prediction[rows],metadata['time_online'][rows])
            left=old_model_class(pc).load(ROOT/f'artifacts/next/experiments/J102_SERIAL_VARIANCE_512/full/fold_{fold}.joblib')
            right=model_class(c).load(ROOT/f'artifacts/next2/experiments/{args.complement}/full/fold_{fold}.joblib')
            model=FixedBlendBundle(left,right,weight)
            model.save(folder/f'fold_{fold}.joblib')
            loaded=FixedBlendBundle.load(folder/f'fold_{fold}.joblib')
            sid=metadata['dataset_id'][rows[0]]
            h,o,_=data.get(sid); state=loaded.make_state(h)
            live=np.asarray([loaded.predict_one(state.update(point)) for point in o],dtype=np.float32)
            np.testing.assert_array_equal(live,prediction[metadata['dataset_id']==sid])
            np.save(folder/f'fold_{fold}.npy',prediction[rows],allow_pickle=False)
            np.save(folder/f'fold_{fold}_global_rows.npy',rows,allow_pickle=False)
            files=[f'fold_{fold}.joblib',f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy']
            write_json(folder/f'fold_{fold}.json',{'status':'PASS','files_sha256':{name:sha256(folder/name) for name in files},
                'live_cached_parity':True,'new_fits':0,'component_folds_match':True})
            scores.append(score)
            cv.append({'experiment_id':eid,'scope':'full','fold':fold,'ts_auc':score,'validation_rows':len(rows),'seal_rows':0})
        full={'status':'COMPLETE','scope':'full','fold_scores':scores,'mean':float(np.mean(scores)),
            'median':float(np.median(scores)),'worst_fold':min(scores),'cv_rows':cv,
            'feature_names':['primary_score','complement_score'],'new_fits':0}
        delta=full['mean']-primary['full']['mean']
        finalist=delta>=-.0005 and full['median']-primary['full']['median']>=-.001 and full['worst_fold']-primary['full']['worst_fold']>=-.001
        result={'candidate':candidate,'full':full,'delta':delta,
            'status':'FINALIST_DIAGNOSTICS_PENDING' if finalist else 'FULL_GATE_FAIL',
            'reason':'Fixed blend full gates passed; paired/ALT/regime/engineering required' if finalist else 'Fixed blend primary numeric gate failed',
            'created_utc':utc_now(),'new_fits':0,'leakage':'Only matching held-out component predictions were combined'}
        write_json(folder/'RESULTS.json',full)
        write_json(out/'RESULTS.json',result)
        export()
        checkpoint('Fixed complement blend '+eid,['Paired and alternate validation','Synthetic behavior'],
            artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
        print('NEXT2 BLEND',eid,full['mean'],delta,result['status'],flush=True)


if __name__=='__main__':
    main()
