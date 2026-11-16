"""Four preregistered B104/C205 controls, with every quality gate against frozen J102."""
from pathlib import Path
import sys
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.next2.io import model_class
from src.next2.blend import FixedBlendBundle, combine, WEIGHTS, source_hash
from src.next2.session import checkpoint
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json, sha256, utc_now
from scripts.run_next2_research import export

A = 'B104_LONG_VARIANCE_NO_GLR_MEMORY'
C = 'C205_DEPENDENCE'
J = 'J102_SERIAL_VARIANCE_512'


def main():
    nomination = ROOT/'configs/next2/ROBUST_ANCHOR_BLEND_LOCK.json'
    identity = {'anchor': A, 'complement': C, 'quality_reference': J, 'weights': list(WEIGHTS),
        'combination_sha256': source_hash(), 'script_sha256': sha256(Path(__file__)),
        'hypothesis_sha256': sha256(ROOT/'docs/NEXT2_ROBUST_ANCHOR_BLEND_HYPOTHESIS.md'),
        'policy_sha256': sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'), 'new_fits': 0,
        'source_report_sha256': {f'artifacts/{g}/experiments/{eid}/RESULTS.json': sha256(ROOT/f'artifacts/{g}/experiments/{eid}/RESULTS.json')
                                 for g, eid in [('next',A),('next',J),('next2',C)]}}
    if nomination.exists():
        assert json.loads(nomination.read_text())['identity'] == identity
    else:
        write_json(nomination, {'created_utc': utc_now(), 'identity': identity, 'prediction_combination_started': False})
    data = prepare_dev(ROOT)
    ac, ar, metadata, ap, ah = load_full_oof(data, A)
    cc, cr, cm, cp, ch = load_full_oof(data, C)
    jr = json.loads((ROOT/f'artifacts/next/experiments/{J}/RESULTS.json').read_text())
    assert cr['status'] == 'COMPLEMENT_ELIGIBLE_FIXED_BLEND'
    for key in metadata:
        np.testing.assert_array_equal(metadata[key], cm[key])
    for weight in WEIGHTS:
        eid = 'BL2_B104_C205_DEPENDENCE_' + str(int(round(100*weight)))
        out = ROOT/'artifacts/next2/experiments'/eid
        folder = out/'full'
        folder.mkdir(parents=True, exist_ok=True)
        candidate = {'experiment_id':eid,'parent_id':A,'generation':'NEXT2_ROBUST_ANCHOR_BLEND_V1',
            'phase':'FIXED_COMPLEMENT_BLEND','hypothesis':'Robust scale anchor plus independent conditional dependence may preserve complementary improvements.',
            'change':f'Fixed B104/C205 weights {weight}/{1-weight}', 'counterexample_target':'scale decrease and dependence',
            'engine_config':ac['engine_config'],'base':'none','groups':[],'learner':'FIXED_BLEND','weighting':'NO_META_FIT','params':{},
            'blend':{'primary':A,'complement':C,'primary_weight':weight,'implementation_sha256':source_hash()},
            'nomination_sha256':sha256(nomination),'quality_reference':J}
        if (out/'RESULTS.json').exists():
            assert json.loads((out/'RESULTS.json').read_text())['candidate'] == candidate
            print('NEXT2 ROBUST BLEND RETAINED',eid,flush=True)
            continue
        write_json(folder/'POLICY.json',{'candidate':candidate,'policy_sha256':identity['policy_sha256'],
            'component_sources':{'anchor':ah,'complement':ch},'script_sha256':sha256(Path(__file__))})
        p = combine(ap,cp,weight)
        scores, cv = [], []
        for fold in range(5):
            rows = np.flatnonzero(metadata['fold']==fold)
            value = ts_auc(metadata['target'][rows],p[rows],metadata['time_online'][rows])
            left = model_class(ac).load(ROOT/f'artifacts/next/experiments/{A}/full/fold_{fold}.joblib')
            right = model_class(cc).load(ROOT/f'artifacts/next2/experiments/{C}/full/fold_{fold}.joblib')
            model = FixedBlendBundle(left,right,weight)
            model.save(folder/f'fold_{fold}.joblib')
            loaded = FixedBlendBundle.load(folder/f'fold_{fold}.joblib')
            sid = metadata['dataset_id'][rows[0]]
            h,o,_ = data.get(sid)
            state = loaded.make_state(h)
            actual = np.asarray([loaded.predict_one(state.update(v)) for v in o],dtype=np.float32)
            np.testing.assert_array_equal(actual,p[metadata['dataset_id']==sid])
            np.save(folder/f'fold_{fold}.npy',p[rows],allow_pickle=False)
            np.save(folder/f'fold_{fold}_global_rows.npy',rows,allow_pickle=False)
            files = [f'fold_{fold}{suffix}' for suffix in ('.joblib','.npy','_global_rows.npy')]
            write_json(folder/f'fold_{fold}.json',{'status':'PASS','new_fits':0,'component_folds_match':True,
                'live_cached_parity':True,'files_sha256':{n:sha256(folder/n) for n in files}})
            scores.append(value)
            cv.append({'experiment_id':eid,'scope':'full','fold':fold,'ts_auc':value,'validation_rows':len(rows),'seal_rows':0})
        full = {'status':'COMPLETE','scope':'full','fold_scores':scores,'mean':float(np.mean(scores)),
            'median':float(np.median(scores)),'worst_fold':min(scores),'cv_rows':cv,'new_fits':0,
            'feature_names':[f'anchor::{n}' for n in ar['full']['feature_names']]+[f'complement::{n}' for n in cr['full']['feature_names']]}
        delta = full['mean']-jr['full']['mean']
        finalist = delta>=-.0005 and full['median']-jr['full']['median']>=-.001 and full['worst_fold']-jr['full']['worst_fold']>=-.001
        write_json(folder/'RESULTS.json',full)
        write_json(out/'RESULTS.json',{'candidate':candidate,'full':full,'delta':delta,'new_fits':0,
            'delta_vs_anchor':full['mean']-ar['full']['mean'], 'created_utc':utc_now(),
            'status':'FINALIST_DIAGNOSTICS_PENDING' if finalist else 'FULL_GATE_FAIL',
            'reason':'Numeric J102 gates passed; further checks required' if finalist else 'Fixed J102 numeric gate failed'})
        export()
        checkpoint('Robust anchor fixed blend '+eid,['Paired J102 bootstrap for admitted finalists','Matched ALT/H and synthetic diagnostics'])
        print('NEXT2 ROBUST BLEND',eid,full['mean'],delta,finalist,flush=True)


if __name__=='__main__':
    main()
