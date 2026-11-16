"""Budget reservation and ordered work queue, before any RP outer scores."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,digest,utc_now,note


def main():
    budget=read_json(BASE/'NEXT7_CANDIDATE_BUDGET.json')
    protocol=read_json(BASE/'NEXT7_PROTOCOL_LOCK.json')
    synthetic=read_json(BASE/'G1_SYNTHETIC_EVALUATION_LOCK.json')
    screens=[v[0] for v in synthetic['variants'] if v[0]!='NEXT6_HARD4_EVENT']
    full=['M_CURRENT','M_EVENT','M_SCALE','M_EVENT_SCALE','M_FULL_FIXED','I3_CURRENT_EVENT',
          'M_FULL','I4_EVENT_SCALE_TREE','M_CORE_EVENT','CURRENT_ONLY','EVENT_ONLY','SCALE_ONLY']
    assert len(screens)<=budget['screen_candidates'] and len(full)<=budget['full_5fold_candidates']
    assert not any((BASE/'outer_scores').glob('*.json'))
    save('G1_CANDIDATE_RESERVATION.json',{'status':'FROZEN','created_utc':utc_now(),
        'screen_candidates':screens,'screen_registration_authority_sha256':digest(BASE/'G1_SYNTHETIC_EVALUATION_LOCK.json'),
        'screen_preregistration_utc':synthetic['created_utc'],
        'full_scientific_candidates':full,'diagnostic_standalone_scores_counted_in_full_budget':True,
        'M_CORE_EVENT':'.9 BL80 +.1 frozen NEXT6 hard4 confirmation-only event memory',
        'PRIMARY_signal':'S1_BLOCK_LOG_MAD + C1_SOFT_3OF4, fixed before scores; core union is explicitly uncalibrated',
        'CURRENT_ONLY':'G1 current ranking score','EVENT_ONLY':'G1 ever-confirmed ranking score',
        'SCALE_ONLY':'G1 abs signed-scale ranking score',
        'learned_meta_candidates':['I3_CURRENT_EVENT','M_FULL','I4_EVENT_SCALE_TREE'],
        'I4_tree':{'learner':'LightGBM regression of y-BL80','n_estimators':24,'num_leaves':4,'max_depth':2,
            'min_child_samples':2000,'reg_lambda':20.,'learning_rate':.05,'random_state':20260908,
            'n_jobs':4,'deterministic':True,'force_col_wise':True,'verbosity':-1,
            'fit_features':['event_score','episode_count/7','first_age','last_age','confirmed_direction',
                            'signed_scale','abs_scale','scale_confirmation'],
            'prediction':'clip(BL80 + predicted residual,1e-6,1-1e-6)',
            'weights':'W3 computed on outer TRAIN inner-OOF targets only'},
        'full_candidates_reserved':len(full),'screen_candidates_reserved':len(screens),
        'remaining_full_slots':budget['full_5fold_candidates']-len(full),
        'new_fitted_families_so_far':['F1_SIGNED_SCALE','F2_FRESH_CONFIRMATION','F3_CONFIRMED_EVENT_MEMORY','F5_CAUSAL_NESTED_INTEGRATION'],
        'threshold_or_weight_sweep':False})
    save('runtime/CANDIDATE_USAGE.json',{'screen_reserved':len(screens),'full_reserved':len(full),
        'screen_completed':len(screens),'full_completed':0,'learned_meta_reserved':3,
        'updated_utc':utc_now(),'hard_limits':protocol['budgets']},False)
    jobs=[]
    script='scripts/next7/fit_nested_baseline.py'
    for fold in (3,4):
        jobs.append({'id':f'NESTED_BASE_FOLD_{fold}','status':'READY','script':script,
            'source_sha256':digest(ROOT/script),'arguments':['--fold',str(fold)],
            'gates':['PREFLIGHT_NEXT7.json','NEXT7_PROTOCOL_LOCK.json','runtime/CORE_TEST_GATE.json'],
            'safe_retry':'Completed immutable parts reused; partial fits cause quarantine without overwrite'})
    save('runtime/TASK_QUEUE.json',jobs,False)
    note('G1 candidates reserved and queue ready','12 full slots reserved including standalone diagnostics;4 slots retained for structural follow-up','F5_CAUSAL_NESTED_INTEGRATION')


if __name__=='__main__':
    main()
