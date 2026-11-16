"""Lock only the admitted V007/Q005 pair before any combined DEV score."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,sha256,utc_now


if __name__=='__main__':
    path = ROOT/'configs/next_late_pair_candidates.json'
    if path.exists():
        print('Late variance-pair grid already frozen')
    else:
        left,right = 'V007_VARIANCE_FORECAST','Q005_VARIANCE_NO_MEMORY'
        evidence_path = ROOT/'artifacts/next/pair_complementarity'/(left+'__'+right)/'RESULTS.json'
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        if evidence['status']!='PASS' or not evidence['complementarity']['eligible_for_predeclared_late_combination'] or evidence['averaged_predictions_formed']:
            raise RuntimeError('A valid uncombined complementarity diagnostic is required')
        candidates = [{'experiment_id':eid,'parent_id':right,'generation':16,'phase':'LATE_SAME_BANK_BLEND',
            'left':left,'right':right,'left_weight':weight,'right_weight':1.-weight,
            'hypothesis':'Memory-retaining V007 gains more on scale/mixed breaks; memory-removed Q005 gains on dependence and some H regimes. Fixed averaging may reduce their observed complementary errors.',
            'change':'Two frozen learners share their identical causal feature bank; float32-quantize each probability before the fixed float64 average and final float32 cast.',
            'risk':'Adaptive DEV nomination; shared training data and feature bank limit independent evidence. Reject marginal improvements that do not justify a second forest.'}
            for eid,weight in [('L101_VARIANCE_PAIR_50_50',.5),('L102_VARIANCE_PAIR_67_33',.67),('L103_VARIANCE_PAIR_33_67',.33)]]
        write_json(path,{'status':'LOCKED','created_utc':utc_now(),'not_before_elapsed_seconds':28800,
            'session_start_authority':'artifacts/next/SESSION_NEXT.json','complementarity_sha256':sha256(evidence_path),
            'observed_rank_correlation':evidence['complementarity']['mean_within_time_rank_correlation'],
            'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
            'thresholds_splits_seeds_unchanged':True,'marginal_mean_gain_vs_best_component_min':.0005,
            'marginal_gate_rationale':'An extra forest needs at least +.0005 beyond the better component before final-role consideration; a tiny +.0002 improvement has low complexity value. This extra criterion is fixed before combined predictions are formed.',
            'screen':'Use matching preexisting 3000-series component screen OOFs and the original fixed admission threshold. No full-CV entry after a failed screen.',
            'authority':'After the late-phase time gate, retain all three weights; no continuous optimization or weight selection using seal/reduced/public evidence.',
            'component_model_changes_permitted':False,'candidates':candidates})
        print('FROZEN LATE PAIR',len(candidates),'not before elapsed 8 hours')
