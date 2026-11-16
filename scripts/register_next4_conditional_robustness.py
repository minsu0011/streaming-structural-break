"""Freeze further validation of existing SCALE-admitted base recipes, without variants."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress


def main():
    candidates=['A2_MAGNITUDE_PORTMANTEAU','C3_SHAPE_RATIO_DISTRIBUTION','D2_DISCRETE_SCALE_MIXTURE','D3_HAZARD_MIXTURE']
    evidence={}
    for candidate in candidates:
        path=BASE/'forensic'/candidate/'RESULTS.json';r=read_json(path)
        assert r['robustness_primary_scale_admission']
        evidence[candidate]={'forensic_sha256':digest(path),'paired_scale':r['paired_scale'],
            'registration_sha256':digest(BASE/'registrations'/(candidate+'.json')),
            'full_result_sha256':digest(BASE/'experiments'/candidate/'full/RESULTS.json')}
        assert not (BASE/'synthetic/candidates'/(candidate+'_full_fold0')/'START.json').exists()
    names=['scripts/robustness_next4_base.py','src/next4/robustness.py','src/next4/cache.py','src/next4/model.py',
        'src/next/bootstrap.py','src/validation/group_bootstrap.py','src/next2/blend.py','src/scoring/ts_auc.py']
    atomic_json(BASE/'diagnostic_policy/CONDITIONAL_ROBUSTNESS_VALIDATION.json',{'created_utc':utc_now(),
        'candidates':candidates,'original_evidence':evidence,'synthetic_suite_sha256':digest(BASE/'diagnostic_policy/SYNTHETIC_POLICY.json'),
        'condition':'The already registered complete matched-full-fold-0 synthetic suite passes both fixed ARCH variance-decrease cases V3/V4 and both heavy-tail N3 decrease cases 0.7/0.5, each with strictly positive point delta. No substitute case or extra seed.',
        'if_condition_passes':'Exactly the original ALT_1/ALT_2/ALT_3/LEAVE_ONE_H_REGIME_OUT fits of the unchanged registered candidate; 20 candidate refits, zero frozen-base refits.',
        'if_condition_fails':'Stop this robustness route, retain descriptive evidence, and do not refit alternate parameters.',
        'maximum_additional_fits':80,'maximum_new_full_candidate_identities':0,
        'execution_source_sha256':{name:digest(ROOT/name) for name in names},
        'scope':'Existing candidates already pass supported real SCALE and overall-loss admission. Synthetic tests alone never admit a candidate. ALT/H are reused DEV partitions, not new independent data.',
        'final_role_limit':1,
        'robustness_role_tie_break':'Among fully validated candidates that satisfy every originally frozen robustness/causal/runtime requirement, prefer the largest original supported SCALE paired CI lower bound, then higher original primary mean gain, then candidate ID. This prioritizes stability of the real SCALE effect; synthetic scores are gates, not a tuning leaderboard.',
        'B1_existing_nomination':'B1 already has an unchanged 20-fit ALT/H nomination based on its larger SCALE point estimate; it remains subject to the same final role gates and tie break.',
        'stop_rule':'No new feature, weight, threshold, learner or split recipes. Only the four frozen candidate identities can trigger these conditional refits.'},immutable=True)
    progress('Further robustness validation conditions frozen','Four existing SCALE-admitted base recipes; execute ALT/H only after their already-fixed ARCH/tail diagnostic gates')
    print('NEXT4 CONDITIONAL ROBUSTNESS POLICY FROZEN',flush=True)


if __name__=='__main__':main()
