"""A single all-DEV candidate refit, justified by fixed replication evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    candidate='B2_LOG_ENERGY_SEMITAIL'
    full=BASE/'experiments'/candidate/'full/RESULTS.json';boot=BASE/'bootstrap'/candidate/'RESULTS.json'
    replication=BASE/'robustness'/candidate/'RESULTS.json'
    assert read_json(full)['performance_numeric_admission']
    assert not read_json(boot)['performance_primary_and_bootstrap_admission']
    assert read_json(replication)['status']=='PASS'
    sources=['src/next4/diagnostic_refit.py','scripts/refit_next4_b2_diagnostic.py','src/next4/model.py',
        'src/next4/cache.py','src/next/data.py','src/next/guard.py','src/models/learners.py',
        'src/models/compact_trees.py','src/next/fitting.py']
    path=BASE/'diagnostic_policy/B2_ALL_DEV_REFIT.json'
    atomic_json(path,{'status':'FROZEN_BEFORE_DIAGNOSTIC_FIT','created_utc':utc_now(),'candidate':candidate,
        'registration_sha256':digest(BASE/'registrations'/(candidate+'.json')),
        'full_result_sha256':digest(full),'bootstrap_result_sha256':digest(boot),'replication_result_sha256':digest(replication),
        'reason':'B2 improves all primary folds and all three fixed ALT means. One all-DEV model permits training-scope-matched synthetic comparisons with the existing all-DEV canonical BL80. Primary CI failure remains unchanged.',
        'new_candidate_identity':False,'intended_candidate_fits':1,'new_frozen_base_fits':0,
        'training_recipe':'Exactly the original B2 M4/W3 recipe on all admitted DEV; train-derived constant-column filter',
        'real_training_set_quality_score_allowed':False,'promotion_allowed':False,'fresh_raw_reproduction':False,
        'source_sha256':{name:digest(ROOT/name) for name in sources},
        'stop_rule':'One fit only. Preserve failures; no seeds, feature subsets, learner or blend changes.'},immutable=True)
    register_job({'id':'B501_ALL_DEV_DIAGNOSTIC_REFIT','kind':'validation','family':'B','candidate_id':candidate,
        'priority':65,'depends_on_success':['B401_FIXED_ALT_H_REPLICATION'],
        'argv':['scripts/refit_next4_b2_diagnostic.py'],'description':'One all-DEV B2 diagnostic fit; primary CI failure retained',
        'retry_safe':False,'nomination_sha256':digest(path)})
    progress('B2 synthetic diagnostic fit preregistered','One unchanged fit, matched all-DEV scope, no new frozen-model fits or in-sample quality estimate')
    print('NEXT4 B2 DIAGNOSTIC REFIT QUEUED',flush=True)


if __name__=='__main__':main()
