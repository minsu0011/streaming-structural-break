"""Freeze four small C205 subsets and record exact sign-branch reuse."""
from pathlib import Path
import sys
import copy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,ROOT,atomic_json,read_json,digest,utc_now,progress
from src.next3.frozen import candidate,C205
from src.next.candidate_io import validate_sources


def main():
    source,folder=candidate(C205)
    predicates={
        'P1_LAGS_1_2':lambda name:any(f'dependence_{k}_' in name for k in (0,1)),
        'P2_LAGS_4_8':lambda name:any(f'dependence_{k}_' in name for k in (2,3)),
        'P3_LOCAL_ONLY':lambda name:not ('bayes' in name),
        'P4_BAYES_ONLY':lambda name:'bayes' in name}
    for kind,keep in predicates.items():
        spec=copy.deepcopy(source);spec.update(experiment_id=kind,parent_id=C205,generation='NEXT3_SPECIALIST_PRUNE_V1',phase='FIXED_BLEND_SPECIALIST_PRUNING')
        spec['keep_names']=[name for name in source['keep_names'] if keep(name)]
        assert len(spec['keep_names']) in (8,12,16)
        atomic_json(BASE/'hypotheses'/f'{kind}.json',{'created_utc':utc_now(),
            'HYPOTHESIS':'A smaller C205 lag/statistic subset may preserve complementary blend ranking with less amplitude sensitivity or runtime cost.',
            'BOTTLENECK':'B3','CHANGE':kind,'EXPECTED_EFFECT':'Retain fixed 80/20 incremental ranking; standalone score alone is not admission authority.',
            'RISKS':'Removing evidence can destroy dependence gain; changes in fitted trees mean this is a scientific ablation, not prediction-exact optimization.',
            'PRIMARY_METRIC':'Fixed .8 J102 + .2 pruned specialist TS-AUC relative to frozen performance',
            'SECONDARY_DIAGNOSTIC':'Standalone score, mechanism support, actual state/runtime if finalist',
            'STOP_RULE':'Frozen screen gate first; failed screen does not receive full evaluation.',
            'candidate':spec,'frozen_source_candidate_sha256':digest(folder/'RESULTS.json'),
            'policy_sha256':digest(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json'),
            'finalist_admission':{'mean_delta_min':-.0008,'median_delta_min':-.001,'worst_delta_min':-.001},
            'new_features':0,'feature_count':len(spec['keep_names'])},immutable=True)
    previous=ROOT/'artifacts/next2/experiments/C211_SIGN_DEPENDENCE/RESULTS.json'
    record=read_json(previous);validate_sources(record['candidate'])
    proof={}
    for fold in (0,4):
        path=previous.parent/'screen'/f'fold_{fold}.json'
        receipt=read_json(path)
        for name,expected in receipt['files_sha256'].items():
            assert digest(path.parent/name)==expected;proof[(path.parent/name).relative_to(ROOT).as_posix()]=expected
    atomic_json(BASE/'reuse/I1_SIGN_PRODUCTS.json',{'status':'REUSED_NEGATIVE_RESULT','created_utc':utc_now(),
        'proposed_hypothesis':'I1_SIGN_PRODUCTS','exact_prior_experiment':'C211_SIGN_DEPENDENCE',
        'source_result_sha256':digest(previous),'candidate':record['candidate'],'source_files_sha256':proof,
        'prior_screen_folds':record['screen']['fold_scores'],'prior_screen_mean':record['screen']['mean'],
        'reason':'Exactly the same sign-before-lag-product intervention with four lags and 24 unchanged evidence features; source/config validated. Do not refit or waive its failed prior screen.',
        'new_fits':0},immutable=True)
    progress('C205 finite pruning menu registered','Two lag subsets, local-only, Bayes-only; exact failed sign-product hypothesis reused without new fits')
    print('NEXT3 C205 PRUNES REGISTERED; SIGN PRODUCT REUSED FAILURE',flush=True)


if __name__=='__main__':main()
