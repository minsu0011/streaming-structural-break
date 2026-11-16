"""Freeze a small ranking-objective comparison before any ranking score."""
from pathlib import Path
import sys
import json
import copy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.next.ranking_objectives import implementation_hash
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    path=ROOT/'configs/next_ranking_candidates.json'
    if path.exists():
        print('Ranking queue already frozen')
        return
    initial=json.loads((ROOT/'configs/next_corrected_candidates.json').read_text(encoding='utf-8'))
    gen2=json.loads((ROOT/'configs/next_generation2_candidates.json').read_text(encoding='utf-8'))
    lookup={c['experiment_id']:c for queue in [initial,gen2] for c in queue['candidates']}
    rows=[]
    variants=[('K001_RANK_BINARY_QUERY','R013_ARCH_REFERENCE',{'kind':'binary_query_sorted'}),
        ('K002_RANK_SAMPLE8','R013_ARCH_REFERENCE',{'kind':'sampled_ranknet','pairs_per_positive':8}),
        ('K003_RANK_SAMPLE32','R013_ARCH_REFERENCE',{'kind':'sampled_ranknet','pairs_per_positive':32}),
        ('K004_RANK_HIST64','R013_ARCH_REFERENCE',{'kind':'histogram_ranknet','bins':64}),
        ('K005_RANK_HIST128','R013_ARCH_REFERENCE',{'kind':'histogram_ranknet','bins':128}),
        ('K006_RANK_XENDCG','K001_RANK_BINARY_QUERY',{'kind':'rank_xendcg'}),
        ('K007_RANK_LAMBDA64','K001_RANK_BINARY_QUERY',{'kind':'lambdarank'}),
        ('K008_RANK_SCORE_HIST128','S003_ARSCORE_ARCH',{'kind':'histogram_ranknet','bins':128}),
        ('K009_RANK_SCALE_HIST128','C002_HCAL_SCALE',{'kind':'histogram_ranknet','bins':128})]
    for eid,parent,objective in variants:
        source=lookup['R013_ARCH_REFERENCE' if parent=='K001_RANK_BINARY_QUERY' else parent]
        candidate=copy.deepcopy(source)
        candidate.pop('expected_reference_role',None)
        candidate.update(experiment_id=eid,parent_id=parent,generation=3,phase='RANKING',learner=objective['kind'],
            hypothesis='Same-time pairwise ranking loss may improve TS-AUC ordering beyond individually weighted binary loss.',
            change='Only supervised objective changes; query-sorted binary control isolates the mandatory row grouping of built-in ranking.',
            expected_effect='Better within-time ordering without any validation-target statistics entering the objective',
            risks='Histogram or pair-sampling approximation; ranking losses remain surrogates; NDCG positions and query normalization do not exactly match pair-count-weighted AUC.',
            ranking_objective=objective,ranking_source_sha256=implementation_hash())
        rows.append(candidate)
    write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':3,
        'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
        'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'ranking_source_sha256':implementation_hash(),'thresholds_splits_seeds_unchanged':True,
        'histogram_ranknet':'At each training time, 64/128 adaptive margin cells with separate positive/negative centroids approximate all same-time pair gradients. Exact at common margins or noncolliding cells; no sampled pair matrix. Gradients are scaled by N/(2*total_pair_count).',
        'metric':'All screening, primary CV and bootstrap use exact float32 TS-AUC; only the training loss gradient is approximated.',
        'NDCG_policy':'Exploratory only. Fixed lambdarank truncation 64 and binary label gains [0,1]. No NDCG result is a selection authority.',
        'sources':[{'title':'LightGBM LGBMRanker API','url':'https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html'},
                   {'title':'Official LightGBM parameters source','url':'https://github.com/lightgbm-org/LightGBM/blob/main/docs/Parameters.rst'}],
        'external_strategy_sources_used':False,'installed_lightgbm_version':'4.7.0','candidates':rows})
    print('FROZEN RANKING',len(rows),'candidates')


if __name__=='__main__':
    main()
