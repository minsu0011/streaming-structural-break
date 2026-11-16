"""Matched objective/weighting changes on the strong two-channel feature bank."""
from pathlib import Path
import copy
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.next.ranking_objectives import implementation_hash as ranking_hash
from src.utils.artifacts import write_json,sha256,utc_now


if __name__=='__main__':
    path = ROOT/'configs/next_variance_objective_candidates.json'
    if path.exists():
        print('Variance objective queue already frozen')
    else:
        parent_path = ROOT/'artifacts/next/experiments/V007_VARIANCE_FORECAST/RESULTS.json'
        parent = json.loads(parent_path.read_text(encoding='utf-8'))['candidate']
        candidates = []
        for eid,weight,rank in [('O101_VARIANCE_UNWEIGHTED','W0',False),('O102_VARIANCE_SERIES_BALANCED','W1',False),
                                ('O103_VARIANCE_PAIR_DENSITY','W2',False),('O104_VARIANCE_RANK128','W3',True)]:
            candidate = copy.deepcopy(parent)
            candidate.update(experiment_id=eid,parent_id='V007_VARIANCE_FORECAST',generation=15,phase='VARIANCE_MATCHED_OBJECTIVE',
                hypothesis='Earlier weighting and rank-loss results used weaker or larger feature banks. Isolate the training objective on the successful 77-feature variance-pair design.',
                change='Replace binary W3 with fixed time-aligned histogram RankNet at 128 bins.' if rank else f'Only training weights change from W3 to the previously frozen {weight} rule.',
                expected_effect='Test whether row density or pointwise calibration limits the independent variance evidence.',
                risks='TS-AUC contribution weighting may already be preferable; inverse-series length changes time emphasis. No validation prevalence, target, total length or mechanism labels enter training weights.',
                weighting=weight)
            if rank:
                candidate['ranking_objective']={'kind':'histogram_ranknet','bins':128}
                candidate['ranking_source_sha256']=ranking_hash()
            candidates.append(candidate)
        write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':15,
            'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
            'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),'thresholds_splits_seeds_unchanged':True,
            'parent_result_sha256':sha256(parent_path),'feature_bank_and_capacity_unchanged':True,
            'grid':'Only the previously frozen W0/W1/W2 rules and 128-bin time-aligned rank loss. The completed W3 parent is reused.',
            'candidates':candidates})
        print('FROZEN VARIANCE OBJECTIVES',len(candidates))
