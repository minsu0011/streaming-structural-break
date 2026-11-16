"""Preregister a directional-score ablation, motivated by K008 ranking signal."""
from pathlib import Path
from dataclasses import asdict
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.engine import EngineConfig
from src.next.score_evidence import ScoreConfig
from src.next.extensions import extension_hash
from src.next.registered_extensions import register_extension, registration_hash
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.next.ranking_objectives import implementation_hash as ranking_hash
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    path = ROOT/'configs/next_score_coordinate_candidates.json'
    if path.exists():
        print('Score-coordinate queue already frozen')
        return
    registration = {'kind':'score_coordinates', 'module':'score_coordinates'}
    candidates = []
    for eid, parent, base, groups, covariance, ranking in [
        ('D001_SCORE_COORD_STANDALONE','S001_ARSCORE_STANDALONE','none',['SCORE','COORDINATE'],'full',False),
        ('D002_SCORE_COORD_ARCH','S003_ARSCORE_ARCH','arch8',['SCORE','COORDINATE'],'full',False),
        ('D003_SCORE_COORD_ONLY_ARCH','D002_SCORE_COORD_ARCH','arch8',['COORDINATE'],'full',False),
        ('D004_SCORE_COORD_RANK','K008_RANK_SCORE_HIST128','arch8',['SCORE','COORDINATE'],'full',True),
        ('D005_SCORE_COORD_DIAGONAL','D002_SCORE_COORD_ARCH','arch8',['SCORE','COORDINATE'],'diagonal',False),
        ('D006_SCORE_COORD_ONLY_RANK','D004_SCORE_COORD_RANK','arch8',['COORDINATE'],'full',True)]:
        extension = {'kind':'score_coordinates', 'settings':asdict(ScoreConfig(covariance=covariance))}
        register_extension(extension, registration)
        item = {'experiment_id':eid, 'parent_id':parent, 'generation':5, 'phase':'SCORE_DIRECTION',
            'hypothesis':'Aggregated score energy, maximum and signed sum can discard which AR lag changes; individual EWMA coordinates may expose stable directional evidence.',
            'change':'Expose the 24 existing AR8 EWMA score coordinates; aggregate features are bitwise unchanged. Matched aggregate-removal, covariance and ranking variants are explicit.',
            'expected_effect':'Let a fixed small learner distinguish conditional mean-change directions without estimating online AR coefficients.',
            'risks':'Coordinates are noisier than pooled energy; tree selection can overfit correlated lag directions; full historical covariance is fitted in-sample.',
            'engine_config':asdict(EngineConfig()), 'groups':groups, 'base':base, 'learner':'M4_lightgbm', 'weighting':'W3', 'params':{},
            'extension':extension, 'extension_frozen_hash':extension_hash(extension),
            'extension_registration':registration, 'registration_source_sha256':registration_hash()}
        if base == 'arch8':
            item['base_family_selection'] = 'ABCD'
        if ranking:
            item.update(learner='histogram_ranknet', ranking_objective={'kind':'histogram_ranknet','bins':128}, ranking_source_sha256=ranking_hash())
        candidates.append(item)
    write_json(path, {'status':'LOCKED', 'created_utc':utc_now(), 'generation':5,
        'fitting_code_sha256':fitting_hash(), 'predictor_code_sha256':predictor_hash(),
        'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'), 'thresholds_splits_seeds_unchanged':True,
        'forensic_input':'K008 full mean improves +0.001294 but misses the +0.0015 primary promotion gate; no promotion or new-generation claim.',
        'candidates':candidates})
    print('FROZEN SCORE COORDINATES', len(candidates))


if __name__ == '__main__':
    main()
