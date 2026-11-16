"""Freeze the two requested cross-order forecast-mismatch pairs."""
from pathlib import Path
from dataclasses import asdict
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.engine import EngineConfig
from src.next.mismatch_evidence import MismatchConfig
from src.next.extensions import extension_hash
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.next.ranking_objectives import implementation_hash as ranking_hash
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    path = ROOT/'configs/next_mismatch_candidates.json'
    if path.exists():
        print('Mismatch queue already frozen')
        return
    candidates = []
    groups = ['MISMATCH_LOCAL','MISMATCH_GLR','MISMATCH_BAYES']
    for eid,parent,base,config,keep,ranking in [
        ('U001_MISMATCH_4_8','N021_STANDALONE_S3','none',MismatchConfig(),groups,False),
        ('U002_MISMATCH_4_8_ARCH','R013_ARCH_REFERENCE','arch8',MismatchConfig(),groups,False),
        ('U003_MISMATCH_8_12_ARCH','U002_MISMATCH_4_8_ARCH','arch8',MismatchConfig(8,12),groups,False),
        ('U004_MISMATCH_LONG_ARCH','U002_MISMATCH_4_8_ARCH','arch8',MismatchConfig(max_age=512),groups,False),
        ('U005_MISMATCH_LOCAL_ARCH','U002_MISMATCH_4_8_ARCH','arch8',MismatchConfig(),['MISMATCH_LOCAL'],False),
        ('U006_MISMATCH_GLR_ARCH','U002_MISMATCH_4_8_ARCH','arch8',MismatchConfig(),['MISMATCH_GLR'],False),
        ('U007_MISMATCH_RANK_ARCH','U002_MISMATCH_4_8_ARCH','arch8',MismatchConfig(),groups,True)]:
        extension = {'kind':'ar_mismatch','settings':asdict(config)}
        item = {'experiment_id':eid,'parent_id':parent,'generation':6,'phase':'AR_MISMATCH',
            'hypothesis':'A change in dependence can reverse the predictive advantage of two frozen historical AR orders before marginal features respond.',
            'change':'Append H-normalized forecast-gap, squared-error advantage, log-energy advantage and high-order innovation times forecast-gap evidence; order, age, family-removal and ranking ablations are explicit.',
            'expected_effect':'Measure loss of historical predictive structure directly without online coefficient fitting.',
            'risks':'Nearby AR fits can have very small mismatch variance; raw H empirical scaling may be sensitive to outliers; related channels are correlated and are not independent Bayes factors.',
            'engine_config':asdict(EngineConfig()),'groups':keep,'base':base,'learner':'M4_lightgbm','weighting':'W3','params':{},
            'extension':extension,'extension_frozen_hash':extension_hash(extension)}
        if base=='arch8':
            item['base_family_selection']='ABCD'
        if ranking:
            item.update(learner='histogram_ranknet',ranking_objective={'kind':'histogram_ranknet','bins':128},ranking_source_sha256=ranking_hash())
        candidates.append(item)
    write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':6,
        'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
        'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),'thresholds_splits_seeds_unchanged':True,
        'pairs':[[4,8],[8,12]],'filtering':'Both H-fitted pole-projected AR models remain frozen online.',
        'H_calibration':'Align residuals on common H endpoints; divide errors by high-order H innovation SD; center each derived channel by H mean and divide by H SD floored at 1e-6; online calibrated channel clips at +/-12.',
        'sequential_policy':'Half-lives 8/32/128, +/-CUSUM drift .25, fixed mean GLR and integrated zero-centered Gaussian mean priors .25/1; age widths weight the mixture; memory is fixed-size.',
        'mathematical_tests':'Direct frozen AR predictions, direct suffix GLR/Bayes, ring wrap, unchanged-prefix, fixed-memory, and synthetic lag-8 coefficient sign-change tests passed.',
        'candidates':candidates})
    print('FROZEN MISMATCH',len(candidates))


if __name__=='__main__':
    main()
