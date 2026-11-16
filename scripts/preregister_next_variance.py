"""Freeze variance-state evidence that normalization alone can suppress."""
from pathlib import Path
from dataclasses import asdict
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.engine import EngineConfig
from src.next.variance_evidence import VarianceConfig,feature_names
from src.next.extensions import extension_hash
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.next.ranking_objectives import implementation_hash as ranking_hash
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    path = ROOT/'configs/next_variance_candidates.json'
    if path.exists():
        print('Variance evidence queue already frozen')
        return
    rows = []
    all_groups = ['VARIANCE_LOCAL','VARIANCE_GLR','VARIANCE_BAYES']
    for eid,parent,base,mode,groups,selection,ranking in [
        ('V001_VARIANCE_STANDALONE','N020_STANDALONE_S2','none','arch1',all_groups,None,False),
        ('V002_VARIANCE_ARCH','R013_ARCH_REFERENCE','arch8','arch1',all_groups,None,False),
        ('V003_VARIANCE_ARCH2','V002_VARIANCE_ARCH','arch8','arch2',all_groups,None,False),
        ('V004_VARIANCE_EWMA','V002_VARIANCE_ARCH','arch8','ewma',all_groups,None,False),
        ('V005_VARIANCE_LOCAL','V002_VARIANCE_ARCH','arch8','arch1',['VARIANCE_LOCAL'],None,False),
        ('V006_VARIANCE_SCORES','V002_VARIANCE_ARCH','arch8','arch1',all_groups,'scores',False),
        ('V007_VARIANCE_FORECAST','V002_VARIANCE_ARCH','arch8','arch1',all_groups,'forecast',False),
        ('V008_VARIANCE_RANK','V002_VARIANCE_ARCH','arch8','arch1',all_groups,None,True)]:
        settings = asdict(VarianceConfig(normalization=mode))
        extension = {'kind':'conditional_variance','settings':settings}
        item = {'experiment_id':eid,'parent_id':parent,'generation':8,'phase':'CONDITIONAL_VARIANCE',
            'hypothesis':'Conditioning residuals on changing volatility can absorb a true scale break. Preserve the forecast state and conditional likelihood score, not only its standardized residual.',
            'change':'Add H-calibrated raw energy, forecast log variance, standardized energy and intercept/lag1/lag2 Gaussian variance scores; normalizer, feature-removal and ranking variants are explicit.',
            'expected_effect':'Retain variance-model mismatch and adapted variance-level evidence under bounded causal updates.',
            'risks':'Historical residuals fit both AR and variance reference; conditional Gaussian scores need not be well calibrated under tails; correlated evidence is learned as features and never multiplied as independent probabilities.',
            'engine_config':asdict(EngineConfig()),'groups':groups,'base':base,'learner':'M4_lightgbm','weighting':'W3','params':{},
            'extension':extension,'extension_frozen_hash':extension_hash(extension)}
        if base=='arch8':
            item['base_family_selection']='ABCD'
        if selection:
            prefixes = ('variance_intercept_score','variance_lag1_score','variance_lag2_score') if selection=='scores' else ('raw_energy','log_variance_forecast')
            item['keep_names'] = ['conditional_variance__'+name for name in feature_names(settings) if name.startswith(prefixes)]
        if ranking:
            item.update(learner='histogram_ranknet',ranking_objective={'kind':'histogram_ranknet','bins':128},ranking_source_sha256=ranking_hash())
        rows.append(item)
    write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':8,
        'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
        'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),'thresholds_splits_seeds_unchanged':True,
        'H_model':'Existing frozen NEXT AR8 and ARCH1/ARCH2/EWMA normalizers; no online coefficient fitting.',
        'causality':'Forecast variance uses previous capped squared innovations only; current residual updates conditional state after current evidence is computed.',
        'channel_calibration':'Replay H with H-fitted parameters, burn min(160, H_residual_length // 4), center by H mean and divide by H SD floored at 1e-6; channel evidence clips at +/-12.',
        'score_identity':'Gaussian log-likelihood derivatives with respect to omega/H_variance, alpha1, alpha2; finite-difference checks passed.',
        'interpretation':'Features only; no formal likelihood validity or posterior probability claim under non-Gaussian/serially dependent innovations.',
        'candidates':rows})
    print('FROZEN VARIANCE',len(rows))


if __name__=='__main__':
    main()
