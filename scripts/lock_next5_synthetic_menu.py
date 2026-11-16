"""Freeze unchanged DGPs and fitted-fold stability coverage before stress scores."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress


def main():
    old=read_json(ROOT/'artifacts/next4/diagnostic_policy/SYNTHETIC_POLICY.json');generator=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
    atomic_json(BASE/'synthetic/MENU_LOCK.json',{'created_utc':utc_now(),'status':'FROZEN_BEFORE_NEXT5_FITTED_SYNTHETIC_SCORES',
      'original_generator_policy_sha256':digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
      'original_NEXT4_synthetic_policy_sha256':digest(ROOT/'artifacts/next4/diagnostic_policy/SYNTHETIC_POLICY.json'),
      'generator_sources_sha256':{n:digest(ROOT/n) for n in ('src/next2/counterexamples.py','src/next2/cross_background_scale.py')},
      'counterexamples':old['counterexamples'],'additional_counterexamples':old['additional_counterexamples'],
      'long_null_regimes':old['long_null_regimes'],'long_null_horizons':old['long_null_horizons'],
      'threshold_alphas':old['threshold_alphas'],'score_quantiles':old['score_quantiles'],
      'historical_length':generator['historical_length'],'counterexample_horizon':generator['online_length'],
      'tau_cycle':generator['tau_cycle'],'generator_seed':generator['seed'],'parameter_changes_from_candidate_results':False,
      'fold_model_sizes':old['fold0_sizes'],'frozen_whole_ResearchPool_model_sizes':old['all_dev_sizes'],
      'candidate_coverage':'Every completed full candidate at all five fitted PRIMARY folds. Failed-screen candidates use the two already-fitted screen folds 0 and 4; no refit to rescue failure.',
      'training_match':'Each candidate uses J102/C205 trained on exactly the same ResearchPool training IDs and scope/fold. Never use historical all-DEV fitted models for these comparisons.',
      'purpose_of_all_fold_coverage':'Assess sensitivity of outlier/long-null conclusions to the fitted artifact, prompted by NEXT4 fold0-versus-deployment reversals. Overlapping fold training sets are not independent replications.',
      'calibration_rule':old['calibration'],'paired_bootstrap_replicates':2000,'paired_bootstrap_seed':20260908,
      'bootstrap_unit':old['bootstrap_unit'],'no_additional_seeds_or_severity_search':True,
      'impulse_horizon':2048,'impulse_horizon_reason':'All frozen t0 values plus requested lag512 must be observable; same DGP/seed paths extended without a parameter change.',
      'impulse_lags':read_json(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json')['outlier_impulse_lags'],
      'reuse_rule':'Generate the longest fixed horizon once and reuse proven identical prefixes; reuse matching baseline/cohort predictions across candidates.',
      'whole_ResearchPool_roles':'Only already-frozen final scientific recipes, using their actual final fitted artifacts and independently specified whole-ResearchPool sample sizes.',
      'feature_and_execution_hashes_required_before_first_fitted_synthetic_score':True,
      'selection_authority':'Secondary robustness diagnostics; cannot replace or relax primary TS-AUC/common promotion gates.',
      'new_scientific_fits':0,'new_real_Shadow_rows':0,'new_LOCAL_SEAL_rows':0},immutable=True)
    progress('Synthetic and outlier menu frozen','Unchanged DGP parameters/seeds; every fitted full fold compared with its matched reference; explicit fold-dependence limitation')


if __name__=='__main__':main()
