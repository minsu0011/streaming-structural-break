"""Matched ablations of the first materially improved full-CV variance model."""
from pathlib import Path
import copy
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.extensions import extension_hash
from src.next.variance_evidence import feature_names
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    path = ROOT/'configs/next_variance_ablation_candidates.json'
    if path.exists():
        print('Variance ablation queue already frozen')
        return
    parent_path = ROOT/'artifacts/next/experiments/V007_VARIANCE_FORECAST/RESULTS.json'
    result = json.loads(parent_path.read_text(encoding='utf-8'))
    parent = result['candidate']
    keep = parent['keep_names']
    variants = [
        ('Q001_VARIANCE_RAW_ONLY','Keep only raw-energy channels; remove variance-forecast channels.',[n for n in keep if '__raw_energy_' in n],None),
        ('Q002_VARIANCE_FORECAST_ONLY','Keep only variance-forecast channels; remove raw-energy channels.',[n for n in keep if '__log_variance_forecast_' in n],None),
        ('Q003_VARIANCE_NO_GLR_MEMORY','Remove explicit GLR summaries and all GLR-derived persistent-memory features; keep local and Bayes summaries.',[n for n in keep if '_glr_' not in n and '_memory_' not in n],None),
        ('Q004_VARIANCE_NO_BAYES','Remove only conjugate mean-Bayes summaries; GLR-derived memory remains.',[n for n in keep if '_bayes' not in n],None),
        ('Q005_VARIANCE_NO_MEMORY','Remove only persistent/decayed GLR memory; current GLR and Bayes summaries remain.',[n for n in keep if '_memory_' not in n],None),
        ('Q006_VARIANCE_LOCAL_ONLY','Keep instantaneous, EWMA and CUSUM summaries only.',[n for n in keep if '_glr_' not in n and '_bayes' not in n and '_memory_' not in n],None),
        ('Q007_VARIANCE_GLR_ONLY','Keep only explicit fixed-age GLR max/logmean/argage summaries.',[n for n in keep if '_glr_' in n],None),
        ('Q008_VARIANCE_BAYES_ONLY','Keep only fixed-age analytic Gaussian mean-prior Bayes summaries.',[n for n in keep if '_bayes' in n],None),
        ('Q009_VARIANCE_ARCH2_PAIR','Keep the same two channels and only change the H variance normalizer from ARCH1 to ARCH2.',keep,'arch2'),
        ('Q010_VARIANCE_EWMA_PAIR','Keep the same two channels and only change the variance forecasting rule from ARCH1 to fixed EWMA.',keep,'ewma')]
    candidates = []
    for eid,change,names,mode in variants:
        candidate = copy.deepcopy(parent)
        candidate.update(experiment_id=eid,parent_id='V007_VARIANCE_FORECAST',generation=10,phase='VARIANCE_MATCHED_ABLATION',
            hypothesis='Determine which of raw-energy evidence, variance-forecast evidence, tau integration and persistent memory accounts for V007 improvement.',
            change=change,expected_effect='Attribute gains or remove unnecessary work without changing split, loss, tree capacity or selection gates.',
            risks='Ablations are adaptively motivated after observing V007; each still needs its own unchanged screen gate. Failed-screen ablations receive no full-CV extrapolation.',keep_names=names)
        if mode:
            candidate['extension']['settings']['normalization']=mode
            candidate['extension_frozen_hash']=extension_hash(candidate['extension'])
        candidates.append(candidate)
    write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':10,
        'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
        'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),'thresholds_splits_seeds_unchanged':True,
        'parent_result_sha256':sha256(parent_path),'parent_mean':result['full']['mean'],
        'candidate_capacity_and_loss_unchanged':True,
        'baseline_full_ablation':'Frozen HIGHEST_MEAN plus E001 exact five-fold reference-adapter refit is the 51-feature matched no-new-statistic baseline. Its full predictions predate V007.',
        'candidates':candidates})
    print('FROZEN VARIANCE ABLATIONS',len(candidates))


if __name__=='__main__':
    main()
