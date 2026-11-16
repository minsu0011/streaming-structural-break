"""Freeze a controlled null/shift test before inspecting its model outputs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.synthetic_variance_stress import REGIMES,SCENARIOS,source_hash
from src.utils.artifacts import write_json,sha256,utc_now


if __name__=='__main__':
    path = ROOT/'configs/next_synthetic_stress_policy.json'
    if path.exists():
        print('Synthetic stress policy already frozen')
    else:
        models = {'PRIMARY':'artifacts/models/real/M7_cross_order_AR4_LGB_AR8_HGB_equal/fold_0.joblib',
            'HIGHEST_MEAN':'artifacts/models/real/M4_lightgbm_AR8_ARCH1_ABCD_S3_1cb5e51ac7ae/fold_0.joblib',
            'V007_VARIANCE_FORECAST':'artifacts/next/experiments/V007_VARIANCE_FORECAST/full/fold_0.joblib'}
        write_json(path,{'status':'LOCKED','created_utc':utc_now(),'seed':202609099,
            'regimes':list(REGIMES),'scenarios':list(SCENARIOS),'historical_points':2048,'online_points':1024,'burn_points':512,
            'break_times':[128,256,512],'calibration_series_per_regime':1000,'evaluation_series_per_regime':1000,'alphas':[.01,.05],
            'models':models,'model_sha256':{name:sha256(ROOT/value) for name,value in models.items()},
            'generator_sha256':source_hash(),'alarm_code_sha256':sha256(ROOT/'src/next/alarm_calibration.py'),
            'randomness':'Independent SeedSequence([seed, partition, regime_index, index]); partition 0 calibration, partition 1 evaluation. Same historical prefix and future noise across shift alternatives for each evaluation index.',
            'null_calibration':'One common threshold per model/alpha from 5000 independent stationary null-series maxima pooled equally across the five regimes; freeze before evaluating changed series.',
            'shifts':'Mean +/-0.5 stationary SD; observed scale x1.5 or /1.5; AR coefficient becomes -0.6 for nonnegative baseline or +0.6 otherwise, with innovation correction; tail becomes variance-normalized Student-t3; ARCH alpha becomes .85; mixed applies mean+scale+dependence.',
            'polarity':'Even/odd series indices balance mean and scale directions. Break time cycles independently across 128/256/512.',
            'historical_reuse':'Fit H once per model/index. Deep-copy every state/tree array for each synthetic future. Verify clones against fresh constructors on all scenarios for the first two indices of every regime.',
            'precision_rationale':'1000 evaluation sequences per controlled cell provide interpretable false-alarm and detection counts; report Wilson intervals and actual event counts rather than optimize synthetic scores.',
            'authority':'Controlled mechanism and null-alarm diagnostic only; no official DEV score, no screen/promotion authority, no model fitting or threshold deployment.',
            'limitations':'Simple stationary AR/ARCH and Student noise are a chosen simulation family, not an inferred competition generator. Model fold zero is fixed from earlier engineering work. These results cannot certify real-world false-alarm guarantees.'})
        print('FROZEN SYNTHETIC STRESS',40000,'unique sequences;',120000,'model-sequence evaluations')
