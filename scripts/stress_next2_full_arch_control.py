"""Whole-ARCH-off existing CV0 model on the unchanged fixed synthetic suite."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stress_next2_components import evaluate,summarize
from src.next2.counterexamples import policy,source_hash as generator_hash
from src.next2.synthetic_analysis import calibration_map
from src.next2.runtime import PreparedBankPredictor,source_hash as runtime_hash
from src.next2.io import model_class
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    eid='D218_FULL_PIPELINE_NO_ARCH'
    folder=ROOT/'artifacts/next2/experiments'/eid
    report=json.loads((folder/'RESULTS.json').read_text())
    assert report['status']=='FULL_GATE_FAIL' and report['full']['status']=='COMPLETE'
    ref={'experiment_id':eid,'candidate':report['candidate'],'full_cv_status':report['status'],
        'model_path':(folder/'full/fold_0.joblib').relative_to(ROOT).as_posix(),
        'model_sha256':sha256(folder/'full/fold_0.joblib'),'outer_fold_model':0}
    identity={'candidate':eid,'model_sha256':ref['model_sha256'],'generator_sha256':generator_hash(),
        'runtime_sha256':runtime_hash(),'script_sha256':sha256(Path(__file__)),
        'adapter':'Existing generic PreparedBankPredictor with already validated registered whole-ARCH-off module; no production admission.',
        'analysis_sha256':sha256(ROOT/'src/next2/synthetic_analysis.py'),
        'driver_sha256':sha256(ROOT/'scripts/stress_next2_components.py'),
        'counterexample_lock_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')}
    nomination=ROOT/'configs/next2/FULL_ARCH_CONTROL_STRESS_LOCK.json'
    if nomination.exists():
        assert json.loads(nomination.read_text())['identity']==identity
    else:
        write_json(nomination,{'created_utc':utc_now(),'identity':identity,'model':ref,'new_fits':0,
            'reason':'Whole-component attribution uses the same original suite as branch-only D208; failed real DEV gates remain failed.'})
    lock=policy();engine=PreparedBankPredictor(model_class(ref['candidate']).load(ROOT/ref['model_path']),resident=False)
    for scenario in calibration_map(lock).values():
        evaluate(lock,engine,ref,'calibration',scenario,lock['calibration_null_series_per_regime'],lock['online_length'],identity)
    for spec in lock['scenarios']:
        evaluate(lock,engine,ref,'counterexample',spec['id'],lock['evaluation_series_per_scenario'],lock['online_length'],identity)
    for scenario in lock['long_null']['regimes']:
        evaluate(lock,engine,ref,'long_null',scenario,lock['long_null']['series_per_regime'],max(lock['long_null']['horizons']),identity)
    summarize(lock,ref,identity)
    print('NEXT2 FULL ARCH CONTROL STRESS COMPLETE',flush=True)


if __name__=='__main__':
    main()
