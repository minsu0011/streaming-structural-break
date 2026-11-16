"""Pin existing fold-0 models for mechanism diagnostics, with no new fitting."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    names=['J102_SERIAL_VARIANCE_512','Y001_LONG_VARIANCE_ARCH1','B104_LONG_VARIANCE_NO_GLR_MEMORY',
        'Y002_LONG_VARIANCE_ARCH2','Y003_LONG_VARIANCE_EWMA',
        'M2_EWMA','M2_LEAKY','M2_POSTERIOR_LIKE','M2_DECAYED_ONLY','N2_LOG_AGE',
        'A2_SHORT','A2_LONG','A2_SPARSE64','V2_WINSOR_3','V2_HUBER_4','V2_LOG1P_4',
        'V2_ROBUST_SCALE_4','V2_ECDF_TAIL_4','D208_NO_ARCH','D210_NO_FORECAST',
        'D202_NO_EXPLICIT_GLR','D205_NO_CUSUM_MEMORY','S205_PARENT_PLUS_LOG_RATIOS','S203_CONDITIONAL_SCALE_LIKELIHOOD']
    path=ROOT/'configs/next2/COMPONENT_STRESS_CV0_LOCK.json'
    if path.exists():
        print('NEXT2 COMPONENT STRESS LOCK RETAINED',flush=True)
        return
    models=[]
    for eid in names:
        folder=ROOT/'artifacts/next2/experiments'/eid
        if not (folder/'RESULTS.json').exists():
            folder=ROOT/'artifacts/next/experiments'/eid
        r=json.loads((folder/'RESULTS.json').read_text())
        model=folder/'full/fold_0.joblib'
        if not model.exists() or not r.get('full') or r['status'] in ('SCREEN_FAIL','BUG'):
            raise RuntimeError('Only existing valid full-CV fold models are admitted')
        models.append({'experiment_id':eid,'candidate':r['candidate'],'full_cv_status':r['status'],
            'model_path':model.relative_to(ROOT).as_posix(),'model_sha256':sha256(model),
            'source_result_sha256':sha256(folder/'RESULTS.json'),'outer_fold_model':0})
    write_json(path,{'status':'FROZEN_DIAGNOSTIC_MENU','created_utc':utc_now(),'models':models,
        'counterexample_lock_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
        'scope':'All 9 independent calibration cohorts, 18 fixed counterexamples and 4 fixed long-null regimes',
        'model_training':'Existing original full-CV fold-0 models; matched training partition, no synthetic refitting',
        'new_fits':0,'failed_screen_candidates_admitted':0,
        'authority':'Exploratory component diagnostics; all models use fold 0, distinct from all-DEV primary/finalist counterexample tables.',
        'purpose':'Explain memory, age-grid, energy, ARCH and scale-likelihood failure modes. Preserve fixed DEV admission failures.',
        'selection':'No synthetic result can promote a candidate, waive a DEV gate, or change this menu.',
        'primary_comparison':'Y001 fold-0 for direct descendants; J102 fold-0 also reported',
        'seal_rows':0,'reduced_new_usage':0})
    print('NEXT2 COMPONENT STRESS LOCKED',len(models),flush=True)


if __name__=='__main__':
    main()
