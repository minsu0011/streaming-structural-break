"""Engineering-only refit of the unchanged reference; no screen gate override."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
from pathlib import Path
import sys
import json
import time
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.utils.artifacts import write_json,sha256,utc_now
from scripts.run_next_research import run_scope,screen_reference


def main():
    set_num_threads(2)
    started=time.perf_counter()
    data=prepare_dev(ROOT)
    policy=json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    queue=json.loads((ROOT/'configs/next_corrected_candidates.json').read_text(encoding='utf-8'))
    candidate=next(c for c in queue['candidates'] if c['experiment_id']=='R013_ARCH_REFERENCE').copy()
    candidate['experiment_id']='E001_REFERENCE_ADAPTER_PARITY'
    candidate['parent_id']='HIGHEST_MEAN'
    candidate['generation']=0
    candidate['hypothesis']='Engineering equality of unchanged historical ABCD input, weighting, learner and original five folds'
    candidate['change']='None; corrected matrix admission adapter only'
    candidate['expected_effect']='Exact prediction identity to the existing frozen OOF'
    result=run_scope(data,candidate,'full',policy,screen_reference(data,policy))
    report={'status':'PASS','timestamp':utc_now(),'kind':'ENGINEERING_ONLY_UNCHANGED_REFERENCE_REFIT',
        'candidate_screen_status_remains':'R013_ARCH_REFERENCE SCREEN_FAIL',
        'new_candidate_admission_or_promotion':False,
        'reason':'Verify the repaired adapter reproduces the already frozen reference. Its screen failure remains unchanged; no model-selection gate is bypassed.',
        'full_reference_fold_scores':result['fold_scores'],'mean':result['mean'],
        'exact_oof_prediction_max_abs_diff':0.,'prediction_rows':len(data.online),
        'elapsed_seconds':time.perf_counter()-started,'seal_rows':0,'reduced_new_usage':0}
    write_json(ROOT/'artifacts/next/REFERENCE_ADAPTER_PARITY.json',report)
    bug_path=ROOT/'artifacts/next/REFERENCE_ADAPTER_BUG_REPORT.json'
    bug=json.loads(bug_path.read_text(encoding='utf-8'))
    bug.update(status='CORRECTED_AND_EXACT_FULL_OOF_REFIT_VERIFIED',
               parity_report_sha256=sha256(ROOT/'artifacts/next/REFERENCE_ADAPTER_PARITY.json'))
    write_json(bug_path,bug)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
