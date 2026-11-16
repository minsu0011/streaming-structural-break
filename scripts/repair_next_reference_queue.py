"""Record the detected base-family bug and preregister the corrected queue."""
from pathlib import Path
import sys
import json
import copy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.design_selection import selection_hash
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    source = ROOT/'configs/next_initial_candidates.json'
    initial = json.loads(source.read_text(encoding='utf-8'))
    destination = ROOT/'configs/next_corrected_candidates.json'
    if destination.exists():
        print('Corrected queue already frozen')
        return
    bug_path = ROOT/'artifacts/next/experiments/N013_ARCH_REFERENCE/RESULTS.json'
    bug = json.loads(bug_path.read_text(encoding='utf-8'))
    original_digest = sha256(bug_path)
    write_json(bug_path.with_name('RESULTS_BEFORE_BUG_CLASSIFICATION.json'),bug)
    bug['status'] = 'BUG'
    bug['reason'] = 'Reference adapter included E/F features beyond the frozen ABCD 51 columns; this was not an unchanged baseline. Preserve predictions as invalid matched-comparison evidence; no promotion allowed.'
    bug['superseded_by'] = 'R013_ARCH_REFERENCE'
    write_json(bug_path,bug)
    corrected = copy.deepcopy(initial)
    corrected['created_utc'] = utc_now()
    corrected['repair'] = {'initial_queue_sha256':sha256(source), 'bug_experiment':'N013_ARCH_REFERENCE',
        'original_result_sha256':original_digest,
        'cause':'All 57 computed old features were passed to the new learner. Original ABCD fit uses only columns 0..50, excluding E/F/Q.',
        'effect':'The original baseline selected 51 columns while the faulty adapter retained 56 nonconstant columns.',
        'correction':'Explicitly select original ABCD base columns before fitting; keep new engine groups unchanged.',
        'selection_code_sha256':selection_hash(),
        'core_only_experiments_affected':False,'promotion_or_screen_threshold_changes':False,
        'new_reference_requires_exact_full_OOF_prediction_parity':True}
    for candidate in corrected['candidates']:
        if candidate['base'] != 'arch8':
            continue
        candidate['experiment_id'] = 'R'+candidate['experiment_id'][1:]
        if candidate['parent_id'] == 'N013_ARCH_REFERENCE':
            candidate['parent_id'] = 'R013_ARCH_REFERENCE'
        candidate['base_family_selection'] = 'ABCD'
        if candidate['experiment_id'] == 'R013_ARCH_REFERENCE':
            candidate['expected_reference_role'] = 'HIGHEST_MEAN'
    write_json(destination,corrected)
    write_json(ROOT/'artifacts/next/REFERENCE_ADAPTER_BUG_REPORT.json',{
        'status':'CORRECTED_PENDING_EXACT_REFERENCE_REFIT','timestamp':utc_now(),**corrected['repair'],
        'corrected_queue_sha256':sha256(destination),'seal_rows':0,'reduced_new_usage':0})
    print('Corrected queue frozen; original faulty baseline preserved as BUG')


if __name__=='__main__':
    main()
