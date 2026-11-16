"""Add final dispositions without rewriting frozen measurement receipts or fit histories."""
from pathlib import Path
import sys
import json
import shutil
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    selection = json.loads((ROOT/'DEV_CANDIDATES_NEXT2.json').read_text()); assert selection['status'] == 'FROZEN_NEXT2_DEV_SELECTION'
    out = ROOT/'artifacts/next2/final_tables'; out.mkdir(parents=True, exist_ok=True)
    before = out/'RESEARCH_TABLES'; before.mkdir(exist_ok=True)
    names = ['MODEL_RESULTS_NEXT2.csv', 'COMPLEMENT_STANDALONE_RESULTS.csv', 'COMPLEMENT_BLEND_RESULTS.csv',
        'DISTILLATION_RESULTS.csv', 'EVIDENCE_MEMORY_RESULTS.csv', 'BAYES_AGE_ABLATION.csv', 'GLR_NECESSITY_RESULTS.csv', 'SERIAL_CORRECTION_RESULTS.csv', 'EXPERIMENT_LEDGER.csv']
    for name in names:
        if not (before/name).exists(): shutil.copyfile(ROOT/name, before/name)
    models = pd.read_csv(before/'MODEL_RESULTS_NEXT2.csv', float_precision='round_trip')
    assert len(models) == 80 and models.experiment_id.nunique() == 80
    dispositions = {}
    for row in models.itertuples():
        eid = row.experiment_id
        if eid == 'J102_SERIAL_VARIANCE_512': decision = 'FROZEN_PRIMARY_WITH_SELECTED_EXACT_DISTILLED_RUNTIME'
        elif eid == 'Y001_LONG_VARIANCE_ARCH1': decision = 'FROZEN_SCIENTIFIC_REFERENCE'
        elif eid == 'B104_LONG_VARIANCE_NO_GLR_MEMORY': decision = 'FROZEN_ROBUSTNESS_REFERENCE_NOT_DISTILLED_SELECTION'
        elif eid == 'BL2_C205_DEPENDENCE_80': decision = 'SELECTED_PERFORMANCE'
        elif eid == 'C205_DEPENDENCE': decision = 'SELECTED_FIXED_BLEND_COMPONENT'
        elif row.status == 'BUG': decision = 'IMPLEMENTATION_BUG_BEFORE_LEARNER_FIT'
        elif 'SCREEN_FAIL' in row.status: decision = 'SCREEN_REJECTED'
        elif row.status == 'FULL_GATE_FAIL': decision = 'FULL_CV_REJECTED'
        elif row.status == 'COMPLEMENT_ELIGIBLE_FIXED_BLEND': decision = 'ELIGIBLE_COMPLEMENT_NOT_SELECTED'
        elif row.status == 'EXACT_PARENT_REUSED': decision = 'EXACT_PARENT_PARITY_REUSE'
        else: decision = 'DEV_DIAGNOSTICS_REVIEWED_NOT_SELECTED'
        dispositions[eid] = decision
    role_rows = []
    runtime = pd.read_csv(ROOT/'RUNTIME_PRUNING_RESULTS.csv', float_precision='round_trip').set_index('runtime_variant')
    for role in ('distilled', 'performance'):
        record = selection[role]; costs = runtime.loc[record['runtime_variant']].to_dict()
        role_rows.append({'selected_role': role.upper(), 'experiment_id': record['experiment_id'], 'entry': record['entry'],
            'model_directory': record['model_directory'], **costs})
    pd.DataFrame(role_rows).to_csv(out/'FINAL_ROLE_RUNTIME.csv', index=False)
    for name in names[:-1]:
        frame = pd.read_csv(before/name, float_precision='round_trip')
        frame['source_experiment_status'] = frame['status'] if 'status' in frame else None
        frame['final_disposition'] = frame.experiment_id.map(dispositions)
        frame['final_role_selection_source'] = 'DEV_CANDIDATES_NEXT2.json'
        for role in ('distilled', 'performance'):
            record = selection[role]; mask = frame.experiment_id == record['experiment_id']; costs = runtime.loc[record['runtime_variant']]
            frame.loc[mask, 'selected_role'] = role.upper(); frame.loc[mask, 'selected_runtime'] = record['runtime']
            frame.loc[mask, 'deployment_point_median_us'] = costs.point_median_us
            frame.loc[mask, 'deployment_private_numeric_bytes'] = costs.private_per_series_numeric_bytes
            frame.loc[mask, 'deployment_materialized_tree_inputs'] = record['materialized_tree_inputs']
            if 'performance_gate' in frame: frame.loc[mask, 'performance_gate'] = record['gates']['performance']['status']
            if 'distilled_gate' in frame: frame.loc[mask, 'distilled_gate'] = record['gates']['distilled']['status']
            if 'generation_gate' in frame: frame.loc[mask, 'generation_gate'] = record['gates']['generation']['status']
        # Scientific CV0 cost fields remain explicitly distinct from deployment fields.
        frame.to_csv(ROOT/name, index=False)
    models = pd.read_csv(ROOT/'MODEL_RESULTS_NEXT2.csv', float_precision='round_trip')
    history = pd.read_csv(ROOT/'artifacts/next2/initialization/EXPERIMENT_LEDGER_NEXT_PRESERVED.csv', float_precision='round_trip')
    history['ledger_section'] = 'PREVIOUS_SESSION_PRESERVED'
    current = models[models.evidence_origin == 'NEXT2_RESEARCH'].copy(); assert len(current) == 71
    current['ledger_section'] = 'NEXT2_RESEARCH'
    pd.concat([history, current], ignore_index=True).to_csv(ROOT/'EXPERIMENT_LEDGER.csv', index=False)
    pd.DataFrame([{'experiment_id': k, 'final_disposition': v} for k, v in dispositions.items()]).to_csv(out/'FINAL_DISPOSITIONS.csv', index=False)
    inventory = json.loads((ROOT/'artifacts/next2/training_inventory/RESULTS.json').read_text())
    assert inventory['completed_training_events'] == 394
    independent = {role: json.loads((ROOT/'artifacts/next2/independent_reproduction'/role/'RESULTS.json').read_text()) for role in ('DISTILLED', 'PERFORMANCE')}
    count = sum(r['new_fits'] for r in independent.values()); assert count == 3
    write_json(out/'FINAL_TRAINING_INVENTORY.json', {'status': 'PASS', 'created_utc': utc_now(),
        'original_research_fit_inventory_sha256': sha256(ROOT/'artifacts/next2/training_inventory/RESULTS.json'),
        'research_learner_fits': 394, 'independent_audit_learner_fits': 3, 'total_NEXT2_learner_fits': 397,
        'by_role': {role: r['new_fits'] for role, r in independent.items()},
        'independent_receipts_sha256': {role: sha256(ROOT/'artifacts/next2/independent_reproduction'/role/'RESULTS.json') for role in independent},
        'named_experiments_and_runtime_initializations_are_separate_counts': True})
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'script_sha256': sha256(Path(__file__)),
        'selection_sha256': sha256(ROOT/'DEV_CANDIDATES_NEXT2.json'), 'scientific_records': 80, 'new_named_records': 71,
        'previous_ledger_rows_preserved': len(history), 'files_sha256': {name: sha256(ROOT/name) for name in names},
        'research_snapshot_sha256': {name: sha256(before/name) for name in names},
        'scope': 'Original experiment status is preserved; final_disposition and selected_role identify completed review. Scientific CV0 costs, exact runtime trial status and final all-DEV deployment costs are distinct.'})
    print('NEXT2 FINAL DISPOSITIONS AND 397-FIT INVENTORY PASS', flush=True)


if __name__ == '__main__': main()
