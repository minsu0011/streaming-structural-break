"""Append NEXT5 rows while preserving every original ledger byte and row."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.report_tables import ScalarEvidence, primary_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9_]+', args.tag)
    assert read_json(BASE / 'SESSION.json').get('audit_started_utc')
    assert read_json(ROOT / 'DEV_CANDIDATES_NEXT5.json')['status'] == 'FROZEN_NEXT5_FINAL_ROLES'
    retained = BASE / 'preflight/retained/EXPERIMENT_LEDGER.csv'
    target = ROOT / 'EXPERIMENT_LEDGER.csv'
    original = retained.read_bytes()
    assert target.read_bytes() == original, 'Never replace user changes or append NEXT5 twice'
    assert original.endswith(b'\n')
    reader = csv.DictReader(io.StringIO(original.decode('utf-8-sig'), newline=''))
    old_rows, fields = list(reader), reader.fieldnames
    assert len(old_rows) == 276 and all(r.get('ledger_section') != 'NEXT5' for r in old_rows)
    inventory_path = BASE / 'final_audit' / args.tag / 'fit_inventory/RESULTS.json'
    inventory = read_json(inventory_path)
    assert inventory['status'] == 'PASS' and inventory['primitive_research_learner_fits'] == 295
    table_path = inventory_path.parent / 'PRIMITIVE_FITS.csv'
    assert digest(table_path) == inventory['table_sha256']
    with table_path.open(encoding='utf-8-sig', newline='') as stream:
        fits = list(csv.DictReader(stream))
    assert len(fits) == 295
    rows = []
    common = {'generation': 'NEXT5', 'ledger_section': 'NEXT5', 'seal_rows': 0,
        'reduced_new_uses': 0, 'new_meta_fits': 0, 'frozen_base_new_fits': 0,
        'primitive_fit_inventory_sha256': digest(inventory_path),
        'session_finalization_status': 'NEXT5_FINAL_AUDIT_LEDGER_RECONCILED'}
    for fit in fits:
        identity = ':'.join(fit[k] for k in ('category', 'experiment_id', 'split', 'scope', 'fold'))
        rows.append({**common, 'experiment_id': 'NEXT5_FIT:' + identity, 'parent_id': fit['experiment_id'],
            'phase': 'PRIMITIVE_RESEARCH_LEARNER_FIT', 'status': 'COMPLETE_VERIFIED', 'new_fits': 1,
            'learner': 'LightGBM', 'weighting': 'W3',
            'params': json.dumps({k: fit[k] for k in ('split', 'scope', 'fold', 'training_series',
                'training_rows', 'validation_series', 'validation_rows')}, sort_keys=True),
            'model': fit['fit_directory'] + '/model.joblib',
            'source_result_path': fit['fit_directory'] + '/RESULTS.json',
            'experiment_report_sha256': fit['fit_result_sha256'],
            'reason': 'Exactly one preregistered RP learner fit, counted once; aggregate candidate rows carry zero additional fits.'})
    evidence = ScalarEvidence()
    candidates = primary_tables(evidence)
    for candidate in candidates:
        cid = candidate['candidate']
        registration = read_json(BASE / 'registrations' / (cid + '.json'))
        rows.append({**common, 'experiment_id': 'NEXT5_RECIPE:' + cid, 'phase': 'SCIENTIFIC_RECIPE_SUMMARY',
            'hypothesis': registration['scientific_question'], 'feature_groups': candidate['family'],
            'status': 'FULL_FAIL_PROMOTION_GATE' if candidate['full_completed'] else 'SCREEN_FAIL',
            'new_fits': 0, 'primary_cv_mean': candidate['mean'] if candidate['full_completed'] else '',
            'primary_cv_median': candidate['median'] if candidate['full_completed'] else '',
            'primary_cv_worst': candidate['worst'] if candidate['full_completed'] else '',
            'delta_vs_primary': candidate['mean_delta'] if candidate['full_completed'] else '',
            'screen_score': candidate['mean'] if not candidate['full_completed'] else '',
            'bootstrap_replicates': candidate.get('paired_replicates', ''),
            'bootstrap_probability': candidate.get('paired_probability_delta_positive', ''),
            'bootstrap_ci_low': candidate.get('paired_ci_2_5', ''), 'bootstrap_ci_high': candidate.get('paired_ci_97_5', ''),
            'failed_gates': candidate['primary_gate_failure_reasons'], 'selected_role': '',
            'evidence_origin': candidate['score_population'], 'source_result_path': candidate['source_file'],
            'experiment_report_sha256': candidate['source_sha256'],
            'reason': 'ResearchPool held-out evaluation; all primitive fits counted in separate NEXT5_FIT rows, no new deployed role.'})
    assert len(rows) == 315 and sum(int(r['new_fits']) for r in rows) == 295
    assert len({r['experiment_id'] for r in rows}) == len(rows)
    assert all(set(row) <= set(fields) for row in rows)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\r\n')
    writer.writerows(rows)
    appended = buffer.getvalue().encode('utf-8')
    combined = original + appended
    parsed = list(csv.DictReader(io.StringIO(combined.decode('utf-8-sig'), newline='')))
    assert parsed[:276] == old_rows and len(parsed) == 591
    assert all(r['ledger_section'] == 'NEXT5' for r in parsed[276:])
    out = BASE / 'ledger' / args.tag
    out.mkdir(parents=True, exist_ok=False)
    (out / 'EXPERIMENT_LEDGER.csv').write_bytes(combined)
    # The standalone table needs its own header; only row bytes are appended to
    # the existing ledger, whose original header and every byte stay untouched.
    header_buffer = io.StringIO(newline='')
    csv.DictWriter(header_buffer, fieldnames=fields, lineterminator='\r\n').writeheader()
    (out / 'NEXT5_APPENDED_ROWS.csv').write_bytes(header_buffer.getvalue().encode('utf-8') + appended)
    assert target.read_bytes() == original
    with target.open('ab') as stream:
        stream.write(appended)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    assert target.read_bytes() == combined
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS',
        'original_rows': 276, 'appended_rows': 315, 'final_rows': 591,
        'primitive_learner_fit_rows': 295, 'recipe_summary_rows': 20, 'new_fits_sum': 295,
        'original_bytes_preserved_as_exact_prefix': True, 'original_cells_unchanged': True,
        'original_sha256': digest(retained), 'final_sha256': digest(target),
        'append_sha256': hashlib.sha256(appended).hexdigest(),
        'standalone_appended_table_sha256': digest(out / 'NEXT5_APPENDED_ROWS.csv'),
        'fit_inventory_sha256': digest(inventory_path), 'source_sha256': digest(Path(__file__)),
        'new_fits_created_by_this_script': 0}, immutable=True)
    progress('NEXT5 final ledger appended', 'Original 276 rows and all bytes preserved; 295 primitive fits plus 20 zero-additional-fit recipe summaries', phase='FINAL_AUDIT')


if __name__ == '__main__':
    main()
