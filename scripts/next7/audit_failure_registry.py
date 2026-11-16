"""Keep repaired engineering failures separate from negative scientific results."""
from pathlib import Path
import argparse
import re
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note

REPAIRS = {
    'a2a93bc9eb744ea7b15fe539edb09215': ('ARCHIVE_AUTHORITY_LEDGER_APPEND',
        'The archive preceded the final original ledger append; reviewed append and frozen authority preserved.',
        ('RESTORED_NEXT6_AUTHORITY.json', 'audit/PROTECTED_AUTHORITY_RESEARCH_001.json')),
    '2572a915c90b4d6dbed5f483eb2e0e3d': ('NUMBA_BOOLEAN_FLOAT_COMPATIBILITY',
        'Boolean-to-float conversion unsupported by the local compiler; explicit numeric conditional restored the same equations before calibration.',
        ('audit/CORE_TESTS_2026-09-13T024046_374520+0000.json', 'G1_IMPLEMENTATION_LOCK.json')),
    'd2275d23de6c4bb79f81b8785505cbc2': ('W3_ZERO_WEIGHT_VALIDATION',
        'Original W3 allows zero-weight time strata. Validation now accepts nonnegative weights with positive total; no fitted model existed before repair.',
        ('G1_CORRECTION_BUG_REPAIR_001.json', 'G1_ALL_OUTER_PREDICTIONS_FROZEN.json', 'G1_OUTER_SCORE_RELEASE.json')),
    'fc9f80162dba44b6a8768c682457a7a3': ('PAIRED_CENSORED_METRIC_SUPPORT',
        'Use the common observable support for candidate and reference recovery metrics. No intervals had been published and no count rows changed.',
        ('SYNTHETIC_UNCERTAINTY_BUG_REPAIR_001.json', 'SYNTHETIC_UNCERTAINTY_DONE_NEXT7.json')),
    '97e608aa3d5f44dab85da4851f66265c': ('G3_H_STANDARDIZATION_MEMORY_LAYOUT',
        'Identical parent advanced-column layout restored exact standardization. One in-memory optimizer attempt discarded before any serialization, outer prediction or score.',
        ('G3_STANDARDIZATION_BUG_REPAIR_001.json', 'G3_ALL_OUTER_PREDICTIONS_FROZEN.json', 'G3_OUTER_SCORE_RELEASE.json')),
    'feba393f038741499cfa69458c2b0c72': ('PORTABLE_PAYLOAD_MISSING_CONFIGS',
        'Required seven feature configuration files added to the isolated payload. Original failure retained;90 paths and92160 outputs reproduced exactly.',
        ('audit/REPRODUCTION_PAYLOAD_CONFIG_REPAIR_RESOLVED.json', 'reproduction/research_002/DONE.json')),
    'f755c9a06a2a468dac3a9b5fbf9c6a9b': ('STATIC_VIEWER_NUMERIC_INDEX_TYPE',
        'Integer-valued JSON display state converted to integer list index; numerical evidence unchanged. Corrected browser and static images reviewed.',
        ('audit/VIEWER_V3_STATIC_INDEX_REPAIR.json', 'visuals/v3/BUILD_DONE.json',
         'visuals/v3/BROWSER_REVIEW.json', 'audit/VIEWER_V3_VISUAL_REVIEW_001.json')),
    '4e7cc8a538594704b4e9773f028de85a': ('HANDOFF_PREFLIGHT_WRONG_REGIME_COUNT',
        'An inline preflight guessed six regimes; the original fixed scope source defines five. Corrected preflight reads original scope names with empty synthetic arrays. Writer, scores and models were unchanged.',
        ('audit/CORE_HANDOFF_BINDING_PREFLIGHT_001_FAILED.json',
         'audit/CORE_HANDOFF_BINDING_PREFLIGHT_002.json', 'audit/CORE_HANDOFF_BINDING_REPAIR_001.json')),
    '6ec7b7c2028546f3af18381fa3f2c772': ('FULL_SUITE_PLOT_COLORBAR_KEYWORD_COMPATIBILITY',
        'Installed Matplotlib requires colorbar tick labels as a keyword argument. The earlier transitive numerical import was also removed by reading the original registered case order. Both failure receipts and the empty failed output directory remain; all368 display cells and both final figures verified with original scientific results unchanged.',
        ('audit/FULL_SUITE_PLOT_IMPORT_FAILURE_001.json', 'audit/FULL_SUITE_PLOT_COLORBAR_FAILURE_001.json',
         'audit/FULL_SUITE_PLOT_REPAIR_001.json', 'visuals/classifier_stress/full_suite_001/MANIFEST.json',
         'audit/CLASSIFIER_FULL_SUITE_VISUAL_REVIEW_001.json')),
}


def run(label, final_audit=False):
    if not re.fullmatch('[a-z0-9_]+', label):
        raise ValueError('Simple NEXT7-owned receipt label required')
    if final_audit:
        from scripts.next7.benchmark_runtime import audit_gate
        audit_gate()
    relative = 'audit/failure_registry/' + label
    entries, evidence, unresolved = [], {}, []
    with activity('Reconcile engineering failure registry ' + label,
                  'FINAL_AUDIT' if final_audit else 'FAILURE_FORENSIC', 'AUTONOMOUS_MAINTENANCE'):
        failed = sorted((read_json(path) for path in (BASE / 'ledger').glob('*.json')),
                        key=lambda row: row['started_epoch'])
        failed = [row for row in failed if row['status'] == 'FAILED']
        for failure in failed:
            identity = failure['id']
            known = REPAIRS.get(identity)
            if known is None:
                classification, repair, names = 'UNMAPPED_FAILURE', 'Requires explicit diagnosis and completed reproduction evidence.', ()
                status = 'UNRESOLVED'
                unresolved.append(identity)
            else:
                classification, repair, names = known
                for name in names:
                    evidence[name] = digest(BASE / name)
                status = 'REPAIRED_AND_REPRODUCED'
            entries.append({'origin': 'ACTIVITY_LEDGER', 'id': identity, 'started_utc': failure['started_utc'],
                'task': failure['task'], 'classification': classification, 'status': status,
                'original_error': failure['error'], 'repair': repair, 'evidence_files': '|'.join(names),
                'scientific_promotion_gate_relaxed': False})
        for item in read_json(BASE / 'runtime/TASK_QUEUE.json'):
            directory = BASE / 'runtime/queue' / item['id']
            terminal_path = directory / 'TERMINAL.json'
            if not terminal_path.exists():
                continue
            terminal = read_json(terminal_path)
            if terminal['status'] == 'COMPLETE':
                continue
            resolution_path = directory / 'RESOLUTION.json'
            status = 'UNRESOLVED'
            names = []
            if resolution_path.exists():
                resolution = read_json(resolution_path)
                assert resolution['terminal_sha256'] == digest(terminal_path)
                for name, sha in resolution['evidence_sha256'].items():
                    assert digest(BASE / name) == sha
                    evidence[name] = sha
                    names.append(name)
                status = resolution['status']
            if status not in ('REPAIRED_AND_REPRODUCED', 'EXPECTED_FAILURE_TEST'):
                unresolved.append(item['id'])
            evidence[terminal_path.relative_to(BASE).as_posix()] = digest(terminal_path)
            entries.append({'origin': 'MAIN_RESEARCH_QUEUE', 'id': item['id'],
                'task': item['script'], 'classification': 'UTF8_BOM_HEADER' if item['id'] == 'classifier_calibration' else 'QUEUE_FAILURE',
                'status': status, 'original_error': str(terminal.get('reason', terminal.get('error', 'See immutable terminal and attempt logs'))),
                'repair': 'Completed strict existing M0/M3 calibration parity and all1024 new calibration paths' if item['id'] == 'classifier_calibration' else '',
                'evidence_files': '|'.join(names), 'scientific_promotion_gate_relaxed': False})
        test_errata = {'H_LENGTH_DIAGNOSTICS_TESTS_001.xml': {
            'classification': 'TEST_EXPECTATION_FLOAT_OPERATION_ORDER',
            'repair': 'Materialized signed decision differences compared before division; production diagnostic implementation unchanged.',
            'passed_receipt': 'H_LENGTH_DIAGNOSTICS_TESTS_002.xml'},
        }
        for name, record in test_errata.items():
            evidence['audit/' + name] = digest(BASE / 'audit' / name)
            evidence['audit/' + record['passed_receipt']] = digest(BASE / 'audit' / record['passed_receipt'])
        isolated = read_json(BASE / 'audit/OWNED_CRASH_RECOVERY_TESTS_001.json')
        assert isolated['status'] == 'PASS'
        result = {'status': 'PASS' if not unresolved else 'UNRESOLVED', 'created_utc': utc_now(),
            'source_sha256': digest(__file__), 'main_ledger_failed_work_units': len(failed),
            'main_queue_failed_terminal_units': sum(row['origin'] == 'MAIN_RESEARCH_QUEUE' for row in entries),
            'unresolved_engineering_failures': unresolved, 'entries': entries,
            'test_development_errata': test_errata,
            'isolated_crash_tests': 'Expected owned process interruptions and quarantine are successful recovery tests, outside the main scientific queue.',
            'scientific_negative_result_separate': 'All15 NEXT7 candidates have negative mean delta versus M0. Those results are valid completed experiments, not software failures to erase or rerun until positive.',
            'scientific_evidence_sha256': digest(BASE / 'SCORE_EVIDENCE_CONSOLIDATED.json'),
            'source_evidence_sha256': evidence}
        csv_write(relative + '/SOFTWARE_FAILURE_REGISTER.csv', entries)
        save(relative + '/DONE.json', result)
    note('Engineering failure registry reconciled', f'{len(failed)} failed work units and completed queue resolutions; unresolved={len(unresolved)}',
         'AUTONOMOUS_MAINTENANCE', phase='FINAL_AUDIT' if final_audit else 'PRODUCTIVE_RESEARCH')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--final-audit', action='store_true')
    args = parser.parse_args()
    run(args.label, args.final_audit)
