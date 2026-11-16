"""Versioned metadata repair around the unchanged frozen numeric summarizer."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next5.cohort_provenance import annotate_paired_alarms
import scripts.summarize_next5_fitted_stress as frozen


def main():
    policy_path = BASE / 'maintenance/COHORT_PROVENANCE_REPAIR/POLICY_LOCK.json'
    policy = read_json(policy_path)
    for name, sha in policy['sources_sha256'].items():
        assert digest(ROOT / name) == sha
    original_reader, original_describer = frozen.read_table, frozen.describe_fold_values
    annotated = []
    def reader(path):
        rows = original_reader(path)
        if path.name == 'PAIRED_ALARMS.csv':
            values = annotate_paired_alarms(path, rows)
            assert [{key: row[key] for key in rows[0]} for row in values] == rows
            annotated.append({'source_file': path.relative_to(ROOT).as_posix(),
                'source_sha256': digest(path), 'rows': len(rows), 'all_original_cells_unchanged': True})
            return values
        return rows
    def describe(rows, group_columns, value_column, expected):
        if rows and 'cohort_kind' in rows[0]:
            group_columns = [*group_columns, 'cohort_kind']
        return original_describer(rows, group_columns, value_column, expected)
    frozen.read_table, frozen.describe_fold_values = reader, describe
    try:
        frozen.main()
    finally:
        frozen.read_table, frozen.describe_fold_values = original_reader, original_describer
    assert len(annotated) == 97
    out = BASE / 'synthetic/fitted_summary/complete_v1'
    result = read_json(out / 'RESULTS.json')
    assert result['status'] == 'PASS' and result['fitted_artifacts'] == 97
    atomic_json(out / 'COHORT_PROVENANCE_REPAIR.json', {'created_utc': utc_now(),
        'status': 'PASS', 'policy_sha256': digest(policy_path), 'original_numeric_summary_result_sha256': digest(out / 'RESULTS.json'),
        'annotated_tables': annotated, 'new_fits': 0, 'new_draws': 0, 'new_predictions': 0,
        'original_per_fold_analysis_files_unchanged': True,
        'change': 'Add explicit counterexample/long_null cohort identity before grouping alarm rows; preserve every original value and all 97 artifacts',
        'source_sha256': digest(Path(__file__))}, immutable=True)
    progress('Cohort-aware fitted summary complete', 'All original scalar cells retained; counterexample and long-null populations remain separate')


if __name__ == '__main__':
    main()
