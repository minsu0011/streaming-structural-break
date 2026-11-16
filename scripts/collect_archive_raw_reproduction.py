"""Preserve independent archive/raw reproduction evidence without model mutation."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import sha256, utc_now, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive')
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('Simple reproduction tag required')
    archive = Path(args.archive).resolve()
    source = (ROOT / 'artifacts/reproduction' / archive.stem).resolve()
    if not source.is_relative_to(ROOT / 'artifacts/reproduction'):
        raise RuntimeError('Reproduction source must remain in its workspace directory')
    frozen = read(ROOT / 'DEV_CHAMPION_CANDIDATES.json')
    with zipfile.ZipFile(archive) as handle:
        manifest_bytes = handle.read('24_SOURCE_MANIFEST.json')
        manifest = json.loads(manifest_bytes)
        records = {row['path']: row for row in manifest['files']}
        if any(name.startswith('data/') for name in handle.namelist()):
            raise RuntimeError('The source archive unexpectedly contained raw, caches or OOF')
        checked = {}
        for name, record in records.items():
            if name.startswith(('src/', 'configs/')) or name in (
                'main.py', 'requirements-lock.txt', 'scripts/train_final_artifacts.py',
                'scripts/reproduce_selected_cv.py', 'scripts/reproduce_final_model_from_raw.py',
            ):
                if sha256(source / name) != record['sha256'] or sha256(ROOT / name) != record['sha256']:
                    raise RuntimeError('A training or inference source changed after archive extraction: ' + name)
                checked[name] = record['sha256']
    if sha256(source / 'DEV_CHAMPION_CANDIDATES.json') != sha256(ROOT / 'DEV_CHAMPION_CANDIDATES.json'):
        raise RuntimeError('Source-copy selection differs')
    environment = read(ROOT / 'artifacts/validation/REPRODUCTION_REPORT.json')
    if environment['status'] != 'PASS' or environment['archive_sha256'] != sha256(archive):
        raise RuntimeError('Matching separate-environment reproduction must pass first')
    provenance = {
        'archive': archive.name, 'archive_sha256': sha256(archive),
        'archive_bytes': archive.stat().st_size,
        'archive_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'separate_source_copy': True, 'separate_virtual_environment': True,
        'python': environment['python'], 'source_copy': str(source.relative_to(ROOT)),
        'raw_access': 'Existing workspace-contained junction to canonical data/raw. Training code only reads raw; no filesystem read-only ACL is claimed.',
        'original_caches_or_oof_in_archive': False,
        'checked_training_and_inference_sources': checked,
    }
    pending = []
    summaries = {}
    for role, item in frozen['candidates'].items():
        eid = item['experiment_id']
        cv_relative = Path('artifacts/real/cv_reproduction') / args.tag / eid / 'RESULTS.json'
        final_relative = Path('artifacts/real/final_model_reproduction') / args.tag / role / 'RESULTS.json'
        cv = read(source / cv_relative)
        final = read(source / final_relative)
        if cv['status'] != 'PASS' or final['status'] != 'PASS' or cv['total_rows'] != 4014405 or final['total_rows'] != 4014405:
            raise RuntimeError('Complete all-row raw reproduction required for ' + role)
        if cv['split_sha256'] != frozen['split_sha256'] or final['selection_sha256'] != sha256(ROOT / 'DEV_CHAMPION_CANDIDATES.json'):
            raise RuntimeError('Reproduction split or selection identity differs')
        if final['seal_rows_fitted'] != 0 or final['seal_series_skipped_before_access'] != 2000:
            raise RuntimeError('Reproduction fitted an invalid scope')
        if any(row['different_float32_rows'] or row['ts_auc_abs_diff'] or row['save_load_max_abs_diff'] for row in cv['folds']):
            raise RuntimeError('CV numerical reproduction differs')
        if any(row['different_float32_rows'] for row in final['partitions']):
            raise RuntimeError('All-DEV model numerical reproduction differs')
        if (source / 'data/processed/oof' / eid).exists():
            raise RuntimeError('Unexpected original OOF arrays in extracted copy')
        oof_manifest = read(source / 'artifacts/handoff/selected_oof_manifests' / eid / 'MANIFEST.json')
        cache_identity = []
        for row in cv['folds']:
            fold = row['fold']
            original_model = ROOT / 'artifacts/models/real' / eid / f'fold_{fold}.joblib'
            reproduced_model = source / cv_relative.parent / f'fold_{fold}.joblib'
            if row['original_model_sha256'] != sha256(original_model) or row['model_sha256'] != sha256(reproduced_model):
                raise RuntimeError('CV model provenance changed')
            name = Path(oof_manifest['cache_directory']) / f'fold_{fold}.parquet'
            left, right = sha256(ROOT / name), sha256(source / name)
            if left != right:
                raise RuntimeError('Raw-rebuilt feature cache differs from original')
            cache_identity.append({'fold': fold, 'original_sha256': left, 'rebuilt_sha256': right,
                                   'byte_identical': True, 'bytes': (source / name).stat().st_size})
        original_final = ROOT / item['model_directory'] / 'model.joblib'
        reproduced_final = source / final_relative.parent / 'model.joblib'
        if final['original_model_sha256'] != sha256(original_final) or final['reproduced_model_sha256'] != sha256(reproduced_final):
            raise RuntimeError('All-DEV model provenance changed')
        for relative, report in ((cv_relative, cv), (final_relative, final)):
            if (ROOT / relative).exists():
                raise RuntimeError('Preserved report already exists; do not overwrite evidence')
            report.update(provenance)
            report['original_result_sha256'] = sha256(source / relative)
            report['rebuilt_cache_fold_identity'] = cache_identity
            report['rebuilt_cache_fold_bytes_match_original'] = True
            report['recorded_utc'] = utc_now()
            pending.append((ROOT / relative, report))
        cv['all_model_bytes_match_original'] = all(row['model_sha256'] == row['original_model_sha256'] for row in cv['folds'])
        summaries[role] = {'candidate': eid, 'cv_report': cv_relative.as_posix(), 'final_model_report': final_relative.as_posix(),
                           'rows_per_comparison': 4014405, 'cv_model_bytes_identical': cv['all_model_bytes_match_original'],
                           'final_model_bytes_identical': final['model_bytes_identical']}
    for path, report in pending:
        write_json(path, report)
    for row in summaries.values():
        row['cv_report_sha256'] = sha256(ROOT / row['cv_report'])
        row['final_model_report_sha256'] = sha256(ROOT / row['final_model_report'])
    summary = {'status': 'PASS', 'timestamp': utc_now(), **provenance,
               'selection_sha256': sha256(ROOT / 'DEV_CHAMPION_CANDIDATES.json'),
               'candidates': summaries, 'LOCAL_FINAL_SEAL_OPENED': False, 'reduced_targets_used': False}
    write_json(ROOT / 'artifacts/real/ARCHIVE_RAW_REPRODUCTION_REAL.json', summary)
    print('ARCHIVE RAW REPRODUCTION COLLECTED', len(summaries), 'roles; all cache bytes and numerical predictions match')


if __name__ == '__main__':
    main()
