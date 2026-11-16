"""Restore all delivered NEXT8 evidence and establish a label-free RP guard."""
from pathlib import Path
import csv
import json
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next9
from src.next9.common import BASE, atomic, read_json, digest, activity, utc_now


def main():
    old = ROOT / read_json(BASE / 'SESSION_NEXT9.json')['previous_NEXT8_base']
    with activity('Restore NEXT8 delivered tables, scientific claims and byte authority', 'RESTORE', 'B9_NORMALIZATION_STATE_AVAILABILITY'):
        delivery = read_json(old / 'DELIVERY_RECEIPT_NEXT8.json')
        assert delivery['status'] == 'COMPLETE'
        protected = read_json(old / 'PROTECTED_BEFORE_NEXT8_SHA256.json')
        protected.update(read_json(old / 'runtime/FINAL_SOURCE_CLOSURE_NEXT8.json')['source_sha256'])
        for name, sha in protected.items():
            assert digest(ROOT / name) == sha, name
        frozen = read_json(old / 'PREDICTION_FREEZE_NEXT8.json')
        # The prior final audit is authoritative for its frozen prediction set;
        # full portable model/source/config membership is rechecked below.
        assert read_json(old / 'final_integrity/DONE.json')['status'] == 'COMPLETE'
        folder = Path(delivery['Desktop_directory'])
        expected = delivery['twenty_file_verification']['files_sha256']
        assert len(expected) == len(list(folder.iterdir())) == 20
        evidence = {}
        for name, sha in expected.items():
            p = folder / name
            assert digest(p) == sha, name
            if p.suffix == '.csv':
                with p.open(encoding='utf-8-sig', newline='') as stream:
                    rows = csv.DictReader(stream)
                    count = 0
                    for row in rows:
                        assert None not in row, name
                        count += 1
                    evidence[name] = dict(kind='CSV_ALL_ROWS_READ', rows=count, columns=rows.fieldnames, sha256=sha)
            elif p.suffix == '.json':
                value = read_json(p)
                evidence[name] = dict(kind='JSON_READ', top_level_keys=list(value), sha256=sha)
            elif p.suffix == '.md':
                value = p.read_text(encoding='utf-8-sig')
                evidence[name] = dict(kind='KOREAN_TEXT_READ', characters=len(value), sha256=sha)
            else:
                evidence[name] = dict(kind='PDF_BYTES_VERIFIED_SEPARATE_VISUAL_REVIEW_REQUIRED', sha256=sha)
        full = Path(delivery['Desktop_full_NEXT8_ZIP'])
        assert digest(full) == delivery['full_ZIP_sha256']
        with zipfile.ZipFile(full) as z:
            assert z.testzip() is None
            payload = json.loads(z.read('PAYLOAD_CONTENT_MANIFEST_NEXT8.json'))
            members = len(z.namelist())
        for name, sha in read_json(old / 'PORTABLE_RUNTIME_MANIFEST_NEXT8.json')['files_sha256'].items():
            assert digest(ROOT / name) == sha, name
            protected[name] = sha
        docs = ('final_reports/00_READ_FIRST_NEXT8_KO.md', 'final_reports/RESEARCH_REPORT_NEXT8.md',
                'NEXT_SESSION_PREVIOUS_MOVEMENT_HYPOTHESIS_NEXT8.json', 'BASELINE_PROVENANCE_NEXT8.json',
                'NEXT8_PROTOCOL_LOCK.json', 'FEATURE_LINEAGE_NEXT8.csv',
                'center_control/NORMAL.json', 'center_control/EARLY.json',
                'evidence_synthesis/SCIENTIFIC_SYNTHESIS.json', 'RAW_ENGINE_GATE_RESULTS_NEXT8_KO.md')
        for name in docs:
            p = old / name
            assert p.is_file(), name
            value = p.read_text(encoding='utf-8-sig')
            protected[p.relative_to(ROOT).as_posix()] = digest(p)
            evidence[name] = dict(kind='PRIOR_RESEARCH_READ', characters=len(value), sha256=digest(p))
        science = read_json(old / 'evidence_synthesis/SCIENTIFIC_SYNTHESIS.json')
        assert science['scientific_status_subject_to_final_integrity_audit'] == 'NEXT8_NO_CERTIFIED_VARIABLE_FEATURE'
        atomic(BASE / 'PROTECTED_BEFORE_NEXT9_SHA256.json', protected)
        atomic(BASE / 'RESTORED_NEXT8_AUTHORITY_NEXT9.json', dict(status='PASS', timestamp=utc_now(),
               previous_base=old.relative_to(ROOT).as_posix(), previous_Desktop_directory=str(folder),
               previous_Desktop_files_sha256=expected, full_zip_sha256=delivery['full_ZIP_sha256'],
               full_zip_entries=members, prior_scientific_status=science['scientific_status_subject_to_final_integrity_audit'],
               previous_source_model_protected_count=len(protected), evidence_read=evidence,
               previous_prediction_freeze_sha256=digest(old / 'PREDICTION_FREEZE_NEXT8.json'),
               prior_scores_read_for_restoration_not_movement_candidate_ranking=True,
               original_ranking_and_nested_splits_not_redefined=True))
    with activity('Verify RP-only bytes and label-free metadata without decoding tau or break target', 'GUARD', 'B9_NORMALIZATION_STATE_AVAILABILITY'):
        import numpy as np
        from src.next5.guard import ResearchGuard, ShadowAccessError
        guard = ResearchGuard(ROOT)
        assert (len(guard.research_ids), len(guard.shadow_ids), len(guard.seal_ids)) == (6400, 1600, 2000)
        for ids in (guard.shadow_ids, guard.seal_ids):
            try:
                guard.admit([min(ids)], purpose='NEXT9 forbidden ID admission probe')
            except ShadowAccessError:
                pass
            else:
                raise AssertionError('Forbidden membership admitted')
        previous = read_json(old / 'PREFLIGHT_NEXT8.json')
        for name, sha in previous['raw_sha256'].items():
            assert digest(ROOT / 'data/raw' / name) == sha, name
        directory = ROOT / 'data/next5/research_5d54e316ff5bf25d'
        assert digest(directory / 'MANIFEST.json') == previous['cache_manifest_sha256']
        manifest = read_json(directory / 'MANIFEST.json')
        for name, sha in manifest['files_sha256'].items():
            assert digest(directory / name) == sha, name
        with np.load(directory / 'series.npz', allow_pickle=False) as source:
            ids = source['dataset_id']
            guard.admit(ids, purpose='NEXT9 label-free series metadata')
            assert set(map(int, ids)) == guard.research_ids and len(ids) == 6400
            offsets = source['online_offsets']
            historical_offsets = source['historical_offsets']
            folds = source['fold']
        # Never ResearchArrays here: its eager loader also decodes tau/target.
        assert offsets[-1] == 3203157
        assert all(guard.split.assignment(int(s))[1] == int(f) for s, f in zip(ids, folds))
        caches = []
        for p in (old / 'feature_cache').glob('*/MANIFEST.json'):
            m = read_json(p)
            if m.get('status') == 'COMPLETE' and m['identity']['width'] == 153:
                caches.append((p, m))
        assert len(caches) == 1, 'Require one explicit authoritative NEXT8 feature cache'
        fp, fm = caches[0]
        for name, sha in fm['files_sha256'].items():
            assert digest(fp.parent / name) == sha, name
        with np.load(fp.parent / 'series.npz', allow_pickle=False) as source:
            np.testing.assert_array_equal(source['dataset_id'], ids)
            feature_metadata_keys = source.files
            assert not {'target', 'tau', 'breakOccurred'} & set(feature_metadata_keys)
        guard.check_state()
        result = dict(status='PASS', timestamp=utc_now(), RP_series=6400, RP_online_rows=int(offsets[-1]),
                      split_sha256=guard.split.digest, shadow_lock_sha256=guard.shadow_lock_sha,
                      raw_sha256=previous['raw_sha256'], cache_manifest_sha256=digest(directory / 'MANIFEST.json'),
                      RP_cache=directory.relative_to(ROOT).as_posix(),
                      feature_cache=fp.parent.relative_to(ROOT).as_posix(), feature_manifest_sha256=digest(fp),
                      feature_identity=fm['identity'], feature_metadata_keys=feature_metadata_keys,
                      normalization_contract_reused_without_refit=True, raw_Parquet_decoding=False,
                      decoded_break_labels=0, decoded_tau=0, original_NPZ_rows_not_opened=True,
                      selected_NPZ_members=['dataset_id', 'online_offsets', 'historical_offsets', 'fold'],
                      SHADOW_ACCESS=0, SEAL_ACCESS=0, REDUCED_NEW_USAGE=0, PUBLIC_NEW_USAGE=0, CLOUD_NEW_USAGE=0,
                      all_previous_sources_models_read_only=True, application_guard_not_OS_sandbox=True)
        atomic(BASE / 'PREFLIGHT_NEXT9.json', result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
