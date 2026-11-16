"""Restore byte authority and initialize isolated NEXT7. No observations decoded."""
from pathlib import Path
import sys
import time
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, digest, read_json, utc_now, activity, note


def main():
    if (BASE / 'RESTORED_NEXT6_AUTHORITY.json').exists():
        raise RuntimeError('Preserve completed NEXT7 restoration')
    prompt = Path(r'C:\Users\minsu\.codex\attachments\3bc957e5-9f31-461f-ba47-5a09b1dc847d\pasted-text.txt')
    session = {
        'started_utc': utc_now(), 'started_epoch': time.time(), 'status': 'START_RECEIPT',
        'prompt_sha256': digest(prompt), 'research_minimum_seconds': 36000,
        'final_audit_target_seconds': 3600, 'no_idle_padding': True,
        'productive_accounting': 'union of live evidenced work intervals; monitoring/idle excluded',
        'pre_initialization_work_credited_seconds': 0,
        'scope': 'RP6400 only; prior grouped splits; no hidden/external competition tests',
    }
    if not (BASE / 'SESSION_NEXT7.json').exists():
        save('SESSION_NEXT7.json', session)
        save('USER_REQUEST_NEXT7.txt', prompt.read_text(encoding='utf-8'))
    else:
        assert read_json(BASE / 'SESSION_NEXT7.json')['prompt_sha256'] == digest(prompt)
    note('restore NEXT6 authority', 'Initialize isolated NEXT7 and verify delivered archive')
    with activity('restore NEXT6 byte authority', 'RESTORE'):
        receipt = read_json(ROOT / 'artifacts/next6_extension/HANDOFF_AUDIT_NEXT6_EXTENSION.json')
        archive = ROOT / 'artifacts/next6_extension/NEXT6_EXTENSION_RESEARCH_HANDOFF.zip'
        assert digest(archive) == receipt['archive_sha256']
        protected = {}
        source_matches = 0
        append_only_delivery_receipts = []
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None, 'Archive CRC failed'
            names = z.namelist()
            required = [
                'artifacts/next6/FINAL_AUDIT_NEXT6.json',
                'artifacts/next6/MODEL_COMPARISON_NEXT6.csv',
                'artifacts/next6/STRICT_INPUT_MODEL_COMPARISON_NEXT6.csv',
                'artifacts/next6/PAIRED_UNCERTAINTY_NEXT6.csv',
                'artifacts/next6/PROMOTION_DECISION_NEXT6.json',
                'artifacts/next6_extension/EXTENSION_RESEARCH_DECISION.json',
            ]
            for relative in required:
                assert relative in names, relative
            for name in names:
                if name.startswith(('src/', 'configs/', 'artifacts/next6/', 'artifacts/next6_extension/')):
                    path = ROOT / name
                    if path.is_file():
                        from hashlib import sha256
                        expected = sha256(z.read(name)).hexdigest()
                        actual = digest(path)
                        if actual != expected:
                            assert name == 'artifacts/next6_extension/EXPERIMENT_LEDGER.csv', name
                            old = z.read(name)
                            current = path.read_bytes()
                            assert current.startswith(old)
                            extra = current[len(old):].decode('utf-8')
                            assert 'raw_free_extension_package_and_extracted_demo,HANDOFF_AUDIT,COMPLETE' in extra
                            assert extra.count('\n') == 1
                            append_only_delivery_receipts.append({'file': name, 'archive_sha256': expected,
                                'current_sha256': actual, 'appended_row': extra.strip(),
                                'reason': 'The package completion ledger row was appended after ZIP creation.'})
                        else:
                            source_matches += 1
                        protected[name] = actual
        # Protect all original source and scripts, including items outside the ZIP.
        for directory in ('src', 'scripts', 'configs'):
            for p in (ROOT / directory).rglob('*'):
                if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.yml', '.mjs', '.html'):
                    relative = p.relative_to(ROOT).as_posix()
                    if '/next7/' not in relative and '__pycache__' not in relative:
                        protected[relative] = digest(p)
        for manifest in ('artifacts/next6/PROTECTED_BASELINE_SHA256.json',
                         'artifacts/next6_extension/INITIAL_NEXT6_FREEZE.json'):
            obj = read_json(ROOT / manifest)
            # The extension freeze uses a named files map; resolve from actual schema.
            candidates = obj.get('files', obj.get('sha256', obj.get('files_sha256', obj)))
            if isinstance(candidates, dict):
                for name, expected in candidates.items():
                    if isinstance(expected, str) and len(expected) == 64 and (ROOT / name).is_file():
                        assert digest(ROOT / name) == expected, name
                        protected[name] = expected
        for p in ROOT.glob('*'):
            if p.is_file():
                protected[p.name] = digest(p)
        for directory in ('artifacts/next6', 'artifacts/next6_extension'):
            for p in (ROOT / directory).glob('*'):
                if p.is_file():
                    protected[p.relative_to(ROOT).as_posix()] = digest(p)
        protected[archive.relative_to(ROOT).as_posix()] = receipt['archive_sha256']
        roles = read_json(ROOT / 'DEV_CANDIDATES_NEXT5.json')
        for role, info in roles['roles'].items():
            assert digest(ROOT / info['model_file']) == info['model_sha256'], role
            protected[info['model_file']] = info['model_sha256']
        for kind in ('J102', 'C205', 'B2_augmented_J', 'D2_expert'):
            root = ROOT / 'artifacts/next5/references/fits/PRIMARY/full' / kind
            for fold in range(5):
                d = root / f'fold_{fold}'
                expected = read_json(d / 'RESULTS.json')['model_sha256']
                assert digest(d / 'model.joblib') == expected
                protected[(d / 'model.joblib').relative_to(ROOT).as_posix()] = expected
        save('PROTECTED_BEFORE_NEXT7_SHA256.json', protected)
        save('RESTORED_NEXT6_AUTHORITY.json', {
            'status': 'PASS', 'created_utc': utc_now(), 'archive_sha256': digest(archive),
            'archive_CRC': 'PASS', 'archive_files': len(names), 'working_files_match_archive': source_matches,
            'protected_files': len(protected), 'frozen_roles': roles['roles'],
            'append_only_delivery_receipts': append_only_delivery_receipts,
            'authority': 'actual archive bytes and local hash-matched artifacts; prompt numbers not authority',
            'historical_DEV_score_not_matched_RP_score': True,
        })
    note('NEXT6 authority restored', f'{len(protected)} files protected; run guard before experiments')


if __name__ == '__main__':
    main()
