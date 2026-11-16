"""Explicit raw-free NEXT5 payload extending the verified prior delivery."""
import os
from pathlib import Path
from src.next5.artifacts import ROOT, BASE, read_json, digest
from src.next4.handoff_payload import SOURCE_EXT, REPORT_EXT, SECRET_PATTERNS, record_exclusion
SOURCE_EXT = SOURCE_EXT | {'.ps1'}

REQUIRED = '''00_GPT_READ_FIRST_NEXT5.md 01_GPT_HANDOFF_NEXT5.md NEXT5_PREFLIGHT.json
NEXT5_SHADOW_LOCK.json NEXT5_SHADOW_GUARD.json NEXT5_SPLIT_LOCK.json NEXT5_RANDOMNESS_LOCK.json
HYPOTHESIS_FAMILY_LOCK_NEXT5.json CANDIDATE_BUDGET_NEXT5.json
SCALE_UNIFICATION_RESULTS.csv PERSISTENCE_RESULTS.csv WINDOWED_EVIDENCE_RESULTS.csv
EPROCESS_RESULTS.csv ROBUST_NULL_RESULTS.csv MEAN_SPECIALIST_RESULTS.csv CROSS_FAMILY_RESULTS.csv
MODEL_RESULTS_NEXT5.csv CV_RESULTS_NEXT5.csv PAIRED_BOOTSTRAP_NEXT5.csv ALT_RESULTS_NEXT5.csv
REGIME_RESULTS_NEXT5.csv MECHANISM_RESULTS_NEXT5.csv TIME_FRACTURE_NEXT5.csv BREAK_AGE_NEXT5.csv
COUNTEREXAMPLE_RESULTS_NEXT5.csv LONG_NULL_RESULTS_NEXT5.csv OUTLIER_IMPULSE_NEXT5.csv
FINALIST_MANIFEST_NEXT5.json SHADOW_RESULTS_NEXT5.csv BOTTLENECK_BOARD_INITIAL_NEXT5.json
BOTTLENECK_BOARD_FINAL_NEXT5.json BOTTLENECK_HISTORY_NEXT5.csv LEAKAGE_AUDIT_NEXT5.json
PREFIX_INVARIANCE_NEXT5.json DETERMINISM_NEXT5.json LATENCY_NEXT5.json DEV_CANDIDATES_NEXT5.json
SEAL_GUARD_NEXT5.json SEAL_OPEN_RECOMMENDATION_NEXT5.md PUBLIC_SUBMISSION_RECOMMENDATION_NEXT5.md
REPRODUCTION_NEXT5.json FINAL_AUDIT_NEXT5.json LIMITATIONS_NEXT5.md NEXT_SESSION_PLAN_NEXT5.md
EXPERIMENT_LEDGER.csv RAW_DATA_REQUIRED_NEXT5.json'''.split()
FORBIDDEN = {'.git', '.crunchdao', '.codex', '.venv', '.figure-venv', '.repro-venv', '__pycache__',
    'node_modules', 'handoff', 'payload', 'portable_staging',
    'raw', 'reproduction_workspace', 'reproduction_workspace_v1', 'pytest_tmp', 'pytest_cache', 'numba_cache'}


def ordinary_files(directory):
    for parent, dirs, files in os.walk(directory, followlinks=False):
        parent = Path(parent)
        dirs[:] = [name for name in dirs if name not in FORBIDDEN
            and not (parent / name).is_symlink() and not (parent / name).is_junction()]
        for name in files:
            path = parent / name
            if not path.is_symlink() and not path.is_junction():
                yield path


def aggregate_array(path):
    """Filename and directory allow-list; no real point arrays are decoded."""
    parts = path.relative_to(BASE).parts
    if path.suffix not in ('.npy', '.npz'):
        return False
    if any(word in path.name.lower() for word in ('prediction', 'oof', 'global_row', 'feature', 'signature', 'descriptor')):
        return False
    if parts[0] in ('bootstrap', 'forensic', 'selection_uncertainty'):
        return path.stat().st_size <= 32 * 1024**2
    if parts[0] == 'stability' and ('BOOTSTRAP' in parts or 'reference_bootstrap' in parts):
        return path.stat().st_size <= 2 * 1024**2
    if parts[0] == 'synthetic' and ('reference_bootstrap' in parts or 'ANALYSIS' in parts):
        return path.name.endswith('_DRAWS.npy') or 'reference_bootstrap' in parts
    if parts[0] == 'synthetic_late_break' and 'reference_draws' in parts:
        return path.stat().st_size <= 2 * 1024**2
    return False


def collect(*, require_final):
    retained = BASE / 'preflight/retained/SOURCE_MANIFEST_NEXT4.json'
    prior = read_json(retained)
    assert digest(retained) == digest(ROOT / 'SOURCE_MANIFEST_NEXT4.json')
    assert prior['status'] == 'FROZEN_NEXT4_PACKAGE_SOURCE_MANIFEST'
    paths, excluded = set(), {}
    # Already reviewed prior payload paths are byte-verified without parsing historical data records.
    for name, expected in prior['files'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file()
        if name != 'EXPERIMENT_LEDGER.csv':
            assert digest(path) == expected['sha256'], 'Prior payload changed: ' + name
        paths.add(path)
    paths.add(ROOT / 'SOURCE_MANIFEST_NEXT4.json')
    for folder in ('src/next5', 'scripts', 'tests', 'configs/next5', 'docs'):
        paths.update(p for p in ordinary_files(ROOT / folder) if p.suffix in SOURCE_EXT)
    for path in ROOT.iterdir():
        if path.is_file() and ('NEXT5' in path.name or path.name in REQUIRED) and path.suffix in SOURCE_EXT | REPORT_EXT:
            if path.name != 'SOURCE_MANIFEST_NEXT5.json':
                paths.add(path)
    terminal_jobs = {p.parent.name for p in (BASE / 'jobs').glob('*/STATE.json')
        if read_json(p)['status'] in ('COMPLETE', 'QUARANTINED')}
    for path in ordinary_files(BASE):
        parts = path.relative_to(BASE).parts
        if parts[0] in ('runtime', 'handoff_inventory', 'delivery_staging', 'delivery_snapshot'):
            continue
        if path.name in ('SESSION.json', 'LATEST.json', 'REVIEW_DUE.json'):
            continue
        if path.name == 'STATE.json' and not (parts[0] == 'jobs' and len(parts) > 1 and parts[1] in terminal_jobs):
            continue
        if parts[0] == 'jobs' and (len(parts) < 2 or parts[1] not in terminal_jobs):
            continue
        if path.suffix in REPORT_EXT or (parts[0] in ('locks', 'maintenance', 'preflight') and path.suffix == '.txt'):
            paths.add(path)
        if path.suffix == '.joblib' and parts[0] in ('references', 'experiments', 'stability', 'negative_controls'):
            paths.add(path)
        if aggregate_array(path):
            paths.add(path)
    safe = []
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        assert path.is_file() and not path.is_symlink() and not path.is_junction()
        assert relative.parts[0] != 'data' and not set(relative.parts).intersection(FORBIDDEN), str(relative)
        assert path.suffix.lower() not in ('.zip', '.pkl', '.pickle', '.pyc', '.nbi', '.nbc', '.env')
        if any(word in path.name.lower() for word in ('credential', 'access_token', 'refresh_token', 'private_key')):
            excluded[relative.as_posix()] = 'Credential-like filename'
            continue
        # The prior manifest was previously audited and is independently byte pinned above.
        if relative.as_posix() not in prior['files']:
            reason = record_exclusion(path)
            if reason:
                if path.name in REQUIRED:
                    raise RuntimeError('Required report includes excluded record: ' + str(relative))
                excluded[relative.as_posix()] = reason
                continue
            if path.suffix in SOURCE_EXT | {'.csv', '.xml', '.log', '.diff', '.preserved'}:
                content = path.read_text(encoding='utf-8-sig')
                assert not any(pattern.search(content) for pattern in SECRET_PATTERNS), 'Credential pattern: ' + str(relative)
        safe.append(path)
    files = {p.relative_to(ROOT).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in safe}
    missing = [name for name in REQUIRED if name not in files]
    if require_final and missing:
        raise RuntimeError('Missing required NEXT5 payload files: ' + ', '.join(missing))
    if sum(r['bytes'] for r in files.values()) > 1024**3:
        raise RuntimeError('Review unexpectedly large raw-free payload')
    return safe, files, excluded, missing
