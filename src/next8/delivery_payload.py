"""Explicit raw-free delivery payload and safe ZIP verification helpers."""
from pathlib import Path, PurePosixPath
import hashlib
import shutil
import zipfile
import re
from src.next8.common import ROOT, BASE, read_json, digest, atomic

REQUIRED_REPORTS = (
    '00_READ_FIRST_NEXT8_KO.md', '01_GPT_HANDOFF_NEXT8.md',
    'NEXT8_PROTOCOL_LOCK.json', 'NEXT8_FEATURE_BUDGET.json', 'NEXT8_MODEL_MENU_LOCK.json', 'NEXT8_DATA_ACCESS_AUDIT.json',
    'FEATURE_LINEAGE_NEXT8.csv', 'FEATURE_CLASSIFICATION_NEXT8.csv', 'INVARIANT_FEATURE_REGISTRY.csv',
    'STABLE_NONLINEAR_FEATURE_REGISTRY.csv', 'VARIABLE_CANDIDATE_REGISTRY.csv', 'VARIABLE_CERTIFICATION_REGISTRY.json',
    'TARGET_DIRECTION_ATLAS.csv', 'CONDITIONAL_DIRECTION_RESULTS.csv', 'MOVEMENT_FORECAST_RESULTS.csv',
    'MOVEMENT_FACTOR_ABLATIONS.csv', 'MECHANICAL_RELATION_AUDIT.csv', 'PREDICTIVE_FACTOR_GRAPH.json',
    'STATE_TRANSITION_CONTRACT.md', 'STATE_TRANSITION_RESULTS.csv', 'STATE_OBSERVABILITY_RESULTS.csv', 'DIRECTION_RULES.json',
    'FEATURE_EXPLANATION_CARDS_KO.md', 'MODEL_COMPARISON_NEXT8.csv', 'NESTED_CV_NEXT8.csv',
    'PAIRED_UNCERTAINTY_NEXT8.csv', 'TIME_GENERALIZATION_NEXT8.csv', 'H_BACKGROUND_RESULTS_NEXT8.csv',
    'COUNTEREXAMPLE_RESULTS_NEXT8.csv', 'NEGATIVE_CONTROLS_NEXT8.csv', 'BOTTLENECK_HISTORY_NEXT8.csv',
    'EXPERIMENT_LEDGER_NEXT8.csv', 'LEAKAGE_AUDIT_NEXT8.json', 'LATENCY_NEXT8.json', 'DETERMINISM_NEXT8.json',
    'REPRODUCTION_NEXT8.json', 'FINAL_AUDIT_NEXT8.json', 'NEXT_SESSION_PLAN_NEXT8.md', 'SOURCE_MANIFEST_NEXT8.json',
    'SELF_HISTORY_CENTER_CONTROL_NEXT8.csv', 'SELF_HISTORY_BASELINE_LIMIT_KO.md')


def relative_member(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('Unsafe archive member')
    value = PurePosixPath(name)
    if value.is_absolute() or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Unsafe archive path traversal')
    return value.as_posix()


def select_payload(final_directory=None):
    """Select by artifact role, never by a favorable scientific result."""
    paths = {}

    def add(path, member=None):
        path = Path(path)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Only existing regular project files can enter payload: ' + str(path))
        member = relative_member(member or path.relative_to(ROOT).as_posix())
        if member in paths and digest(paths[member]) != digest(path):
            raise ValueError('Conflicting payload members: ' + member)
        paths[member] = path

    # All Python implementation/test sources are small; no environments/caches.
    for directory in (ROOT / 'src', ROOT / 'tests', ROOT / 'scripts/next8'):
        for path in directory.rglob('*.py'):
            if '__pycache__' not in path.parts:
                add(path)
    closure = read_json(BASE / 'runtime/FINAL_SOURCE_CLOSURE_NEXT8.json')
    for name, sha in closure['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
        add(ROOT / name)
    manifest = read_json(BASE / 'PORTABLE_RUNTIME_MANIFEST_NEXT8.json')
    for name, sha in manifest['files_sha256'].items():
        assert digest(ROOT / name) == sha, name
        add(ROOT / name)
    # Only the pinned ID/group/role/fold table supports the legacy test guard.
    metadata = ROOT / 'artifacts/validation/GROUP_ASSIGNMENTS.parquet'
    assert digest(metadata) == 'db0a562da579eac2251c2d82301de16d2211f7bcc5212bc52240f49464f5bbee'
    add(metadata)
    for name in ('pyproject.toml', 'pytest.ini'):
        if (ROOT / name).is_file():
            add(ROOT / name)
    for path in BASE.iterdir():
        if path.is_file() and path.suffix in ('.json', '.md', '.csv', '.txt', '.xml'):
            add(path)
    artifact_roots = ('selection', 'h_context_selection', 'parent_selection', 'forecasts', 'transitions',
                      'state_forecasts', 'graph_states', 'meta_state_inputs', 'early_state_inputs', 'models',
                      'integration', 'orientation_refits', 'early_baseline', 'baseline_conditioned', 'early_integration')
    concise_discovery = {'selection', 'h_context_selection', 'parent_selection'}
    discovery_names = {'SELECTION.json', 'GRAPH.json', 'DONE.json', 'MULTIPLICITY.json', 'H_STATE_ESTIMATORS.json'}
    for name in artifact_roots:
        for path in (BASE / name).rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.joblib'):
                if name not in concise_discovery or path.name in discovery_names or path.suffix == '.joblib':
                    add(path)
    evidence_roots = ('model_summary', 'movement_summary', 'synthetic_summary', 'feature_certification',
                      'evidence_synthesis', 'structural_inventory', 'discovery_diagnostics', 'figure_data',
                      'FIGURES', 'science_reports', 'execution_summary', 'center_control', 'negative_controls',
                      'bottleneck_reviews', 'registrations', 'repairs', 'audit', 'tests', 'full_replay_audit',
                      'final_integrity', 'final_tests', 'portable_fixture', 'computation_audit')
    for name in evidence_roots:
        for path in (BASE / name).rglob('*'):
            if not path.is_file():
                continue
            if path.suffix in ('.json', '.csv', '.md', '.xml', '.png', '.pdf', '.log'):
                add(path)
            elif name == 'portable_fixture' and path.name == 'PREDICTIONS.npz':
                add(path)
    # Whole-group bootstrap draws are aggregate uncertainty, not real-point OOF.
    for path in (BASE / 'model_evaluation').rglob('BOOTSTRAP.npz'):
        add(path)
    for name in ('FINAL_AUDIT_ADMISSION.json', 'FINAL_SOURCE_CLOSURE_NEXT8.json'):
        add(BASE / 'runtime' / name)
    if final_directory is not None:
        final_directory = Path(final_directory)
        assert set(REQUIRED_REPORTS) <= {p.name for p in final_directory.iterdir() if p.is_file()}
        for path in final_directory.iterdir():
            if path.is_file():
                add(path, path.name)
        for path in (BASE / 'FIGURES').iterdir():
            if path.suffix in ('.png', '.pdf', '.json'):
                add(path, 'FIGURES/' + path.name)
    assert not any(name.startswith('data/') or '/feature_cache/' in name or '/synthetic_controls/' in name for name in paths)
    assert not any(Path(name).suffix.lower() in ('.zip', '.npy', '.pkl', '.pickle') for name in paths)
    for name in paths:
        if Path(name).suffix.lower() == '.npz':
            assert ('/portable_fixture/' in name or '/frozen_reference_fixture/' in name or
                    ('/model_evaluation/' in name and name.endswith('/BOOTSTRAP.npz'))), name
    review_text_payload(paths)
    return dict(sorted(paths.items()))


def review_text_payload(paths):
    """Reject obvious literal credentials without printing their values."""
    patterns = (r'(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}', r'AKIA[0-9A-Z]{16}',
                r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                r'''(?i)(?:api_key|api_token|access_token|client_secret|password)\s*[:=]\s*["']([A-Za-z0-9_./+=-]{16,})["']''')
    flagged = []
    for name, path in paths.items():
        if Path(name).suffix.lower() not in ('.py', '.json', '.txt', '.md', '.yaml', '.yml', '.toml'):
            continue
        text = path.read_text(encoding='utf-8-sig')
        if any(re.search(pattern, text) for pattern in patterns):
            flagged.append(name)
    if flagged:
        raise ValueError('Possible literal credentials require payload review; values withheld: ' + ', '.join(flagged))
    return {'text_files_reviewed': sum(Path(n).suffix.lower() in ('.py', '.json', '.txt', '.md', '.yaml', '.yml', '.toml') for n in paths),
            'literal_credential_pattern_hits': 0, 'credential_files_and_environments_selected': False}


def materialize(paths, destination):
    destination = Path(destination).resolve()
    assert destination.is_relative_to((BASE / 'delivery').resolve()) and not destination.exists()
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, source in paths.items():
        name = relative_member(name)
        target = destination / name
        assert target.resolve().is_relative_to(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = digest(source)
        assert digest(target) == hashes[name]
    atomic(destination / 'PAYLOAD_CONTENT_MANIFEST_NEXT8.json', {
        'status': 'FROZEN', 'files_sha256': hashes, 'manifest_self_hash_excluded': True,
        'no_raw_competition_values_full_real_OOF_feature_caches_credentials_or_old_archives': True,
        'one_pinned_Parquet_is_ID_group_role_fold_metadata_only': True,
        'NPZ_members_are_synthetic_fixtures_or_group_bootstrap_draws_only': True})
    return hashes


def zip_payload(payload, archive):
    payload, archive = Path(payload).resolve(), Path(archive).resolve()
    assert not archive.exists()
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in sorted(payload.rglob('*')):
            if path.is_file():
                stream.write(path, relative_member(path.relative_to(payload).as_posix()))
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
    return digest(archive)


def verify_extract(archive, destination):
    """Check CRC, unique members, exact hashes and paths before trusting source."""
    destination = Path(destination).resolve()
    assert destination.is_relative_to((BASE / 'delivery').resolve()) and not destination.exists()
    with zipfile.ZipFile(archive) as stream:
        entries = stream.infolist()
        names = [relative_member(item.filename) for item in entries]
        assert len(set(names)) == len(names) and stream.testzip() is None
        manifest_name = 'PAYLOAD_CONTENT_MANIFEST_NEXT8.json'
        import json
        manifest = json.loads(stream.read(manifest_name))
        assert set(names) == set(manifest['files_sha256']) | {manifest_name}
        for name, sha in manifest['files_sha256'].items():
            assert hashlib.sha256(stream.read(name)).hexdigest() == sha, name
        destination.mkdir(parents=True, exist_ok=False)
        for name in names:
            target = destination / name
            assert target.resolve().is_relative_to(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                with stream.open(name) as source:
                    shutil.copyfileobj(source, output, length=1 << 20)
    for name, sha in manifest['files_sha256'].items():
        assert digest(destination / name) == sha
    return {'status': 'PASS', 'archive_sha256': digest(archive), 'CRC_verified': True,
            'manifest_files_verified': len(manifest['files_sha256']), 'extra_or_duplicate_members': 0,
            'fresh_extraction_sha256_verified': True, 'no_original_cache_or_OOF_required_for_inference': True}
