"""Explicit raw-free NEXT9 payload, preserving pinned legacy dependencies."""
from pathlib import Path
import shutil
import zipfile
import json
from src.next9.common import ROOT,BASE,read_json,digest,atomic
from src.next8.delivery_payload import relative_member,review_text_payload

REQUIRED_REPORTS=('00_READ_FIRST_NEXT9_KO.md','RESEARCH_REPORT_NEXT9.md','NEXT9_PROTOCOL_LOCK.json',
    'NEXT9_CANDIDATE_BUDGET.json','TARGET_FREE_MOVEMENT_SCREEN.csv','MOVEMENT_ELIGIBILITY.csv',
    'H_CENTER_COMPARISON.csv','EARLY_LATE_MOVEMENT_RESULTS.csv','MOVEMENT_STATE_CONTRACT.md',
    'MOVEMENT_STATE_SUPPORT.csv','COMMON_RANGE_RESULTS.csv','CONDITIONAL_ORIENTATION_RESULTS.csv',
    'VARIABLE_CERTIFICATION_NEXT9.json','SELF_TRANSITION_RESULTS.csv','SELF_TRANSITION_MODELS.json',
    'TRANSITION_PATH_FEATURES.csv','CROSS_FACTOR_RESULTS.csv','PREDICTIVE_FACTOR_GRAPH_NEXT9.json',
    'MODEL_COMPARISON_NEXT9.csv','CV_RESULTS_NEXT9.csv','PAIRED_UNCERTAINTY_NEXT9.csv',
    'TIME_GENERALIZATION_NEXT9.csv','H_GENERALIZATION_NEXT9.csv','COUNTEREXAMPLES_NEXT9.csv',
    'LEAKAGE_AUDIT_NEXT9.json','LATENCY_NEXT9.json','DETERMINISM_NEXT9.json','REPRODUCTION_NEXT9.json',
    'FINAL_AUDIT_NEXT9.json','BOTTLENECK_HISTORY_NEXT9.csv','EXPERIMENT_LEDGER_NEXT9.csv',
    'FEATURE_CARDS_NEXT9_KO.md','NEXT_SESSION_PLAN_NEXT9.md')

def select_payload(*,final=False):
    paths={}
    def add(path,member=None):
        path=Path(path)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):raise ValueError('Invalid payload source: '+str(path))
        name=relative_member(member or path.relative_to(ROOT).as_posix())
        if name in paths and digest(paths[name])!=digest(path):raise ValueError('Conflicting payload member: '+name)
        paths[name]=path
    old=ROOT/read_json(BASE/'SESSION_NEXT9.json')['previous_NEXT8_base']
    for name,sha in read_json(old/'runtime/FINAL_SOURCE_CLOSURE_NEXT8.json')['source_sha256'].items():
        assert digest(ROOT/name)==sha,name
        add(ROOT/name)
    for name,sha in read_json(old/'PORTABLE_RUNTIME_MANIFEST_NEXT8.json')['files_sha256'].items():
        assert digest(ROOT/name)==sha,name
        add(ROOT/name)
    for directory in [ROOT/'src/next9',ROOT/'scripts/next9']:
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.py','.ini') and '__pycache__' not in path.parts:add(path)
    for path in [ROOT/'artifacts/next8/CURRENT_RUN.json',ROOT/'artifacts/next9/CURRENT_RUN.json',old/'SESSION_NEXT8.json',old/'PORTABLE_RUNTIME_MANIFEST_NEXT8.json']:
        add(path)
    metadata=ROOT/'artifacts/validation/GROUP_ASSIGNMENTS.parquet'
    assert digest(metadata)=='db0a562da579eac2251c2d82301de16d2211f7bcc5212bc52240f49464f5bbee';add(metadata)
    for path in BASE.iterdir():
        if path.is_file() and path.suffix in ('.json','.md','.csv','.txt','.xml'):add(path)
    # Complete fitted outer procedures; no per-observation predictions.
    for tree in ['joint_heads','cross_and_tabular','meta_correction','self_transitions']:
        for fold in range(5):
            directory=BASE/tree/f'outer_{fold}'
            for path in directory.rglob('*'):
                if path.is_file() and path.suffix in ('.json','.joblib'):add(path)
    # Concise numerical evidence, source locks, original failures, and tests.
    trees=['pair_kernel_audit','state_transport','fitted_model_audit','candidates','orientation_logloss_corrected','orientation_selection','outer_orientation_evaluation','outer_movement_evaluation',
        'outer_self_transition_evaluation','outer_cross_evaluation','outer_model_evaluation','frozen_predictions',
        'center_decomposition','observability','FIGURES','bottleneck_reviews','registrations','repairs','tests',
        'portable_fixture','portable_verification','replay_rank_verification','negative_controls','counterexamples',
        'audit','final_integrity','ledger','live_replay','reverse_order_replay']
    for tree in trees:
        for path in (BASE/tree).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:continue
            if path.suffix in ('.json','.csv','.md','.xml','.png','.pdf','.log'):add(path)
            elif tree=='repairs' and path.suffix=='.py':add(path)
            elif tree=='portable_fixture' and path.name=='PREDICTIONS.npz':add(path)
            elif tree=='outer_model_evaluation' and path.name=='BOOTSTRAP.npz':add(path)
    for name in ['FINAL_AUDIT_ADMISSION_NEXT9.json','FINAL_SOURCE_CLOSURE_NEXT9.json']:
        add(BASE/'runtime'/name)
    # Failed and successful queue executions remain auditable without logs
    # containing input data; own worker logs contain only task progress/errors.
    for path in (BASE/'runtime/queue').rglob('*'):
        if path.is_file() and path.suffix in ('.json','.log'):add(path)
    for path in (BASE/'runtime/jobs').glob('*.json'):add(path)
    for name in ['pyproject.toml','pytest.ini']:
        if (ROOT/name).is_file():add(ROOT/name)
    if final:
        for name in REQUIRED_REPORTS:add(BASE/name,name)
        for path in (BASE/'FIGURES').iterdir():
            if path.suffix in ('.png','.pdf','.json'):add(path,'FIGURES/'+path.name)
    for name in paths:
        assert not name.startswith('data/') and '/feature_cache/' not in name and '/state_cache/' not in name
        assert Path(name).suffix.lower() not in ('.zip','.npy','.pkl','.pickle')
        if Path(name).suffix.lower()=='.npz':
            assert '/portable_fixture/' in name or '/frozen_reference_fixture/' in name or ('/outer_model_evaluation/' in name and name.endswith('/BOOTSTRAP.npz')),name
    review_text_payload(paths)
    return dict(sorted(paths.items()))

def materialize(paths,destination):
    destination=Path(destination).resolve()
    assert destination.is_relative_to((BASE/'delivery').resolve()) and not destination.exists()
    destination.mkdir(parents=True);hashes={}
    for name,source in paths.items():
        name=relative_member(name);target=destination/name
        assert target.resolve().is_relative_to(destination)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        hashes[name]=digest(source);assert digest(target)==hashes[name]
    atomic(destination/'PAYLOAD_CONTENT_MANIFEST_NEXT9.json',dict(status='FROZEN',files_sha256=hashes,
        manifest_self_hash_excluded=True,no_competition_observations_labels_or_full_real_OOF=True,
        NPZ_only_synthetic_fixtures_or_whole_group_bootstrap=True,pinned_group_table_metadata_only=True))
    return hashes

def zip_payload(payload,archive):
    payload=Path(payload).resolve();archive=Path(archive).resolve()
    assert not archive.exists()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for path in sorted(payload.rglob('*')):
            if path.is_file():stream.write(path,relative_member(path.relative_to(payload).as_posix()))
    with zipfile.ZipFile(archive) as stream:assert stream.testzip() is None
    return digest(archive)

def verify_extract(archive,destination):
    destination=Path(destination).resolve()
    assert destination.is_relative_to((BASE/'delivery').resolve()) and not destination.exists()
    with zipfile.ZipFile(archive) as stream:
        names=[relative_member(i.filename) for i in stream.infolist()]
        assert len(names)==len(set(names)) and stream.testzip() is None
        manifest_name='PAYLOAD_CONTENT_MANIFEST_NEXT9.json';manifest=json.loads(stream.read(manifest_name))
        assert set(names)==set(manifest['files_sha256'])|{manifest_name}
        import hashlib
        for name,sha in manifest['files_sha256'].items():assert hashlib.sha256(stream.read(name)).hexdigest()==sha,name
        destination.mkdir(parents=True)
        for name in names:
            target=destination/name;assert target.resolve().is_relative_to(destination)
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(stream.read(name))
    for name,sha in manifest['files_sha256'].items():assert digest(destination/name)==sha,name
    return dict(status='PASS',files=len(names),archive_sha256=digest(archive),extracted=str(destination),CRC_and_manifest_exact=True)
