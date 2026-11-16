"""Explicit source/evidence payload; raw data and real point OOF are excluded."""
import csv
import json
import os
from pathlib import Path
import re
from src.next4.artifacts import ROOT,BASE,digest

REQUIRED='''RESEARCH_OVERVIEW_NEXT4.md RESEARCH_REPORT_NEXT4.md NEXT4_PREFLIGHT.json
HYPOTHESIS_FAMILY_LOCK.json CANDIDATE_BUDGET.json DEPENDENCE_V2_RESULTS.csv
SCALE_ASYMMETRY_RESULTS.csv DISTRIBUTION_SHIFT_RESULTS.csv SEQUENTIAL_LIKELIHOOD_RESULTS.csv
RANKING_OBJECTIVE_NEXT4.csv H_CONTEXT_RESULTS.csv COMPLEMENT_RESULTS_NEXT4.csv MODEL_RESULTS_NEXT4.csv
CV_RESULTS_NEXT4.csv PAIRED_BOOTSTRAP_NEXT4.csv ALT_RESULTS_NEXT4.csv REGIME_RESULTS_NEXT4.csv
MECHANISM_RESULTS_NEXT4.csv TIME_FRACTURE_NEXT4.csv COUNTEREXAMPLE_RESULTS_NEXT4.csv LONG_NULL_RESULTS_NEXT4.csv
BOTTLENECK_BOARD_INITIAL_NEXT4.json BOTTLENECK_BOARD_FINAL_NEXT4.json BOTTLENECK_HISTORY_NEXT4.csv
LEAKAGE_AUDIT_NEXT4.json PREFIX_INVARIANCE_NEXT4.json DETERMINISM_NEXT4.json LATENCY_NEXT4.json
DEV_CANDIDATES_NEXT4.json SEAL_GUARD_NEXT4.json SEAL_OPEN_RECOMMENDATION_NEXT4.md
PUBLIC_SUBMISSION_RECOMMENDATION_NEXT4.md FINAL_AUDIT_NEXT4.json REPRODUCTION_NEXT4.json
LIMITATIONS_NEXT4.md NEXT_SESSION_PLAN_NEXT4.md EXPERIMENT_LEDGER.csv RAW_DATA_REQUIRED_NEXT4.json'''.split()
FORBIDDEN={'.git','.crunchdao','.codex','.venv','.repro-venv','.figure-venv','__pycache__',
    'node_modules','handoff','payload','portable_staging','raw','.next-portable-v2','.next-portable-v3',
    'reproduction_workspace','reproduction_workspace_v1'}
SOURCE_EXT={'.py','.ps1','.md','.txt','.json','.toml','.ini','.cfg','.yaml','.yml','.lock'}
REPORT_EXT={'.json','.csv','.md','.xml','.png','.svg','.pdf','.log','.diff','.preserved'}
SECRET_PATTERNS=[re.compile(r'X-Amz-Signature=[a-fA-F0-9]{32,}'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s+[A-Za-z0-9+/]{30}'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b')]


def ordinary_files(directory):
    for parent,dirs,files in os.walk(directory,followlinks=False):
        parent=Path(parent)
        dirs[:]=[n for n in dirs if n not in FORBIDDEN and not (parent/n).is_symlink() and not (parent/n).is_junction()]
        for name in files:
            path=parent/name
            if not path.is_symlink() and not path.is_junction():yield path


def record_exclusion(path):
    # Aggregate support is useful; individual observation/label records are not.
    if path.name in {'03_RAW_DATA_MANIFEST.json','05_DATA_AUDIT.json','REAL_DATA_AUDIT.json','REAL_DATA_MANIFEST.json'}:
        return 'Historical raw-record manifest excluded by filename before JSON decoding'
    labels={'tau','tau_index','target','break_time','break_index','label'}
    if path.suffix=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as stream:columns=set(next(csv.reader(stream),[]))
        if columns.intersection({'id','dataset_id','series_id'}):
            if columns.intersection(labels):return 'Per-series outcome records'
            if columns.intersection({'time_online','prediction','score','observation','value'}):return 'Real per-series or point prediction/observation records'
    elif path.suffix=='.json':
        pending=[json.loads(path.read_text(encoding='utf-8-sig'))]
        while pending:
            value=pending.pop()
            if isinstance(value,dict):
                if isinstance(value.get('historical'),list) and isinstance(value.get('online'),list):return 'Observation-bearing fixture'
                if set(value).intersection({'id','dataset_id','series_id'}) and set(value).intersection(labels):return 'Per-series outcome record'
                pending.extend(value.values())
            elif isinstance(value,list):pending.extend(value)
    return None


def collect(*,require_final):
    paths=set();excluded={};missing=[n for n in REQUIRED if not (ROOT/n).is_file()]
    if require_final and missing:raise RuntimeError('Missing required NEXT4 reports: '+', '.join(missing))
    for name in REQUIRED:
        if (ROOT/name).is_file():paths.add(ROOT/name)
    for name in ('.gitattributes','.gitignore','.python-version','artifacts/next3/SESSION.json','artifacts/next3/runtime/USER_STOP_POINTER.json'):
        if (ROOT/name).is_file():paths.add(ROOT/name)
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in SOURCE_EXT|REPORT_EXT and path.name not in {'SOURCE_MANIFEST_NEXT4.json'}:
            if path.suffix=='.log':continue
            paths.add(path)
    for folder in ('src','scripts','tests','configs','docs','vendor/official'):
        paths.update(p for p in ordinary_files(ROOT/folder) if p.suffix in SOURCE_EXT)
    for folder in ('artifacts/validation',):
        paths.update(p for p in ordinary_files(ROOT/folder) if p.suffix in {'.json','.xml','.md'})
    paths.add(ROOT/'artifacts/validation/GROUP_ASSIGNMENTS.parquet')
    # Older science receipts/models remain interpretable, but repeated maintenance snapshots do not dominate the archive.
    old_roots={
        'next':['experiments/J102_SERIAL_VARIANCE_512','refits/J102_SERIAL_VARIANCE_512','forensic/J102_SERIAL_VARIANCE_512',
                'robustness/candidates/J102_SERIAL_VARIANCE_512','current_forensic'],
        'next2':['experiments/C205_DEPENDENCE','refits/C205_DEPENDENCE','canonical_deployment','forensic/BL2_C205_DEPENDENCE_80',
                 'robustness/C205_DEPENDENCE','independent_reproduction','final_tables/RESEARCH_TABLES'],
        'next3':['preflight','ledger','registrations','experiments','nested','forensic','diagnostics','final_tables/prepared_v2']}
    model_paths=set()
    for stage,folders in old_roots.items():
        for folder in folders:
            root=ROOT/'artifacts'/stage/folder
            for path in ordinary_files(root):
                if path.suffix in REPORT_EXT:paths.add(path)
                if path.suffix in {'.sbnext2','.joblib'} and stage in ('next','next2'):
                    model_paths.add(path)
    for path in ordinary_files(BASE):
        relative=path.relative_to(BASE)
        if relative.parts[0] in {'runtime','queue','checkpoints','handoff_inventory','delivery_staging','delivery_snapshot'}:
            continue
        if path.name in {'SESSION.json','LATEST.json','STATE.json'}:continue
        if path.suffix in REPORT_EXT or (relative.parts[0]=='maintenance' and path.suffix in {'.py','.ps1','.txt'}):paths.add(path)
        if path.suffix=='.joblib' and relative.parts[0] in {'experiments','robustness','refits','negative_controls','reproduction'}:
            model_paths.add(path)
        # These fixed-size arrays are aggregate bootstrap score draws, not point OOF.
        aggregate_scope=relative.parts[0] in {'bootstrap','forensic','robustness_forensic','joint_uncertainty','winner_gap'}
        if aggregate_scope and path.suffix in {'.npy','.npz'} and path.stat().st_size<=2*1024**2:
            if 'global_rows' not in path.name and 'predictions' not in path.name:paths.add(path)
    paths.update(model_paths)
    safe=set()
    for path in sorted(paths):
        relative=path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink() or path.is_junction():raise RuntimeError('Ordinary payload file required: '+str(relative))
        if relative.parts[0]=='data' or any(p in FORBIDDEN for p in relative.parts):raise RuntimeError('Forbidden payload path: '+str(relative))
        if path.suffix.lower() in {'.zip','.pkl','.pickle','.pyc','.nbi','.nbc','.env'}:raise RuntimeError('Forbidden payload extension')
        if any(word in path.name.lower() for word in ('credential','access_token','refresh_token','private_key')):
            excluded[relative.as_posix()]='Credential-like filename';continue
        if path.suffix=='.parquet' and relative.as_posix()!='artifacts/validation/GROUP_ASSIGNMENTS.parquet':raise RuntimeError('Unexpected parquet payload')
        if path.suffix in {'.joblib','.sbnext2'} and path not in model_paths:raise RuntimeError('Unexpected fitted model')
        reason=record_exclusion(path)
        if reason:
            if relative.as_posix() in REQUIRED:raise RuntimeError('Required report contains excluded records: '+str(relative))
            excluded[relative.as_posix()]=reason;continue
        if path.suffix in SOURCE_EXT|{'.csv','.xml','.log','.diff','.preserved'}:
            content=path.read_text(encoding='utf-8-sig')
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):raise RuntimeError('Credential-bearing content requires review: '+str(relative))
        safe.add(path)
    files={p.relative_to(ROOT).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(safe)}
    total=sum(r['bytes'] for r in files.values())
    if total>512*1024**2:raise RuntimeError('Review unexpectedly large raw-free payload before archiving')
    return sorted(safe),files,excluded,missing
