from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

with activity('Freeze actual public generators and full fresh-ZIP reproduction protocol','ARCHIVE_PROTOCOL','B9_NORMALIZATION_STATE_AVAILABILITY'):
    paths=['src/next9/delivery_payload.py','scripts/next9/reproduce_next9_archive.py',
        'scripts/next9/replay_next9_public_entries.py','scripts/next9/replay_next9_synthetic_portable.py',
        'scripts/next9/infer_next9.py','scripts/next9/audit_next9_integrity.py','scripts/next9/tests/test_delivery_payload.py']
    for path in paths:compile((ROOT/path).read_text(encoding='utf-8'),path,'exec')
    atomic(BASE/'ARCHIVE_REPRODUCTION_IMPLEMENTATION_LOCK_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),
        source_sha256={p:digest(ROOT/p) for p in paths},payload_tests_sha256=digest(BASE/'tests/PAYLOAD_TESTS.xml'),
        source_integrity_research_receipt_sha256=digest(BASE/'final_integrity/PRE_FINAL_PROTECTED_NEXT9.json'),
        coverage='Full168syntheticcases x10fittedprocedures x9models x512observations, actualM7publicgenerator168series, threecanonicalpublicroles9historicalfixtures',
        raw_free=True,no_new_model_fit=True,short_fresh_cache_relative_path='nc',
        fresh_process_original_workspace_assets_denied=True,legacy_venv_packages_reused_no_install=True,
        final_audit_time='Import actual completed child CPU-backed ledger/job receipts bytewise; union only, never parent wait or idle',
        archive_self_hash_receipt_external=True,original_failed_files_never_removed=True))
    script='scripts/next9/replay_next9_public_entries.py'
    task=dict(id='PUBLIC_ENTRY_REPLAY',status='READY',script=script,source_sha256=digest(ROOT/script),arguments=[],
        gates=['ARCHIVE_REPRODUCTION_IMPLEMENTATION_LOCK_NEXT9.json','portable_fixture/outer_0/NORMAL/DONE.json'],
        phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True)
    queue=read_json(BASE/'runtime/TASK_QUEUE.json');assert not any(x['id']==task['id'] for x in queue)
    atomic(BASE/'registrations/PUBLIC_ENTRY_REPLAY_TASK_NEXT9.json',dict(timestamp=utc_now(),task=task))
    atomic(BASE/'runtime/TASK_QUEUE.json',queue+[task],immutable=False)
