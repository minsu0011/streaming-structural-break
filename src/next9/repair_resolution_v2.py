"""Reconcile preserved failed attempts with explicit successful replacements."""
import re
from src.next9.common import BASE,read_json,digest


def reconcile_failed_attempts():
    fixed={
        'ORIENTATION_V2_outer_4_inner_0':('ORIENTATION_NUMERIC_RECOVERY_outer_4_inner_0','LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json'),
        'PROFILE_FROZEN_M7_F0':('PROFILE_FROZEN_M7_F0_MEMBERSHIP_V4','repairs/REFERENCE_MEMBERSHIP_SERIALIZATION_REPAIR_NEXT9.json'),
        'QUALIFY_FAST_HEADS_ALL10_MENU_V2':('QUALIFY_FAST_HEADS_ALL10_MAP_V3','FAST_HEADS_QUALIFICATION_NEXT9.json')}
    records=[]
    for path in sorted((BASE/'runtime/queue').glob('*/attempt_*/DONE.json')):
        r=read_json(path)
        if r.get('returncode',0)==0:continue
        name=path.parent.parent.name
        if name in fixed:replacement,receipt=fixed[name]
        elif re.fullmatch(r'LIVE_RP_F[0-4]_(NORMAL|EARLY)_P[01]_V2_META_V3_MEMBERSHIP_V4_FAST_V5',name):
            replacement=name+'_CACHE_V6'
            receipt='EAGER_SCHEMA_CACHE_QUALIFICATION_NEXT9.json'
        else:
            match=re.fullmatch(r'STATE_TRANSPORT_F([0-4])(?:_FULL_HELDOUT_V2)?',name)
            assert match,'Unreconciled failed attempt: '+name
            replacement=f'STATE_TRANSPORT_F{match.group(1)}_FULL_HELDOUT_V2_FINAL_ROWS_V3'
            receipt='STATE_TRANSPORT_DONE_NEXT9.json'
        terminal=BASE/'runtime/queue'/replacement/'TERMINAL.json'
        assert read_json(terminal)['status']=='COMPLETE',replacement
        assert read_json(BASE/receipt)['status']=='PASS',receipt
        records.append(dict(failed_task=name,failed_attempt=path.parent.name,returncode=r['returncode'],
            original_attempt_sha256=digest(path),successful_replacement=replacement,
            replacement_terminal_sha256=digest(terminal),repair_receipt=receipt,repair_receipt_sha256=digest(BASE/receipt),
            original_failure_retained=True))
    return records
