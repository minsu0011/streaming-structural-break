"""Require all corrected internal losses and unchanged frozen selections."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now


with activity('Audit all twenty signed-loss reconstructions and invariant selection','LOGLOSS_REPAIR_AUDIT','B9_NORMALIZATION_STATE_AVAILABILITY'):
    records=[]
    for fold in range(5):
        for inner in [None,0,1,2]:
            scope=f'outer_{fold}'+('' if inner is None else f'_inner_{inner}')
            p=BASE/'orientation_logloss_corrected'/scope/'DONE.json';r=read_json(p)
            assert r['status']=='COMPLETE' and r['selection_unchanged'] and r['new_fits']==0
            records.append(dict(scope=scope,pair_panels=r['pair_panels'],recomputed=r['recomputed_complete_panels'],sha256=digest(p)))
    assert sum(r['pair_panels'] for r in records)==8592
    assert sum(r['recomputed'] for r in records)==6457
    atomic(BASE/'LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json',dict(status='PASS',timestamp=utc_now(),records=records,
        total_panels=8592,recomputed_supported_panels=6457,all_selections_unchanged=True,no_new_fits=True,
        old_internal_logloss_invalid=True,authoritative_internal_tree='orientation_logloss_corrected',
        corrected_outer_source='scripts/next9/evaluate_next9_outer_orientation_v4.py',
        original_rank_TS_AUC_bootstrap_and_orientation_unchanged=True,
        diagnosis='Unsigned uint8 unary negation wrapped positive labels to255; corrected loss casts labels and probabilities tofloat64 before arithmetic.'))
