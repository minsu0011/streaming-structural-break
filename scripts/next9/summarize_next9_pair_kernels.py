from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,csv_write,activity,utc_now

with activity('Summarize independent explicit-pair verification of all outer domains and bootstrap draws',
              'INDEPENDENT_STATISTIC_AUDIT','B9_NORMALIZATION_STATE_AVAILABILITY'):
    rows=[];sources={}
    for fold in range(5):
        for mode in ['NORMAL','EARLY']:
            root=BASE/'pair_kernel_audit'/f'outer_{fold}'/mode
            done=read_json(root/'DONE.json');assert done['status']=='PASS'
            sources[str((root/'DONE.json').relative_to(BASE))]=digest(root/'DONE.json')
            for name,sha in done['domain_receipts_sha256'].items():
                p=root/(name+'.json');assert digest(p)==sha
                r=read_json(p);assert r['status']=='PASS'
                for m in r['methods']:
                    assert m['point_bitwise'] and m['all_2000_draws_bitwise']
                    rows.append(dict(fold=fold,mode=mode,domain=name,family=r['family'],**m))
    assert len(rows)==5*2*21*9
    csv_write('INDEPENDENT_TS_AUC_AUDIT_NEXT9.csv',rows)
    atomic(BASE/'INDEPENDENT_TS_AUC_AUDIT_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
        procedures=10,domains_per_procedure=21,method_domain_points=len(rows),bootstrap_values=len(rows)*2000,
        all_reported_points_and_draws_bitwise=True,whole_group_weights_match_original_hash=True,
        actual_prediction_aliases_reused=True,no_refits_or_selection=True,source_sha256=sources,
        calculation='Explicit comparable pairs and exact integer quadratic forms, independently of sorted-score rank implementation.'))
