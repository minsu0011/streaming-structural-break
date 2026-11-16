"""Fixed-gate failure funnel using authoritative corrected internal evidence."""
from pathlib import Path
import sys
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now,csv_write


with activity('Decompose common-range and fixed orientation gate failures without new selection','ORIENTATION_FORENSICS','B2_NO_REPLICATED_OPPOSITE_ORIENTATION'):
    assert read_json(BASE/'LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json')['status']=='PASS'
    counts=defaultdict(Counter);opposites=[];sources={}
    for path in (BASE/'orientation_logloss_corrected').glob('*/DONE.json'):
        for mode in ['NORMAL','EARLY']:
            p=path.parent/mode/'SELECTION_FROZEN.json';selection=read_json(p);sources[p.relative_to(BASE).as_posix()]=digest(p)
            for r in selection['records']:
                c=counts[(mode,r['feature'])];c['panels']+=1
                if r['status']!='COMPLETE':c['common_range_insufficient']+=1;continue
                c['common_range_supported']+=1
                e=r['evidence'];a=e['orientation']['first'];b=e['orientation']['second']
                if a['status']!='SUPPORTED' or b['status']!='SUPPORTED':c['orientation_bootstrap_insufficient']+=1;continue
                c['orientation_bootstrap_supported']+=1
                opposite=(a['ci_low']>0 and b['ci_high']<0) or (b['ci_low']>0 and a['ci_high']<0)
                if not opposite:c['no_opposite_orientation']+=1;continue
                c['opposite_orientation']+=1
                record=dict(scope=path.parent.name,mode=mode,feature=r['feature'],split=r['split'],pair_index=r['pair_index'],
                            method=r['method'],difference_Holm_p=r['difference_Holm_p'],common_low=r['domain']['low'],common_high=r['domain']['high'])
                gates=[('Holm',r['difference_Holm_p']<.05)]
                for ref in ['O1','STATE_ONLY']:
                    contrast=e['contrasts'][r['method']+'__vs__'+ref]
                    for metric in ['TS_AUC','logloss_gain']:
                        lo=contrast[metric]['ci_low'];record[ref+'_'+metric+'_ci_low']=lo
                        gates.append((ref+'_'+metric,lo is not None and lo>0))
                passed=True
                for name,good in gates:
                    if not good:c['independent_fail_'+name]+=1
                    passed &= good
                    if passed:c['cumulative_pass_'+name]+=1
                record['full_fixed_gate_pass']=passed;opposites.append(record)
    rows=[dict(mode=key[0],feature=key[1],**value) for key,value in sorted(counts.items())]
    csv_write('ORIENTATION_FAILURE_FUNNEL_NEXT9.csv',rows)
    csv_write('INTERNAL_RAW_OPPOSITE_CASES_NEXT9.csv',opposites)
    assert len(opposites)==9 and not any(r['full_fixed_gate_pass'] for r in opposites)
    assert all(r['O1_TS_AUC_ci_low']<0 for r in opposites)
    atomic(BASE/'ORIENTATION_FAILURE_FINDING_NEXT9.json',dict(status='COMPLETE',timestamp=utc_now(),
        panels=sum(r['panels'] for r in rows),supported=sum(r['common_range_supported'] for r in rows),raw_opposite_cases=len(opposites),
        all_raw_opposites_fail_O1_TS_AUC_improvement=True,all_fixed_gate_pass_cases=0,
        internal_overlapping_procedures_not_independent_replications=True,no_new_fit_or_threshold_change=True,
        interpretation='Direction-sign contrasts alone do not establish useful conditional F. Every raw opposite case fails the fixed-nonlinear comparator under the prespecified TS-AUC criterion, independently of logloss.',
        authoritative_signed_logloss=True,source_sha256=sources,
        outputs_sha256={p:digest(BASE/p) for p in ['ORIENTATION_FAILURE_FUNNEL_NEXT9.csv','INTERNAL_RAW_OPPOSITE_CASES_NEXT9.csv']}))
