"""Interpret frozen center decomposition without altering A selection."""
from pathlib import Path
import sys
import csv
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

with activity('Summarize TRAINconstant versus Hmedian contribution for the9frozen A candidates','CENTER_INTERPRETATION','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
    records=[]
    for part in range(4):
        path=BASE/'center_decomposition'/f'PART_{part}.csv'
        assert digest(path)==read_json(path.with_name(f'PART_{part}_DONE.json'))['csv_sha256']
        with path.open(encoding='utf-8-sig',newline='') as f:records.extend(csv.DictReader(f))
    candidates=read_json(BASE/'candidates/outer_0/FROZEN.json')['normal_candidates'];result=[]
    for j in candidates:
        for mode in ['NORMAL','EARLY_LATE','EARLY_SELECT']:
            for h in [1,4,16]:
                rows=[r for r in records if int(r['feature'])==j and r['inner']=='' and r['mode']==mode and int(r['horizon'])==h]
                assert len(rows)==15
                constant=sum(float(r['Hmedian_to_TRAINconstant_gain']) for r in rows)/15
                own=sum(float(r['TRAINconstant_to_D1_gain']) for r in rows)/15
                result.append(dict(feature=j,mode=mode,horizon=h,Hmedian_to_TRAINconstant=constant,TRAINconstant_to_D1=own,
                    own_gain_CI_positive_panels=sum(float(r['TRAINconstant_to_D1_ci_low'])>0 for r in rows),
                    total_internal_panels=15,centering_fraction=constant/(constant+own) if constant+own>0 else None,
                    overlapping_internal_panels_not_independent=True))
    atomic(BASE/'CENTER_RESEARCH_FINDING_NEXT9.json',dict(status='COMPLETE',timestamp=utc_now(),records=result,
        interpretation='Hmedian is notMSE-optimalmean. Diagnostic fraction isaverageconstantgain/averageD1totalgain,notcausalvarianceexplained; mayexceed1ifownhistoryhurts. Noeligibilityrerankingornewmodel.',
        new_fits=0,break_labels_used=False))
    print([r for r in result if r['mode']=='NORMAL' and r['horizon']==1],flush=True)
