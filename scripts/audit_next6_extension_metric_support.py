"""Make post-128 observation support explicit; preserve the locked metric code."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.next6_extension.common import *
from summarize_next6_extension import rows

def main():
    lock=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json');durations={'O1':1,'OB16':16,'REPEATED_SHOCK':385,'TEMP_SCALE_PULSE':32,'RETURN_NORMAL':256};out=[]
    for case,duration in durations.items():
        rr=rows(BASE/f'extended_synthetic/fresh/{case}_SERIES.csv')
        for horizon in (1024,4096):
            pp=[r for r in rr if int(r['horizon'])==horizon];assert len(pp)==512;eligible=[];none=[]
            for r in pp:
                scheduled=lock['tau_cycle'][int(r['seed'])%len(lock['tau_cycle'])];end=scheduled+duration;observations=max(0,horizon-end-128)
                (eligible if observations else none).append(r)
            out.append({'scenario':case,'horizon':horizon,'total_seeds':512,'observed_after_end_plus128_seeds':len(eligible),'no_followup_after128_seeds':len(none),'maximum_observed_FRESH_paired_effect_after128':max(float(r['paired_effect_after_128_max']) for r in eligible) if eligible else None,'all_observed_effects_exact_zero':all(float(r['paired_effect_after_128_max'])==0 for r in eligible) if eligible else None,
                'empty_tail_metric_zero_is_not_evidence':True,'recovery_definition':'first16 consecutive paired differences within tolerance; not necessarily permanent recovery'})
    csv_write('POST128_OBSERVATION_SUPPORT_AUDIT.csv',out);save('METRIC_SUPPORT_AUDIT_DONE.json',{'status':'PASS_WITH_EXPLICIT_SUPPORT_LIMIT','created_utc':utc_now(),'locked_metric_source_unchanged':True,'vacuous_zero_identified':'REPEATED_SHOCK at horizon1024 for seeds with scheduled512 has no observed point after last intervention end+128; exclude those from post128 evidence.','recovery_not_state_normality_or_permanent_coupling':True})
    print(out)
if __name__=='__main__':main()
