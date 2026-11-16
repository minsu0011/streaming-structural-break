"""Close direct timing and prioritize fitted-model memory and transfer evidence."""
from pathlib import Path
import sys
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,progress,utc_now
from src.next4.bottlenecks import record_board


def main():
    assert len(list((BASE/'bottlenecks/history').glob('*.json')))==5
    prior=read_json(BASE/'bottlenecks/BOARD.json');items=prior['bottlenecks']
    interval=(datetime.fromisoformat(utc_now())-datetime.fromisoformat(prior['created_utc'])).total_seconds()
    timing=read_json(BASE/'latency/B2_DIRECT/RESULTS.json');assert timing['status']=='PASS' and timing['direct_runtime_within_3x']
    hazard=read_json(BASE/'hazard_pressure/RESULTS.json');assert hazard['status']=='PASS'
    for item in items:
        if item['id']=='B2':
            item['next_test']='Evaluate all fixed B1/A2/C3/D2/D3 synthetic decrease conditions, then only admitted unchanged ALT/H replications. B2 direct runtime passes but its original primary and SCALE CI failures remain. Rank any fully validated robustness candidates by the already frozen original SCALE lower bound.'
            item['evidence']+=['artifacts/next4/latency/B2_DIRECT/RESULTS.json','artifacts/next4/diagnostic_policy/CONDITIONAL_ROBUSTNESS_VALIDATION.json']
        elif item['id']=='B5':
            item['next_test']='Run the 12 frozen fitted-model synthetic suites. Separate local null score drift from accumulating first-alarm opportunities, and use fixed outlier contribution/recovery diagnostics. D3 h=1/32 has positive ideal-null existing-odds log drift for mean +/-0.25; assess actual fitted-model behavior before conclusions.'
            item['evidence']+=['artifacts/next4/hazard_pressure/RESULTS.json','artifacts/next4/diagnostic_policy/MEMORY_DIAGNOSTIC_POLICY.json','artifacts/next4/diagnostic_policy/OUTLIER_ATTRIBUTION_POLICY.json']
        elif item['id']=='B4':
            item['next_test']='B2-B1 point gap +0.00075585 has CI [-0.00051849,+0.00195732], P=0.8835. B2 is bootstrap-best in 0.7935 of original draws, not a posterior probability. Maintain closed model menus and original failed primary CI.'
            item['evidence']+=['artifacts/next4/winner_gap/RESULTS.json']
    record_board(items,'Isolated five-pair direct timing completed. All three B2 runtime ratios pass 3x; this does not change scientific promotion. The analytic D3 prior-pressure diagnostic and both diagnostic-oracle suites passed without real data fits. Continue supported SCALE transfer and fixed fitted-memory diagnostics. Review interval was '+format(interval/60,'.2f')+' minutes; record this small cadence overrun rather than claiming every review met exactly 45 minutes.')
    progress('Bottleneck review 006','B2 direct timing closed; fixed-model synthetic memory and supported mechanism transfer remain the two active questions')
    print('NEXT4 BOTTLENECK REVIEW 006 COMPLETE',flush=True)


if __name__=='__main__':main()
