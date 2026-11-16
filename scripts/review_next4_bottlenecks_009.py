"""Prioritize actual-artifact validation after D2's fixed ALT/H completion."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,progress
from src.next4.bottlenecks import record_board


def main():
    assert len(list((BASE/'bottlenecks/history').glob('*.json')))==8
    result=read_json(BASE/'robustness/D2_DISCRETE_SCALE_MIXTURE/RESULTS.json');assert result['status']=='PASS'
    items=read_json(BASE/'bottlenecks/BOARD.json')['bottlenecks']
    for item in items:
        if item['id']=='B2':
            item['next_test']='D2 passes the original supported SCALE and all four fold-0 decrease cases. All three ALT point deltas are positive but their intervals include zero; H-held-out overall delta is -0.00021543, minimum matched regime -0.00268446. Review fixed SCALE transfer, then validate only the original highest SCALE-CI-ranked all-DEV artifact. A2/C3 routes are closed; no recipe changes.'
            item['evidence']+=['artifacts/next4/robustness/D2_DISCRETE_SCALE_MIXTURE/RESULTS.json',
                'artifacts/next4/synthetic/candidates/D2_DISCRETE_SCALE_MIXTURE_full_fold0/RESULTS.json',
                'artifacts/next4/robustness_screen/C3_SHAPE_RATIO_DISTRIBUTION.json']
        elif item['id']=='B5':
            item['next_test']='Finish original D2/D3/H memory and attribution, fixed sign reflection, and actual all-DEV finalist confirmation. D2 improves original SCALE and MIXED while DEPENDENCE delta is -0.000763; do not call it an across-the-board improvement. Final-artifact confirmation keeps all original generator seeds, sample sizes, thresholds and memory arithmetic.'
            item['evidence']+=['scripts/register_next4_finalist_stress.py','scripts/diagnose_next4_finalist_memory.py',
                'artifacts/next4/diagnostic_policy/SIGN_REFLECTION_POLICY.json']
    record_board(items,'D2 completed the prospectively authorized 20 unchanged ALT/H fits after four strictly positive fixed decrease scenarios. ALT point gains persist but remain uncertain; H overall is mildly negative. Highest original SCALE CI lower bound makes D2 the first deployment-validation nominee, not a new performance champion. Complete the actual artifact and all original lower-priority suites before the separate time-gated final audit.')
    progress('Bottleneck review 009','D2 actual-artifact validation is the next constrained test; primary performance still frozen, H/DEPENDENCE limitations retained')
    print('NEXT4 BOTTLENECK REVIEW 009 COMPLETE',flush=True)


if __name__=='__main__':main()
