"""First adaptive review from closed A/B evidence and uncertainty warnings."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,progress
from src.next4.bottlenecks import record_board


def main():
    items=read_json(BASE/'bottlenecks/BOARD.json')['bottlenecks']
    risk=read_json(BASE/'runtime/SATURATION.json')
    assert len(list((BASE/'bottlenecks/history').glob('*.json')))==1
    for item in items:
        item['status']='ACTIVE' if item['id'] in ('B9','B10') else 'QUEUED'
        if item['id']=='B9':
            item['evidence_strength']=.85;item['solvability']=.45
            item['next_test']='Finish frozen C/D hypotheses; describe standalone diversity and mechanism slices before any conditional multi-expert nomination.'
            item['evidence']+=['artifacts/next4/bootstrap/A1_INNOVATION_PORTMANTEAU/RESULTS.json',
                'artifacts/next4/bootstrap/A2_MAGNITUDE_PORTMANTEAU/RESULTS.json','artifacts/next4/bootstrap/A3_SIGN_PORTMANTEAU/RESULTS.json']
        elif item['id']=='B10':
            item['next_test']='Prepare exactly two small nested pairwise learners using retained honest inner OOF. Give this independent objective family priority over additional feature families if ready; obey global saturation stop.'
            item['evidence']+=['artifacts/next3/nested/full/POLICY.json','docs/PAIRWISE_OBJECTIVE_POLICY.md']
        elif item['id']=='B4':
            item['status']='MONITOR_HIGH';item['evidence_strength']=1.
            item['next_test']='Enforce registered CASE G stop at 10 completed structural hypotheses if best full gain <0.0007; no relaxation for pending experiments. All intervals remain conditional on fitted OOF.'
            item['evidence']+=['artifacts/next4/runtime/SATURATION.json','artifacts/next4/maintenance/SCHEDULER_V2/RESTART.json']
        elif item['id']=='B2':
            item['next_test']='Complete paired SCALE mechanism and signed raw-SD direction diagnostics for B1/B2; require supported real SCALE improvement before synthetic robustness gate.'
            item['evidence']+=['artifacts/next4/experiments/B1_SPLIT_ENERGY_BRANCHES/full/RESULTS.json']
        elif item['id']=='B3':item['next_test']='Compute frozen full-OOF standalone Pearson/Spearman/within-time ranks/top-tail disagreement against J102, C205 and BL80.'
    record_board(items,'Closed Family A gains 0.000119-0.000229 with all three 95% intervals crossing zero. B1 full gain 0.000548 after screen 0.002603. '+
        'DEV saturation risk '+risk['risk']+'; keep finite registered C/D, slow new generation, prioritize an independent nested pairwise objective and descriptive validation.')
    progress('Bottleneck review 002','Independent signal and nested metric alignment active; HIGH DEV uncertainty is a standing constraint, not a new tuning target')
    print('NEXT4 BOTTLENECK REVIEW 002 COMPLETE')


if __name__=='__main__':main()
