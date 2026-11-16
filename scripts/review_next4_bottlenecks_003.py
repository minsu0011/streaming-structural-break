"""Close objective/context menus and prioritize B2 mechanism evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,progress
from src.next4.bottlenecks import record_board


def main():
    assert len(list((BASE/'bottlenecks/history').glob('*.json')))==2
    items=read_json(BASE/'bottlenecks/BOARD.json')['bottlenecks']
    for item in items:
        item['status']='ACTIVE' if item['id'] in ('B9','B2') else 'QUEUED'
        if item['id']=='B9':
            item['next_test']='Complete the unchanged C/D/H menus, then apply the frozen F complement admission once to the closed independent-expert pool.'
        elif item['id']=='B2':
            item['evidence_strength']=.95
            item['next_test']='Inspect supported real SCALE and other mechanism effects. Evaluate B2 with fixed ARCH/tail decrease and outlier/long-null cohorts; preserve the failed primary CI gate.'
            item['evidence']+=['artifacts/next4/bootstrap/B2_LOG_ENERGY_SEMITAIL/RESULTS.json','artifacts/next4/robustness/B2_LOG_ENERGY_SEMITAIL/RESULTS.json']
        elif item['id']=='B10':
            item['status']='CLOSED_NEGATIVE_THIS_MENU';item['solvability']=.15
            item['next_test']='Report E1 full delta -0.00004675 and E2 failed screen; no extra objective, pair-count or regularization variants in NEXT4.'
            item['evidence']+=['artifacts/next4/experiments/E1_NESTED_PAIRWISE_LINEAR/full/RESULTS.json','artifacts/next4/experiments/E2_NESTED_PAIRWISE_INTERACTIONS/screen/RESULTS.json']
        elif item['id']=='B11':
            item['status']='CLOSED_NEGATIVE_THIS_MENU';item['solvability']=.1
            item['next_test']='Diagnose the G1 screen failure without refitting. Report G2 full delta +0.000001773; no context-subset mining.'
            item['evidence']+=['artifacts/next4/experiments/G1_H_CONTEXT_BASE/screen/RESULTS.json','artifacts/next4/experiments/G2_H_CONTEXT_NESTED_PAIRWISE/full/RESULTS.json']
        elif item['id']=='B4':
            item['status']='MONITOR_HIGH'
            item['next_test']='Keep all intervals conditional on selected fitted OOF. Do not relax the B2 CI requirement or draw additional seeds to change its sign.'
    record_board(items,'B2 full delta +0.001303673 with all five folds positive, but its primary paired CI includes zero. Fixed ALT 3/3 improve and H worst matched delta is -0.001430. E and G closed without material gain. Finish finite independent-signal menus and evaluate mechanism/stability rather than add variants.')
    progress('Bottleneck review 003','B2 mechanism/generalization and remaining independent signals active; E/G closed negative, DEV risk remains HIGH')
    print('NEXT4 BOTTLENECK REVIEW 003 COMPLETE',flush=True)


if __name__=='__main__':main()
