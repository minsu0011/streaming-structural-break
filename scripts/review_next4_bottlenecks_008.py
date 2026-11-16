"""Update the fixed research routes after completed B2/A2 counterexamples."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,progress
from src.next4.bottlenecks import record_board

def main():
    assert len(list((BASE/'bottlenecks/history').glob('*.json')))==7
    items=read_json(BASE/'bottlenecks/BOARD.json')['bottlenecks']
    for item in items:
        if item['id']=='B2':
            item['next_test']='B2 heavy-tail decreases fail with paired deltas -0.00623355 and -0.05363612, both intervals negative. A2 also fails both fixed heavy-tail decrease cases and its conditional ALT/H route is closed. Finish original B1/C3/D2/D3 robustness screens; no B2 confidence-interval repair, new candidate or blend search.'
            item['evidence']+=['artifacts/next4/synthetic/candidates/B2_LOG_ENERGY_SEMITAIL_all_dev/RESULTS.json','artifacts/next4/robustness_screen/A2_MAGNITUDE_PORTMANTEAU.json']
        elif item['id']=='B5':
            item['next_test']='B2 repeated-outlier score excess is worse than BL80: mean excess difference +0.033498, last-64 difference +0.057189 on 512 pairs. TreeSHAP on a fixed different 128-seed subset implicates upper-energy branches as well as legacy ARCH states. Finish remaining fixed memory suites and the prospectively locked original-DEV sign reflection; no feature edits.'
            item['evidence']+=['artifacts/next4/memory_diagnostics/B2_LOG_ENERGY_SEMITAIL_all_dev/RESULTS.json','artifacts/next4/outlier_attribution/B2_LOG_ENERGY_SEMITAIL_all_dev/RESULTS.json','docs/NEXT4_B2_FAILURE_INTERPRETATION.md']
        elif item['id']=='B6':item['status']='DIAGNOSED_B2_PERSISTENCE_WORSE'
    record_board(items,'Both preserved worker failures are resolved without a new fit or changed scientific prediction. B2 native-crash continuation preserved all 116 existing files and reused 18 completed analyses. Its heavy-tail decrease and repeated-outlier limitations are now measured. A2 fails the original conditional robustness screen. Board review interval exceeded 45 minutes while consolidating these scientific results; preserve the actual timestamp and do not backdate.')
    progress('Bottleneck review 008','B2 fixed failure modes resolved scientifically; A2 robustness closed; remaining fixed robustness and orientation stress pending')
    print('NEXT4 BOTTLENECK REVIEW 008 COMPLETE',flush=True)

if __name__=='__main__':main()
