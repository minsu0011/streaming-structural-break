"""Close finite research priorities after the actual scientific role freeze."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,ROOT,read_json,progress
from src.next4.bottlenecks import record_board


def main():
    selection=read_json(ROOT/'DEV_CANDIDATES_NEXT4.json');assert selection['status']=='FROZEN_NEXT4_DEV_SELECTION'
    assert read_json(BASE/'SESSION.json')['status']=='ACTIVE_RESEARCH'
    items=read_json(BASE/'bottlenecks/BOARD.json')['bottlenecks'];role=selection['roles'].get('NEXT4_ROBUSTNESS')
    for item in items:
        if item['id']=='B2':
            item['status']='CLOSED_SCIENTIFIC_DECISION_AUDIT_REQUIRED'
            item['next_test']=('Limited SCALE role '+role['experiment_id']+' passed the fixed research requirements. ' if role else 'No new robustness role passed all research requirements. ')+ 'No further model menu in this session. Complete the separate final audit, with raw-feature training reproduction only for an actually selected new role.'
            item['evidence']+=['DEV_CANDIDATES_NEXT4.json','docs/NEXT4_FINAL_TRANSFER_REVIEW.md']
        elif item['id']=='B5':
            item['status']='DIAGNOSED_LONG_NULL_LIMITATION_RETAINED'
            item['next_test']='All fixed 1024/4096/16384 suites and paired alarm diagnostics are complete. Preserve horizon-dependent alarm inflation and actual-file burst losses. No threshold repair or anytime claim; final audit and handoff remain.'
        elif item['id']=='B4':
            item['next_test']='DEV saturation remains HIGH. All 14 original primary intervals include zero. Keep frozen BL80 as the primary performance champion, record the limited role separately, and do not open seals or use reduced/public/cloud testing.'
    assert not any(item['status']=='ACTIVE' for item in items)
    record_board(items,'Finite scientific research and required actual-artifact validation complete; '+selection['session_outcome']+'. No active model-search priority remains. The actual separately timed final audit and verified delivery are still required; no retrospective audit credit or blanket robustness claim.')
    progress('Research bottleneck board closed','Finite menu and actual-file research checks complete; separate final audit and reproducibility remain')
    print('NEXT4 RESEARCH BOARD CLOSED; final audit still required',flush=True)


if __name__=='__main__':main()
