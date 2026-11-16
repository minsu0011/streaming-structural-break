"""Keep scientific gates fixed while recovering a native diagnostic failure."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,progress
from src.next4.bottlenecks import record_board


def main():
    assert len(list((BASE/'bottlenecks/history').glob('*.json')))==6
    items=read_json(BASE/'bottlenecks/BOARD.json')['bottlenecks']
    cold=read_json(BASE/'cold_api_readonly_v2/RESULTS.json');assert cold['status']=='PASS'
    probe=read_json(BASE/'maintenance/SYN_NATIVE_FAILURE_001/probe/normal.json');assert probe['status']=='PASS'
    for item in items:
        if item['id']=='B2':
            item['next_test']='Complete unchanged B2 synthetic continuation, then the five originally SCALE-admitted robustness routes. B2 improved Gaussian/ARCH decrease cases V3/V4, but the fixed heavy-tail decrease CB_N3_070 point score is 0.398904 versus matched BL80 0.405138 in the crash-reproduction probe; final suite interval remains pending. No model or gate change.'
            item['evidence']+=['artifacts/next4/maintenance/SYN_NATIVE_FAILURE_001/probe/normal.json',
                'artifacts/next4/diagnostic_policy/ALARM_RANK_GAP_POLICY.json']
        elif item['id']=='B5':
            item['next_test']='Finish A1 and explicitly recovered B2 synthetic suites, then the remaining fixed suites and existing memory/attribution jobs. The timed-stack native failure is preserved; future timed dumps are disabled without changing any prediction code. Long-null score drift, repeated alarm opportunities and outlier recovery remain unresolved scientific questions.'
            item['evidence']+=['artifacts/next4/maintenance/SYN_NATIVE_FAILURE_001/RECOVERY_POLICY.json',
                'artifacts/next4/cold_api_readonly_v2/RESULTS.json']
    record_board(items,'B2 cold read-only import repair passed two fresh processes with exact outputs and zero write attempts. Separately, first B2 synthetic job failed at its 1500s timed stack dump with python313.dll access violation; a short same-array normal/timed probe did not reproduce the crash. Preserve uncertainty about native cause and every original completed file; one unchanged continuation is registered. Scientific bottlenecks remain supported decrease transfer and actual fitted-model memory, with no new candidate menu.')
    progress('Bottleneck review 007','Cold API issue resolved; native synthetic failure has bounded same-recipe continuation; heavy-tail transfer and long-null memory remain active')
    print('NEXT4 BOTTLENECK REVIEW 007 COMPLETE',flush=True)


if __name__=='__main__':main()
