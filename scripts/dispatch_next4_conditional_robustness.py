"""Apply the already frozen conditional routing promptly after each suite."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,digest
from src.next4.scheduler import register_job
from scripts.advance_next4_robustness_from_synthetic import main as advance


def main():
    advance()
    policy=read_json(BASE/'diagnostic_policy/CONDITIONAL_ROBUSTNESS_VALIDATION.json')
    for candidate in policy['candidates']:
        dependency='RB_'+candidate;identifier='DX_'+candidate+'_SCALE_DISPATCH'
        if not (BASE/'queue'/(dependency+'.json')).exists() or (BASE/'queue'/(identifier+'.json')).exists():continue
        register_job({'id':identifier,'kind':'maintenance','priority':116,'depends_on_success':[dependency],
            'argv':['scripts/register_next4_alt_scale_diagnostics.py'],
            'description':'Queue only the previously fixed SCALE views after an admitted unchanged ALT/H replication finishes',
            'retry_safe':False,'source_sha256':digest(ROOT/'scripts/register_next4_alt_scale_diagnostics.py')})
        print('NEXT4 POST-ALT DISPATCH QUEUED',candidate,flush=True)


if __name__=='__main__':main()
