from pathlib import Path
import argparse
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.scheduler import register_job


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args();lock=BASE/'analysis_figures/POLICY_LOCK.json'
    if args.register:
        candidates=sorted(p.parent.parent.name for p in (BASE/'experiments').glob('*/full/RESULTS.json'))
        atomic_json(lock,{'created_utc':utc_now(),'renderer_sha256':digest(ROOT/'scripts/render_next5_primary_figures.py'),
            'candidate_count':len(candidates),'raw_observation_access':False,'interim_research_not_final_handoff':True,
            'environment':'Existing .figure-venv, no package installation or global environment changes'},immutable=True)
        register_job({'id':'FIGURES_PRIMARY_INTERPRETATION','kind':'validation','priority':79,
            'depends_on_success':['UNC001_MATCHED_CATALOG']+['FORENSIC_'+cid for cid in candidates],
            'argv':['scripts/run_next5_primary_figures.py'],'description':'Standalone primary uncertainty, time/age and supported-mechanism figures from fixed aggregate tables only','retry_safe':False})
        return
    policy=read_json(lock);assert digest(ROOT/'scripts/render_next5_primary_figures.py')==policy['renderer_sha256']
    progress('Render frozen ResearchPool diagnostic figures','Aggregate tables only; primary candidate intervals, time/age and supported mechanisms')
    subprocess.run([str(ROOT/'.figure-venv/Scripts/python.exe'),'-B','-u',str(ROOT/'scripts/render_next5_primary_figures.py')],cwd=ROOT,check=True)
    progress('Research diagnostic figures complete','Three standalone figures and one PDF, with source hashes and conditional uncertainty/support caveats')


if __name__=='__main__':main()
