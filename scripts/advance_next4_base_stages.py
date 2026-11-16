"""Advance completed, registered base candidates through prelocked gates only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json
from src.next4.scheduler import register_job


def main():
    for menu_path in sorted((BASE/'families').glob('*_MENU_LOCK.json')):
        menu=read_json(menu_path);family=menu['family']
        if family not in ('A','B','C','D','E','G','H'):continue
        for index,candidate in enumerate(menu['candidate_ids']):
            screen_id=f'{family}{index+1:03d}_SCREEN';full_id=f'{family}{index+101:03d}_FULL';bootstrap_id=f'{family}{index+201:03d}_BOOTSTRAP'
            screen=BASE/'experiments'/candidate/'screen/RESULTS.json'
            if not screen.exists() or not read_json(screen)['screen_pass']:continue
            if not (BASE/'queue'/(full_id+'.json')).exists():
                runner='scripts/evaluate_next4_pairwise.py' if family=='E' else 'scripts/evaluate_next4_replacement.py' if family=='B' else 'scripts/evaluate_next4_base.py'
                runner=menu.get('evaluation_runners',{}).get(candidate,runner)
                priority=menu.get('dispatch_priority',50 if family=='E' else 100)
                register_job({'id':full_id,'kind':'research','candidate_id':candidate,'family':family,
                    'depends_on_success':[screen_id],'argv':[runner,'--candidate',candidate,'--scope','full'],
                    'description':'Frozen successful-screen candidate; locked original 5-fold full CV','retry_safe':False,'priority':priority})
                print('QUEUED',full_id,flush=True)
            if not (BASE/'queue'/(bootstrap_id+'.json')).exists():
                register_job({'id':bootstrap_id,'kind':'validation','candidate_id':candidate,'family':family,
                    'depends_on_success':[full_id],'argv':['scripts/bootstrap_next4_candidate.py','--candidate',candidate],
                    'description':'All-time 2000 paired whole-series draws; preregistered Monte Carlo seed and fixed OOF','retry_safe':False,'priority':55 if family in ('E','G') else 100})
                print('QUEUED',bootstrap_id,flush=True)


if __name__=='__main__':main()
