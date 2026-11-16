"""Fixed top-two gap and search-pool context, reusing original paired draws."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next.bootstrap import paired_summary


def main():
    paths=list((BASE/'experiments').glob('*/full/RESULTS.json'))
    full=[read_json(p) for p in paths];assert len(full)==14 and all(r['status']=='COMPLETE' for r in full)
    ordered=sorted(full,key=lambda r:(-r['mean'],r['candidate']));first,second=ordered[:2]
    out=BASE/'winner_gap';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'source_sha256':digest(Path(__file__)),
        'fixed_candidates_after_full_menu':[r['candidate'] for r in ordered],
        'top_two':[first['candidate'],second['candidate']],'new_draws':0,'new_fits':0},immutable=True)
    arrays=[];references=[];proof={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    for r in (first,second):
        directory=BASE/'bootstrap'/r['candidate'];receipt=read_json(directory/'RESULTS.json')
        path=directory/'AGGREGATE_DRAWS.npz';assert digest(path)==receipt['files_sha256'][path.name]
        start=read_json(directory/'START.json');assert start['seed_base']==20260908 and start['draws']==2000
        with np.load(path,allow_pickle=False) as data:
            arrays.append(data['candidate_fold_scores'].copy());references.append(data['reference_fold_scores'].copy())
        proof[path.relative_to(ROOT).as_posix()]=digest(path)
    np.testing.assert_array_equal(references[0],references[1])
    gap=paired_summary(arrays[1],arrays[0]);assert abs(gap['estimate_mean_delta']-(first['mean']-second['mean']))<1e-14
    joint_path=BASE/'joint_uncertainty/CONDITIONAL_JOINT_DIAGNOSTIC.csv';joint=pd.read_csv(joint_path).set_index('candidate')
    proof[joint_path.relative_to(ROOT).as_posix()]=digest(joint_path)
    summaries=[]
    for rank,r in enumerate(ordered,1):
        cid=r['candidate'];screen_path=BASE/'experiments'/cid/'screen/RESULTS.json';screen=read_json(screen_path)
        proof[screen_path.relative_to(ROOT).as_posix()]=digest(screen_path)
        summaries.append({'candidate':cid,'family':r['family'],'full_rank':rank,'screen_mean_delta':screen['mean_delta'],
            'full_mean_delta':r['mean_delta'],'full_minus_screen_gain':r['mean_delta']-screen['mean_delta'],
            'screen_full_gain_sign_agreement':(r['mean_delta']>0)==(screen['mean_delta']>0),
            'bootstrap_best_frequency':float(joint.loc[cid,'bootstrap_best_frequency']),
            'single_step_joint_lower':float(joint.loc[cid,'one_sided_95_joint_lower'])})
    pd.DataFrame(summaries).to_csv(out/'SCREEN_FULL_SEARCH_CONTEXT.csv',index=False)
    registrations=[read_json(p) for p in (BASE/'registrations').glob('*.json')]
    families=[]
    for family in 'ABCDEFGH':
        families.append({'family':family,'registered_scored_recipes':sum(r['family']==family for r in registrations),
            'completed_full_recipes':sum(r['family']==family for r in full),
            'conditional_menu_only':family=='F'})
    pd.DataFrame(families).to_csv(out/'FAMILY_MULTIPLICITY.csv',index=False)
    robustness=read_json(BASE/'robustness'/first['candidate']/'RESULTS.json')
    alt=[r['mean_delta'] for r in robustness['summaries'] if r['split'].startswith('ALT_')]
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'primary_winner':first['candidate'],
        'runner_up':second['candidate'],'winner_mean':first['mean'],'runner_up_mean':second['mean'],
        'observed_winner_runner_up_gap':first['mean']-second['mean'],'conditional_fixed_top_two_gap':gap,
        'winner_bootstrap_best_frequency':float(joint.loc[first['candidate'],'bootstrap_best_frequency']),
        'winner_ALT_mean_deltas':alt,'winner_ALT_delta_min':min(alt),'winner_ALT_delta_max':max(alt),
        'winner_ALT_delta_sample_sd':float(np.std(alt,ddof=1)),
        'new_fits':0,'new_draws':0,'seal_rows':0,'gate_override':False,
        'interpretation':'The top two identities are selected from the completed original full menu and held fixed in this paired gap calculation. It is conditional descriptive uncertainty, not winner-selection correction, independent test confirmation or a posterior probability. Screen and full stages change training size and validation coverage, so their gain difference is not an unbiased estimate of selection bias. Family F has frozen conditional policies but no new scored recipe.',
        'source_evidence_sha256':proof,'files_sha256':{p.name:digest(p) for p in out.glob('*.csv')}},immutable=True)
    progress('Winner-curse context assessed','Original top-two paired gap, family multiplicity, screen/full changes and fixed ALT dispersion; no gate override')
    print('NEXT4 WINNER GAP COMPLETE',first['candidate'],second['candidate'],flush=True)


if __name__=='__main__':main()
