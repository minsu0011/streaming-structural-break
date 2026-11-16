"""Reject failed necessary promotion gates; never declare a complete promotion."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress


def main():
    policy=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json');checks=[];decisions=[]
    for directory in sorted((BASE/'experiments').glob('*')):
        if not (directory/'RESULTS.json').exists():continue
        candidate=directory.name;current=read_json(directory/'RESULTS.json');scope=current['scope']
        necessary={'PERFORMANCE':[],'ROBUSTNESS':[]}
        def add(role,key,value,threshold,comparison):
            state='PENDING' if value is None else 'PASS' if comparison(value,threshold) else 'FAIL'
            necessary[role].append(state)
            checks.append({'candidate':candidate,'role':role,'gate':key,'observed':value,'threshold':threshold,'state':state})
        if scope!='full':
            state='SCREEN_FAIL_STOP' if current.get('screen_pass') is False else 'NOT_EVALUATED_SCREEN_STAGE'
            decisions.append({'candidate':candidate,'current_scope':scope,'performance_necessary_gates':state,
                'robustness_necessary_gates':state,'screen_pass':current.get('screen_pass'),'promotion':False})
            continue
        p=policy['performance'];r=policy['robustness']
        for role in necessary:
            add(role,'locked_full_finalist_admission',current.get('finalist_admitted',False),True,lambda a,b:a is b)
        add('PERFORMANCE','mean_delta',current['mean_delta'],p['mean_delta_min'],lambda a,b:a>=b)
        add('PERFORMANCE','median_delta',current['median_delta'],p['median_delta_min'],lambda a,b:a>=b)
        add('PERFORMANCE','worst_delta',current['worst_delta'],p['worst_fold_delta_min'],lambda a,b:a>=b)
        add('ROBUSTNESS','mean_delta',current['mean_delta'],r['mean_delta_min'],lambda a,b:a>=b)
        add('ROBUSTNESS','worst_delta',current['worst_delta'],r['worst_fold_delta_min'],lambda a,b:a>=b)
        path=BASE/'forensic'/candidate/'PAIRED_FRACTURES.csv'
        overall=scale=None
        if path.exists() and (path.parent/'RESULTS.json').exists():
            forensic=pd.read_csv(path);overall=forensic.loc[forensic.scope=='all_time'].iloc[0]
            scale=forensic.loc[forensic.scope=='mechanism_SCALE_DOMINANT'].iloc[0]
        add('PERFORMANCE','conditional_P_positive',None if overall is None else float(overall.probability_delta_positive),p['bootstrap_probability_positive_min'],lambda a,b:a>=b)
        add('PERFORMANCE','conditional_CI_lower_positive',None if overall is None else float(overall.ci_2_5),0.,lambda a,b:a>b)
        add('ROBUSTNESS','SCALE_delta',None if scale is None else float(scale.estimate_mean_delta),r['SCALE_delta_vs_frozen_blend_min'],lambda a,b:a>=b)
        add('ROBUSTNESS','SCALE_CI_lower_positive',None if scale is None else float(scale.ci_2_5),0.,lambda a,b:a>b)
        if candidate=='R1_SCORE_GAP':
            path=BASE/'counterexamples/R1_SCORE_GAP/COUNTEREXAMPLES.csv'
            frame=pd.read_csv(path) if path.exists() and (path.parent/'RESULTS.json').exists() else None
            for scenario in ('V3','V4'):
                value=None
                if frame is not None:
                    part=frame[frame.scenario==scenario].set_index('candidate');value=float(part.loc[candidate,'ts_auc']-part.loc['BL80','ts_auc'])
                add('ROBUSTNESS',scenario+'_ARCH_delta',value,r['ARCH_scale_down_delta_min_each'],lambda a,b:a>=b)
        elif candidate=='R3_DEPTH2_GATE':
            path=BASE/'counterexamples/R3_DEPTH2_GATE/arch_gate/RESULTS.json'
            rows={row['scenario']:row for row in read_json(path)['scenarios']} if path.exists() else {}
            for scenario in ('V3','V4'):add('ROBUSTNESS',scenario+'_ARCH_delta',rows.get(scenario,{}).get('delta'),r['ARCH_scale_down_delta_min_each'],lambda a,b:a>=b)
        def summary(role):
            states=necessary[role]
            return 'NECESSARY_GATE_FAIL' if 'FAIL' in states else 'PENDING_NECESSARY_EVIDENCE' if 'PENDING' in states else 'CONTINUE_FULL_VALIDATION'
        decisions.append({'candidate':candidate,'current_scope':scope,'performance_necessary_gates':summary('PERFORMANCE'),
            'robustness_necessary_gates':summary('ROBUSTNESS'),'screen_pass':True,'promotion':False,
            'remaining_if_necessary_gates_pass':'Supported mechanisms, fixed ARCH/outlier evidence, ALT/H, engineering, deterministic raw reproduction and final audit.'})
    out=BASE/'selection';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(checks).to_csv(out/'NECESSARY_GATE_CHECKS.csv',index=False)
    pd.DataFrame(decisions).to_csv(out/'NECESSARY_GATE_DECISIONS.csv',index=False)
    atomic_json(out/'NECESSARY_GATES.json',{'status':'ONGOING_NECESSARY_GATES_ONLY','created_utc':utc_now(),
        'policy_sha256':digest(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json'),'decisions':decisions,'final_promotion':False,
        'warning':'Passing these early necessary gates is not sufficient for any final role; omitted final gates remain mandatory.',
        'files_sha256':{name:digest(out/name) for name in ('NECESSARY_GATE_CHECKS.csv','NECESSARY_GATE_DECISIONS.csv')}})
    progress('Necessary promotion gates assessed',f'{len(decisions)} recipes checked; no final promotion declared')
    print('NEXT3 NECESSARY GATES',len(decisions),'NOT FINAL SELECTION',flush=True)


if __name__=='__main__':main()
