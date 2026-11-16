"""Expose fixed scientific rejections and pending robustness validation separately."""
import argparse
from pathlib import Path
import re
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+',args.tag):raise ValueError('Safe immutable snapshot tag required')
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    rows=[];evidence={};supported_min=policy['robustness']['SCALE_break_support_min']
    def read(path):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        return read_json(path)
    for path in sorted((BASE/'registrations').glob('*.json')):
        reg=read(path);cid=reg['experiment_id'];full_path=BASE/'experiments'/cid/'full/RESULTS.json'
        row={'candidate':cid,'family':reg['family'],'selected_for_deployment':False,
            'performance_failed_gates':[],'robustness_failed_gates':[],
            'robustness_pending_evidence':[],'robustness_fixed_real_and_synthetic_admission':False}
        if not full_path.exists():
            sr=read(BASE/'experiments'/cid/'screen/RESULTS.json');assert sr['screen_pass'] is False
            row['performance_failed_gates']=['LOCKED_SCREEN'];row['robustness_failed_gates']=['LOCKED_SCREEN']
            row['disposition']='SCREEN_FAIL_STOP';rows.append(row);continue
        r=read(full_path);b=read(BASE/'bootstrap'/cid/'RESULTS.json');p=b['paired_all_time'];g=policy['performance']
        checks={'mean_delta':r['mean_delta']>=g['mean_delta_min'],
            'median_delta':r['median_delta']>=g['median_delta_min'],
            'worst_delta':r['worst_delta']>=g['worst_delta_min'],
            'paired_probability':p['probability_delta_positive']>=g['bootstrap_probability_min'],
            'paired_CI_lower_strict_positive':p['ci_2_5']>0}
        row['performance_failed_gates']=[k for k,v in checks.items() if not v]
        assert all(checks.values())==b['performance_primary_and_bootstrap_admission']
        row.update(primary_mean=r['mean'],primary_gain=r['mean_delta'],primary_CI_lower=p['ci_2_5'],
            primary_CI_upper=p['ci_97_5'],primary_P_positive=p['probability_delta_positive'])
        if r['mean_delta']<policy['robustness']['overall_mean_delta_min']:
            row['robustness_failed_gates'].append('overall_mean_delta')
        fp=BASE/'forensic'/cid/'RESULTS.json'
        if fp.exists():
            f=read(fp);s=f['paired_scale'];rg=policy['robustness']
            row.update(SCALE_gain=s['estimate_mean_delta'],SCALE_CI_lower=s['ci_2_5'],
                SCALE_break_series=s['break_series'],SCALE_support=s['break_series']>=supported_min)
            for key,ok in {'supported_SCALE':s['break_series']>=supported_min,
                'SCALE_delta':s['estimate_mean_delta']>=rg['SCALE_delta_min'],
                'SCALE_CI_lower_strict_positive':s['ci_2_5']>0}.items():
                if not ok:row['robustness_failed_gates'].append(key)
            fractures_path=fp.parent/'PAIRED_FRACTURES.csv';evidence[fractures_path.relative_to(ROOT).as_posix()]=digest(fractures_path)
            frame=pd.read_csv(fractures_path)
            for scope in ('mechanism_DEPENDENCE_DOMINANT','mechanism_MIXED','mechanism_WEAK_OR_AMBIGUOUS','mechanism_MEAN_DOMINANT','mechanism_TAIL_DOMINANT'):
                mechanism=frame[frame.scope==scope].iloc[0]
                if mechanism.break_series>=supported_min and mechanism.estimate_mean_delta<rg['minimum_other_supported_mechanism_delta']:
                    row['robustness_failed_gates'].append(scope+'_damage')
            increase=frame[frame.scope=='mechanism_observed_scale_increase'].iloc[0]
            row['observed_scale_increase_support']=int(increase.break_series)
            row['variance_increase_gate_assessable']=bool(increase.break_series>=supported_min)
            if increase.break_series>=supported_min and increase.estimate_mean_delta<rg['minimum_supported_variance_increase_delta']:
                row['robustness_failed_gates'].append('supported_variance_increase_damage')
        elif not row['robustness_failed_gates']:
            row['robustness_pending_evidence'].append('original_supported_SCALE_and_mechanisms')
        scope='all_dev' if cid.startswith('B2_') else 'full_fold0'
        sp=BASE/'synthetic/candidates'/(cid+'_'+scope)/'RESULTS.json'
        if sp.exists():
            syn=read(sp);assert syn['status']=='PASS';row['synthetic_training_scope']=syn['training_scope']
            for name in ('ARCH_decrease_each_delta_positive','heavy_tail_decrease_each_delta_positive'):
                row[name]=syn[name]
                if not syn[name]:row['robustness_failed_gates'].append(name)
        elif not row['robustness_failed_gates']:
            row['robustness_pending_evidence'].append('fixed_matched_synthetic_suite')
        rp=BASE/'robustness'/cid/'RESULTS.json'
        if rp.exists():
            alt=read(rp);assert alt['status']=='PASS'
            row['ALT_positive_count']=alt['ALT_positive_count'];row['H_minimum_matched_delta']=alt['H_minimum_matched_delta']
            row['ALT_H_scope']='Frozen reused-DEV grouped partitions; not independent new data'
        elif not row['robustness_failed_gates']:
            row['robustness_pending_evidence'].append('fixed_ALT_H_replication')
        science_admitted=not row['robustness_failed_gates'] and not row['robustness_pending_evidence']
        row['robustness_fixed_real_and_synthetic_admission']=science_admitted
        if science_admitted:
            row['robustness_pending_evidence']+=['review_existing_ALT_H_transfer','all_DEV_finalist_model',
                '300_prefix_and_direct_API','determinism_save_load','direct_runtime','fresh_raw_reproduction_if_promoted']
        row['disposition']='NO_SCIENTIFIC_PROMOTION' if row['robustness_failed_gates'] and row['performance_failed_gates'] else 'ROBUSTNESS_VALIDATION_PENDING'
        rows.append(row)
    assert len(rows)==16 and all(r['performance_failed_gates'] for r in rows)
    admitted=sorted([r for r in rows if r['robustness_fixed_real_and_synthetic_admission']],
        key=lambda r:(-r['SCALE_CI_lower'],-r['primary_gain'],r['candidate']))
    out=BASE/'gate_reviews'/args.tag;out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(out/'GATE_EVIDENCE.csv',index=False)
    atomic_json(out/'RESULTS.json',{'status':'SCIENTIFIC_GATE_REVIEW_NOT_FINAL_SELECTION','created_utc':utc_now(),
        'policy_sha256':digest(policy_path),'performance_admitted':[],
        'generation_admitted':[],'robustness_deployment_validation_order':[r['candidate'] for r in admitted],
        'unresolved_robustness_candidates':[r['candidate'] for r in rows if not r['robustness_failed_gates']],
        'final_role_selected':False,'new_fits':0,'new_scores_or_bootstrap_draws':0,'rows':rows,
        'scope':'Existing fixed policy only. Missing diagnostics remain pending; unsupported real variance-increase support is not presented as validation. ALT/H values are surfaced for review, without inventing an additional post-hoc numerical robustness gate.',
        'source_sha256':digest(Path(__file__)),'evidence_sha256':evidence,
        'files_sha256':{'GATE_EVIDENCE.csv':digest(out/'GATE_EVIDENCE.csv')}},immutable=True)
    progress('Scientific gate evidence reconciled','Original performance/generation paths rejected; supported robustness evidence and pending deployment validation remain separate')
    print('NEXT4 GATE REVIEW',len(rows),'identities;',len(admitted),'robustness deployment-validation nominees',flush=True)


if __name__=='__main__':main()
