"""Check prospective real-DEV experiments and frozen model sources without refits."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import importlib
from pathlib import Path
import re
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress

REQUIRED='experiment_id parent family hypothesis causal_mechanism exact_changes parameters candidate_menu_id expected_gain risk screen_gate full_gate stop_rule created_before_score'.split()


def stamp(record):return datetime.fromisoformat(record['created_utc'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+',args.tag):raise ValueError('Safe immutable audit tag required')
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    families=read_json(ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json')['families']
    budget=read_json(ROOT/'configs/next4/CANDIDATE_BUDGET.json')
    rows=[];evidence={};source_files={};counts=Counter();reference=None
    def read(path):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        return read_json(path)
    for path in sorted((BASE/'registrations').glob('*.json')):
        reg=read(path);cid=reg['experiment_id'];family=reg['family']
        assert set(REQUIRED)<=set(reg) and reg['created_before_score'] is True
        assert cid in families[family]['candidate_ids'] and reg['policy_sha256']==digest(policy_path)
        assert reg['full_gate']==policy['performance']
        assert reg['screen_gate']=={k:policy['screen'][k] for k in ('mean_delta_min','each_fold_delta_min')}
        if 'robustness_gate' in reg:assert reg['robustness_gate']==policy['robustness']
        menu_path=BASE/'families'/(family+'_MENU_LOCK.json');menu=read(menu_path)
        assert digest(menu_path)==reg['family_menu_sha256'] and stamp(menu)<=stamp(reg)
        assert stamp(policy)<=stamp(menu)
        for section in ('execution_source_sha256','nested_source_sha256'):
            for name,sha in reg.get(section,{}).items():
                assert digest(ROOT/name)==sha,(cid,name)
                source_files[name]=sha
        if 'feature_module' in reg:
            module=importlib.import_module(reg['feature_module'])
            assert module.implementation_hash(reg['feature_settings'])==reg['feature_sha256']
            assert list(module.feature_names(reg['feature_settings']))==reg['feature_names']
        source_module=('src.next4.pairwise' if family=='E' else
            'src.next4.context_pairwise' if cid.startswith('G2_') else
            'src.next4.state_readout' if family=='H' else 'src.next4.model')
        assert importlib.import_module(source_module).source_hash()==reg['model_source_sha256'],cid
        parent=BASE/'experiments'/cid;screen=read(parent/'screen/RESULTS.json')
        full_path=parent/'full/RESULTS.json';stages=[('screen',screen)]
        if full_path.exists():
            assert screen['screen_pass'] is True
            stages.append(('full',read(full_path)));counts['full']+=1
        else:
            assert screen['screen_pass'] is False and not (parent/'full/START.json').exists()
            counts['screen_failed']+=1
        first_fit=None
        for scope,result in stages:
            start=read(parent/scope/'START.json')
            assert result['status']=='COMPLETE' and result['registration_sha256']==digest(path)==start['registration_sha256']
            assert stamp(reg)<=stamp(start)<=stamp(result)
            if scope=='full':assert stamp(screen)<=stamp(start)
            expected=policy['screen']['folds'] if scope=='screen' else policy['fixed_primary_folds']
            assert [r['fold'] for r in result['fold_records']]==expected
            assert len(result['fold_records'])==result['new_fits'] and result['seal_rows']==0
            if scope=='full':
                if reference is None:reference=result['reference_fold_scores']
                np.testing.assert_array_equal(result['reference_fold_scores'],reference)
                paired=read(BASE/'bootstrap'/cid/'RESULTS.json')
                assert paired['source_result_sha256']==digest(full_path)
                assert paired['policy_sha256']==digest(policy_path)
                assert paired['draws']==policy['bootstrap']['draws'] and paired['folds']==5
                assert stamp(result)<=stamp(paired)
            for fold in expected:
                fit_start=read(parent/scope/f'fold_{fold}_FIT_START.json')
                fitted=read(parent/scope/f'fold_{fold}.json')
                assert fit_start['registration_sha256']==digest(path)
                assert stamp(start)<=stamp(fit_start)<=stamp(fitted)<=stamp(result)
                assert fitted['new_fits']==1 and fitted.get('frozen_base_new_fits',0)==0
                first_fit=stamp(fit_start) if first_fit is None else min(first_fit,stamp(fit_start))
                counts['real_selection_fits']+=1
        counts['screen']+=1;counts['family_'+family]+=1
        rows.append({'candidate':cid,'family':family,'menu_utc':menu['created_utc'],
            'registration_utc':reg['created_utc'],'first_real_DEV_fit_utc':first_fit.isoformat(),
            'registered_seconds_before_first_real_fit':(first_fit-stamp(reg)).total_seconds(),
            'screen_complete_utc':screen['created_utc'],'screen_pass':screen['screen_pass'],
            'full_complete_utc':stages[-1][1]['created_utc'] if full_path.exists() else '',
            'real_selection_fits':sum(r['new_fits'] for _,r in stages),
            'all_14_preregistration_fields_present':True,'feature_model_execution_sources_unchanged':True,
            'full_after_screen_pass_only':True,'primary_partition_unchanged':True})
    assert counts['screen']==16 and counts['full']==14 and counts['screen_failed']==2
    assert counts['real_selection_fits']==102 and counts['full']<=budget['new_full_candidates_max']
    assert len(families)==8
    out=BASE/'registration_audit'/args.tag;out.mkdir(parents=True,exist_ok=False)
    with (out/'CHRONOLOGY.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'counts':dict(counts),
        'declared_hypothesis_families':8,'families_with_new_real_DEV_scores':7,
        'conditional_family_F_no_new_score':True,'registered_before_real_DEV_fit':True,
        'all_execution_feature_model_hashes_match':True,'screen_failure_override_count':0,
        'real_DEV_refits_in_this_check':0,'new_quality_scores_or_bootstrap_draws':0,
        'scope':'Actual real-DEV chronology. Earlier synthetic/unit probes are not claimed to have followed each candidate registration; their preserved receipts are separate.',
        'protocol_order_note':'Forensics preceded ALT/H in this run, and four optional ALT/H routes were prospectively conditioned on synthetic gates; see NEXT4_PROTOCOL_SCOPE_NOTES.md.',
        'source_sha256':digest(Path(__file__)),'verified_execution_source_sha256':source_files,
        'evidence_sha256':evidence,'policy_sha256':digest(policy_path),
        'files_sha256':{'CHRONOLOGY.csv':digest(out/'CHRONOLOGY.csv')}},immutable=True)
    progress('Prospective experiment chronology verified','16 registered real-DEV screens, 14 admitted full evaluations, unchanged feature/model source hashes and no failed-screen override')
    print('NEXT4 REGISTRATION CHRONOLOGY PASS',len(rows),'candidates;',len(source_files),'execution files',flush=True)


if __name__=='__main__':main()
