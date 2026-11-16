"""Audit prospective conditional validation and actual-artifact nominations."""
import argparse
from datetime import datetime
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress


def time_of(record):return datetime.fromisoformat(record['created_utc'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9_]+',args.tag)
    session=read_json(BASE/'SESSION.json');assert session.get('audit_started_utc')
    evidence={}
    def read(path):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path);return read_json(path)
    policy_path=BASE/'diagnostic_policy/CONDITIONAL_ROBUSTNESS_VALIDATION.json';policy=read(policy_path)
    for name,sha in policy['execution_source_sha256'].items():assert digest(ROOT/name)==sha
    conditional=[]
    for cid in policy['candidates']:
        start=read(BASE/'synthetic/candidates'/(cid+'_full_fold0')/'START.json')
        syn=read(BASE/'synthetic/candidates'/(cid+'_full_fold0')/'RESULTS.json')
        decision_path=BASE/'robustness_screen'/(cid+'.json');decision=read(decision_path)
        assert time_of(policy)<time_of(start)<=time_of(syn)<=time_of(decision)
        assert decision['policy_sha256']==digest(policy_path) and decision['synthetic_result_sha256']==digest(BASE/'synthetic/candidates'/(cid+'_full_fold0')/'RESULTS.json')
        passed=all(syn['diagnostic_deltas'][s]>0 for s in ('V3','V4','CB_N3_070','CB_N3_050'))
        assert passed==decision['conditional_ALT_H_admission']
        starts=list((BASE/'robustness'/cid).rglob('*FIT_START.json'))
        if passed:
            nompath=BASE/'robustness_nominations'/(cid+'.json');nom=read(nompath)
            assert nom['decision_sha256']==digest(decision_path) and time_of(decision)<=time_of(nom)
            assert len(starts)==20
            for p in starts:assert time_of(nom)<=time_of(read(p))
        else:assert not starts and not (BASE/'robustness_nominations'/(cid+'.json')).exists()
        conditional.append({'candidate':cid,'fixed_four_case_condition':passed,'actual_ALT_H_fit_starts':len(starts),
            'policy_precedes_any_synthetic_predictions':True,'failed_gate_not_overridden':True})
    deployment=[]
    for path in sorted((BASE/'finalist_nominations').glob('*.json')):
        nom=read(path);cid=nom['candidate']
        assert nom['new_candidate_identity'] is False and nom['new_frozen_base_fits']==0
        for name,sha in nom['execution_sources_sha256'].items():assert digest(ROOT/name)==sha
        directory=BASE/'refits'/cid/'nominated_all_dev_v1'
        fit_start=read(directory/'FIT_START.json');fit=read(directory/'RESULTS.json')
        assert time_of(nom)<=time_of(fit_start)<=time_of(fit) and fit['nomination_sha256']==digest(path)
        syn_nompath=BASE/'synthetic_nominations'/(cid+'_all_dev.json');syn_nom=read(syn_nompath)
        syn_start=read(BASE/'synthetic/candidates'/(cid+'_all_dev')/'START.json')
        syn=read(BASE/'synthetic/candidates'/(cid+'_all_dev')/'RESULTS.json')
        assert time_of(fit)<=time_of(syn_nom)<time_of(syn_start)<=time_of(syn)
        assert syn_nom['finalist_nomination_sha256']==digest(path) and syn_nom['model_sha256']==digest(directory/'model.joblib')
        assert syn_start['nomination_sha256']==digest(syn_nompath)
        mp=read(BASE/'diagnostic_policy'/('MEMORY_FINALIST_'+cid+'_all_dev.json'))
        assert time_of(mp)<time_of(syn_start) and mp['nominations'][cid+'_all_dev']==digest(syn_nompath)
        for name,sha in mp['sources_sha256'].items():assert digest(ROOT/name)==sha
        assert read(BASE/'memory_diagnostics'/(cid+'_all_dev')/'RESULTS.json')['status']=='PASS'
        for p in (BASE/'finalist_validation'/cid/('API_'+cid)/'RESULTS.json',BASE/'finalist_validation'/cid/'COLD/RESULTS.json',BASE/'latency'/(cid+'_DIRECT')/'RESULTS.json'):
            r=read(p);assert r['status']=='PASS' and time_of(nom)<time_of(r)
        deployment.append({'candidate':cid,'execution_sources_unchanged':len(nom['execution_sources_sha256']),
            'actual_artifact_model_sha256':syn_nom['model_sha256'],'fit_and_confirmation_prospectively_nominated':True,
            'fixed_ARCH_confirmation':syn['ARCH_decrease_each_delta_positive'],
            'fixed_heavy_tail_confirmation':syn['heavy_tail_decrease_each_delta_positive']})
    out=BASE/'final_audit'/args.tag/'nomination_chronology';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'conditional_routes':conditional,'actual_artifact_routes':deployment,'new_fits':0,'new_scores':0,
        'new_candidates_or_gates':False,'original_execution_hashes_unchanged':True,
        'conditional_and_actual_artifact_policies_precede_their_predictions':True,
        'evidence_sha256':evidence,'source_sha256':digest(Path(__file__))},immutable=True)
    progress('Final audit nomination chronology','Four conditional paths and actual artifact nomination timestamps, unchanged execution hashes and no failed-gate refits verified',phase='FINAL_AUDIT',counts_as_research=False)
    print('NEXT4 FINAL NOMINATION CHRONOLOGY PASS',flush=True)


if __name__=='__main__':main()
