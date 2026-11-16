"""Freeze at most one additional validated robustness role before final audit."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gate-tag',required=True);parser.add_argument('--transfer-review',required=True);args=parser.parse_args()
    review_path=BASE/'gate_reviews'/args.gate_tag/'RESULTS.json';review=read_json(review_path)
    assert not review['performance_admitted'] and not review['generation_admitted']
    for name,sha in review['evidence_sha256'].items():assert digest(ROOT/name)==sha
    for path in (BASE/'synthetic_nominations').glob('*.json'):
        nomination=read_json(path);r=read_json(BASE/'synthetic/candidates'/nomination['id']/'RESULTS.json');assert r['status']=='PASS'
    assert read_json(BASE/'sign_reflection/RESULTS.json')['status']=='PASS'
    note=(ROOT/args.transfer_review).resolve();assert note.is_relative_to(ROOT) and note.is_file()
    refs=read_json(BASE/'preflight/FROZEN_REFERENCES.json');base=refs['roles']['FROZEN_PERFORMANCE_BL80'];evidence={
        review_path.relative_to(ROOT).as_posix():digest(review_path),note.relative_to(ROOT).as_posix():digest(note),
        'artifacts/next4/preflight/FROZEN_REFERENCES.json':digest(BASE/'preflight/FROZEN_REFERENCES.json')}
    roles={'FROZEN_PERFORMANCE':{'experiment_id':'BL2_C205_DEPENDENCE_80','entry':'next2_shared_performance_main.py',
        'model_directory':base['model_directory'],'model_file':base['model_file'],'model_sha256':base['actual_model_sha256'],
        'model_bytes':base['model_bytes'],'original_primary':{k:base['full'][k] for k in ('mean','median','worst_fold','fold_scores')},
        'delta_vs_session_start':0.,'retained_frozen_role':True,'new_fit':False}}
    pending=[]
    for cid in review['unresolved_robustness_candidates']:
        row=next(r for r in review['rows'] if r['candidate']==cid)
        if not row['robustness_fixed_real_and_synthetic_admission']:pending.append(cid)
    assert not pending,('Unfinished scientific robustness routes',pending)
    nominees=review['robustness_deployment_validation_order'];rejections=[]
    for cid in nominees:
        directory=BASE/'refits'/cid/'nominated_all_dev_v1';api=BASE/'finalist_validation'/cid/('API_'+cid)/'RESULTS.json'
        cold=BASE/'finalist_validation'/cid/'COLD/RESULTS.json';runtime=BASE/'latency'/(cid+'_DIRECT')/'RESULTS.json'
        if not all(p.exists() for p in (directory/'RESULTS.json',api,cold,runtime)):
            raise RuntimeError('Higher-ranked robustness nominee lacks required deployment validation: '+cid)
        results=[read_json(p) for p in (directory/'RESULTS.json',api,cold,runtime)]
        assert all(r['status']=='PASS' for r in results)
        actual_syn_path=BASE/'synthetic/candidates'/(cid+'_all_dev')/'RESULTS.json'
        actual_syn=read_json(actual_syn_path);assert actual_syn['status']=='PASS'
        evidence[actual_syn_path.relative_to(ROOT).as_posix()]=digest(actual_syn_path)
        if not actual_syn['ARCH_decrease_each_delta_positive'] or not actual_syn['heavy_tail_decrease_each_delta_positive']:
            rejections.append({'candidate':cid,'reason':'FINAL_ARTIFACT_FIXED_SYNTHETIC_DECREASE_CONFIRMATION_FAILED'});continue
        if not results[-1]['direct_runtime_within_3x']:
            rejections.append({'candidate':cid,'reason':'ORIGINAL_DIRECT_RUNTIME_GATE_FAILED'});continue
        r=read_json(BASE/'experiments'/cid/'full/RESULTS.json');b=read_json(BASE/'bootstrap'/cid/'RESULTS.json');f=read_json(BASE/'forensic'/cid/'RESULTS.json')
        assert results[1]['prefix_cases']>=300 and results[2]['all_repeated_predictions_exact']
        roles['NEXT4_ROBUSTNESS']={'experiment_id':cid,'entry':'next4_robustness_main.py',
            'model_directory':directory.relative_to(ROOT).as_posix(),'model_file':(directory/'model.joblib').relative_to(ROOT).as_posix(),
            'model_sha256':digest(directory/'model.joblib'),'model_bytes':(directory/'model.joblib').stat().st_size,
            'original_primary':{k:r[k] for k in ('mean','median','worst_fold','fold_scores','mean_delta')},
            'paired_all_time':b['paired_all_time'],'paired_SCALE':f['paired_scale'],
            'canonical_reference_reused':'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1',
            'direct_runtime_median_ratios':results[-1]['summary_median_ratios'],
            'original_primary_performance_gate_passed':False,'fresh_raw_reproduction_required_in_separate_audit':True,
            'scope':'Supported SCALE robustness role only; not a new primary performance champion or generation.'}
        for p in (directory/'RESULTS.json',api,cold,runtime,BASE/'experiments'/cid/'full/RESULTS.json',
                  BASE/'bootstrap'/cid/'RESULTS.json',BASE/'forensic'/cid/'RESULTS.json'):
            evidence[p.relative_to(ROOT).as_posix()]=digest(p)
        break
    if nominees and 'NEXT4_ROBUSTNESS' not in roles:assert len(rejections)==len(nominees)
    best=read_json(BASE/'experiments/B2_LOG_ENERGY_SEMITAIL/full/RESULTS.json')
    result={'status':'FROZEN_NEXT4_DEV_SELECTION','created_utc':utc_now(),'roles':roles,'role_count':len(roles),
        'maximum_roles':4,'primary_champion':'BL2_C205_DEPENDENCE_80','primary_score':base['full']['mean'],
        'primary_delta_vs_session_start':0.,'new_performance_promotion':False,'new_generation':False,
        'robustness_role_nominated_for_final_audit':'NEXT4_ROBUSTNESS' in roles,
        'session_outcome':'NEXT4_ROBUSTNESS_ONLY_NO_PERFORMANCE_PROMOTION' if 'NEXT4_ROBUSTNESS' in roles else 'NEXT4_NO_PROMOTION',
        'final_audit_completion_claimed':False,'deployment_release_requires_final_audit_PASS':True,
        'best_exploratory':{'candidate':'B2_LOG_ENERGY_SEMITAIL','mean':best['mean'],'delta':best['mean_delta'],
            'not_selected_reason':'Original primary paired CI lower endpoint is negative. No ALT, runtime or synthetic override.'},
        'unfilled_roles':['NEXT4_PERFORMANCE','NEXT4_EXPERIMENTAL_GENERATION']+([] if 'NEXT4_ROBUSTNESS' in roles else ['NEXT4_ROBUSTNESS']),
        'runtime_rejections':rejections,'evidence_sha256':evidence,'source_sha256':digest(Path(__file__)),
        'all_seals_unopened':True,'new_reduced_usage':0,'new_public_submission':0,'new_cloud_submission':0}
    atomic_json(ROOT/'DEV_CANDIDATES_NEXT4.json',result,immutable=True)
    progress('Final scientific roles frozen',result['session_outcome']+'; separate actual audit still required')
    print('NEXT4 SCIENTIFIC ROLES FROZEN',result['session_outcome'],flush=True)


if __name__=='__main__':main()
