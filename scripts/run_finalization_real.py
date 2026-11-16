"""Execute the reserved final buffer after research stops; never submit anything."""
import argparse, os, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now
from src.validation.adaptive import ResearchBudget


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true',help='After explicit research closure, prepare frozen DEV artifacts early without reduced target access, final score disclosure or Desktop ZIP creation')
    args=parser.parse_args()
    budget=ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json')
    if budget.remaining_experiment_seconds()>0 and not (args.prepare_only and budget.state.get('model_research_closed')):raise RuntimeError('Before the original final buffer, only explicitly closed audit preparation is permitted')
    write_json(ROOT/'artifacts/real/CONTROL.json',{'pause_after_candidate':True,'reason':'Model research is closed; frozen DEV validation and packaging only. Original final-diagnostic deadline is preserved.'})
    deadline=time.monotonic()+600
    while read('artifacts/real/RESEARCH_STATE.json').get('active_candidate'):
        if time.monotonic()>deadline:raise RuntimeError('Research did not reach a clean checkpoint within ten minutes')
        time.sleep(5)
    path=ROOT/'artifacts/real/FINALIZATION_STATE.json'
    progress=read(str(path.relative_to(ROOT))) if path.exists() else {'status':'RUNNING','started_utc':utc_now(),'steps':{},'LOCAL_FINAL_SEAL_OPENED':False,'submitted':False}
    progress['status']='RUNNING'
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    def run(name,script,*arguments):
        if progress['steps'].get(name,{}).get('status')=='PASS':print('FINAL REUSE',name,flush=True);return
        log=ROOT/'experiments/logs/final'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(ROOT/script),*map(str,arguments)]
        progress['active_step']=name;progress['steps'][name]={'status':'RUNNING','start_utc':utc_now(),'command':command,'log':str(log.relative_to(ROOT))};write_json(path,progress)
        print('FINAL START',name,flush=True);start=time.monotonic()
        with log.open('w',encoding='utf-8') as output:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        progress['steps'][name].update(status='PASS' if result.returncode==0 else 'FAIL',exit_code=result.returncode,elapsed_seconds=time.monotonic()-start,completed_utc=utc_now())
        write_json(path,progress);print('FINAL END',name,progress['steps'][name]['status'],flush=True)
        if result.returncode:raise RuntimeError('Final stage failed: '+name)
    run('selection_plan','scripts/prepare_final_selection.py')
    plan=read('artifacts/real/FINAL_SELECTION_PLAN.json')
    if not (ROOT/'DEV_CHAMPION_CANDIDATES.json').exists():
        for index,eid in enumerate(plan['benchmark_candidates']):
            mode=plan['benchmark_modes'][eid];flags=['--fast-initialization'] if mode=='fused' else []
            run(f'selection_latency_{index}','scripts/benchmark_real_candidate.py','--candidate',eid,'--tag','final_selection','--mode',mode,
                '--points',10000000,'--series',128,'--isolated',*flags)
        closure_flags=['--audit-phase-closure'] if args.prepare_only else []
        run('freeze_candidates','scripts/prepare_final_selection.py','--freeze','--benchmark-tag','final_selection',*closure_flags)
    frozen=read('DEV_CHAMPION_CANDIDATES.json');primary=frozen['PRIMARY_DEV_CANDIDATE']
    run('final_dev_refits','scripts/train_final_artifacts.py')
    run('selected_cv_refit','scripts/reproduce_selected_cv.py','--candidate',primary,'--tag','final_selected')
    for index,(role,selected) in enumerate(frozen['candidates'].items()):
        eid=selected['experiment_id'];descriptor=read(selected['model_directory']+'/model.json')
        if role!='PRIMARY_DEV_CANDIDATE':
            run(f'backup_cv_refit_{index}','scripts/reproduce_selected_cv.py','--candidate',eid,'--tag','final_selected')
        mode='fused' if descriptor['use_fused_inference'] else 'native';flags=['--fast-initialization'] if mode=='fused' else []
        run(f'final_contract_{index}','scripts/verify_selected_contract.py','--candidate',eid,'--mode',mode,'--series',32,
            '--model-directory',ROOT/selected['model_directory'],*flags)
        if mode=='fused':
            run(f'full_selected_streaming_oof_{index}','scripts/verify_fused_real.py','--candidate',eid,'--fast-initialization')
        run(f'final_artifact_latency_{index}','scripts/benchmark_real_candidate.py','--candidate',eid,'--tag','final_deployment','--mode',mode,
            '--points',10000000 if role=='PRIMARY_DEV_CANDIDATE' else 1000000,'--series',128,'--isolated','--model-directory',ROOT/selected['model_directory'],*flags)
    run('final_feature_health','scripts/audit_fitted_features.py','--candidate',primary)
    score_arguments=[]
    for selected in frozen['candidates'].values():score_arguments.extend(['--candidate',selected['experiment_id']])
    run('full_selected_official_scorer','scripts/verify_selected_official_scores.py',*score_arguments)
    run('actual_final_full_streaming','scripts/verify_final_streaming_real.py','--primary-socket')
    if args.prepare_only:
        progress.update(status='PREPARED_AWAITING_FINAL_DIAGNOSTIC_WINDOW',active_step=None,prepared_utc=utc_now(),
            final_diagnostic_not_before_utc=budget.state['new_experiment_deadline_utc']);write_json(path,progress)
        print('FINAL PREPARATION PASS; reduced targets untouched; resume at original final buffer',flush=True)
        return
    # No candidate/feature/parameter changes are permitted after this step.
    primary_directory=ROOT/frozen['candidates']['PRIMARY_DEV_CANDIDATE']['model_directory']
    if not any(e['kind']=='primary' and e.get('scored_once') for e in read('REDUCED_TEST_USAGE_LEDGER.json')['entries']):
        run('official_primary_diagnostic','scripts/reduced_diagnostic.py','--kind','primary','--model-directory',primary_directory)
    run('final_tests','scripts/run_pytest_final.py')
    run('final_audit','scripts/final_audit_real.py')
    budget.checkpoint(status='FINAL_VALIDATION_PASS_HANDOFF_IN_PROGRESS',LOCAL_FINAL_SEAL_OPENED=False,
        reduced_scored_uses=2,next_action='Build numerical handoff, reproduce the final source archive, then verify a new Desktop ZIP.')
    run('numerical_handoff','scripts/build_handoff_real.py')
    run('source_snapshot','scripts/build_snapshot_real.py')
    source=read('artifacts/handoff/SNAPSHOT_VERIFICATION_REAL.json')
    run('source_reproduction','scripts/reproduce_snapshot.py',source['archive'],'--artifact-audit','final')
    run('handoff_after_reproduction','scripts/build_handoff_real.py')
    budget.checkpoint(status='FINAL_SOURCE_REPRODUCED_PACKAGING_DESKTOP',LOCAL_FINAL_SEAL_OPENED=False,
        reduced_scored_uses=2,next_action='Create and verify the final Desktop ZIP; completed project status is summarized in 01_PROJECT_STATUS.json.')
    run('desktop_zip','scripts/build_snapshot_real.py','--desktop','--require-reproduction')
    archive=read('artifacts/handoff/DESKTOP_ZIP_VERIFICATION_REAL.json')
    if archive['status']!='PASS' or sha256(archive['archive'])!=archive['sha256']:raise RuntimeError('Final Desktop ZIP identity failed')
    progress.update(status='PASS',active_step=None,completed_utc=utc_now(),desktop_archive=archive);write_json(path,progress)
    budget.checkpoint(status='COMPLETE_FINAL_AUDIT_AND_HANDOFF',LOCAL_FINAL_SEAL_OPENED=False,reduced_scored_uses=2,desktop_archive=archive['archive'])
    print('FINALIZATION PASS',archive['archive'],archive['bytes'],'bytes',flush=True)


if __name__=='__main__':main()
