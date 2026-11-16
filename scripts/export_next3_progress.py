"""Consolidate honest ongoing results without declaring final selection or audit."""
from pathlib import Path
import sys
import os
import uuid
import json
import hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next.guard import SealGuard


def csv_report(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8',newline='') as stream:
        frame.to_csv(stream,index=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def completed_csvs(paths):
    """Never publish a worker's partial table or a table whose receipt disagrees."""
    frames=[]
    for path in sorted(paths):
        receipt=path.parent/'RESULTS.json'
        if not receipt.exists():continue
        record=read_json(receipt)
        if record.get('status')!='PASS':continue
        expected=record.get('files_sha256',{}).get(path.name)
        if expected is None or digest(path)!=expected:
            raise RuntimeError('Completed table lacks an exact committed receipt: '+str(path))
        frame=pd.read_csv(path)
        frame['source_receipt']=receipt.relative_to(ROOT).as_posix()
        frame['source_receipt_sha256']=digest(receipt)
        frames.append(frame)
    return frames


def main():
    guard=SealGuard(ROOT);guard.check_state()
    policy=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    rows=[];folds=[]
    for directory in sorted((BASE/'experiments').glob('*')):
        path=directory/'RESULTS.json'
        if not path.exists():continue
        current=read_json(path)
        name=directory.name
        hypothesis_path=BASE/'hypotheses'/f'{name}.json'
        hypothesis=read_json(hypothesis_path) if hypothesis_path.exists() else {}
        binding={'candidate_configuration':current.get('candidate',{}),
            'hypothesis_sha256':digest(hypothesis_path) if hypothesis_path.exists() else None,
            'research_policy_sha256':digest(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')}
        row={'experiment_id':name,'generation':'NEXT3','bottleneck':hypothesis.get('BOTTLENECK'),
            'hypothesis':hypothesis.get('HYPOTHESIS'),'change':hypothesis.get('CHANGE'),
            'current_stage':current.get('scope'),'status':current.get('scientific_disposition',current['status']),
            'screen_mean':None,'screen_delta':None,'full_mean':None,'full_median':None,'full_worst':None,
            'delta_vs_frozen_performance':None,'new_learner_fits':0,'new_gate_fits':0,'new_residual_fits':0,
            'result_sha256':digest(path),'hypothesis_sha256':digest(hypothesis_path) if hypothesis_path.exists() else None,
            'configuration_binding_sha256':hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'hypothesis_authority':'Immutable NEXT3 registration; inherited parent descriptive text in candidate config is not the new hypothesis',
            'seal_rows':0,'promoted':False}
        for scope in ('screen','full'):
            stage_path=directory/scope/'RESULTS.json'
            if not stage_path.exists():continue
            result=read_json(stage_path)
            if result['status']!='COMPLETE':continue
            residual_count=sum(record.get('new_residual_fits',0) for record in result.get('fold_records',[]))
            row['new_learner_fits']+=result.get('new_fits',0)-result.get('new_gate_fits',0)-residual_count
            row['new_gate_fits']+=result.get('new_gate_fits',0)
            row['new_residual_fits']+=residual_count
            if scope=='screen':row.update(screen_mean=result['mean'],screen_delta=result['mean_delta'])
            else:row.update(full_mean=result['mean'],full_median=result['median'],full_worst=result['worst_fold'],delta_vs_frozen_performance=result['mean_delta'])
            for detail in result.get('fold_records',[]):
                folds.append({'experiment_id':name,'scope':scope,'fold':detail['fold'],'ts_auc':detail['ts_auc'],
                    'reference_ts_auc':detail['reference_ts_auc'],'delta':detail['delta'],'seal_rows':0})
        rows.append(row)
    frame=pd.DataFrame(rows)
    csv_report(ROOT/'MODEL_RESULTS_NEXT3.csv',frame)
    csv_report(ROOT/'CV_RESULTS_NEXT3.csv',pd.DataFrame(folds))
    if len(frame):
        csv_report(ROOT/'ROUTER_RESULTS.csv',frame[frame.experiment_id.str.startswith('R')])
        csv_report(ROOT/'SCALE_REPAIR_RESULTS.csv',frame[frame.experiment_id.str.startswith('S')])
        csv_report(ROOT/'C205_ABLATION_RESULTS.csv',frame[frame.experiment_id.str.startswith('P')])
        csv_report(ROOT/'DEPENDENCE_SPECIALIST_RESULTS.csv',frame[frame.experiment_id.str.startswith(('P','I'))])
        csv_report(ROOT/'ORTHOGONAL_COMPLEMENT_RESULTS.csv',frame[frame.experiment_id.str.startswith('O')])
        csv_report(BASE/'ledger/EXPERIMENT_LEDGER_NEXT3.csv',frame)
    paired=completed_csvs((BASE/'forensic').glob('*/PAIRED_FRACTURES.csv'))
    if paired:
        combined=pd.concat(paired,ignore_index=True)
        combined['validation_kind']='ORIGINAL_GROUP_5FOLD_OOF_DIAGNOSTIC_SLICE'
        csv_report(ROOT/'PAIRED_BOOTSTRAP_NEXT3.csv',combined)
        csv_report(ROOT/'MECHANISM_RESULTS_NEXT3.csv',combined[combined.scope.str.startswith('mechanism_')])
        regimes=combined[combined.scope.str.startswith('regime_')].copy()
        regimes['validation_kind']='ORIGINAL_OOF_H_SLICE_NOT_LEAVE_H_OUT'
        csv_report(ROOT/'REGIME_RESULTS_NEXT3.csv',regimes)
    counters=completed_csvs((BASE/'counterexamples').rglob('COUNTEREXAMPLES.csv'))
    if counters:csv_report(ROOT/'COUNTEREXAMPLE_RESULTS_NEXT3.csv',pd.concat(counters,ignore_index=True))
    nulls=completed_csvs((BASE/'counterexamples').glob('*/LONG_NULL.csv'))
    if nulls:csv_report(ROOT/'LONG_NULL_RESULTS_NEXT3.csv',pd.concat(nulls,ignore_index=True))
    # Count actual fit completions; intended starts remain separately visible.
    inner=list((BASE/'nested').glob('*/outer_*/inner_*/attempt_*/COMPLETE.json'))
    inner_starts=list((BASE/'nested').glob('*/outer_*/inner_*/attempt_*/START.json'))
    starts=list((BASE/'experiments').glob('*/*/fold_*_FIT_START.json'))
    records=list((BASE/'experiments').glob('*/*/fold_?.json'))
    base_fits=len(inner)
    supervised=sum(read_json(p).get('new_fits',0) for p in records)
    gates=sum(read_json(p).get('new_gate_fits',0) for p in records)
    residuals=sum(read_json(p).get('new_residual_fits',0) for p in records)
    refits=[p for p in (BASE/'refits').glob('*/*/RESULTS.json') if read_json(p)['status']=='PASS']
    refit_count=sum(read_json(p).get('new_gate_fits',0)+read_json(p).get('new_base_fits',0)+read_json(p).get('new_fits',0) for p in refits)
    refit_starts=list((BASE/'refits').glob('*/*/FIT_START.json'))
    inventory={'created_utc':utc_now(),'status':'ONGOING','nested_base_fit_completions':base_fits,
        'candidate_supervised_fit_completions':supervised,'tiny_gate_fit_completions':gates,
        'tiny_residual_fit_completions':residuals,'complete_DEV_refit_completions':refit_count,
        'total_fit_completions':base_fits+supervised+gates+residuals+refit_count,
        'fit_starts':len(inner_starts)+len(starts)+len(refit_starts),
        'scientific_candidate_records':len(rows),'old_fits_recounted_as_new':0,
        'primitive_fit_receipt_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in inner+records+refits}}
    atomic_json(BASE/'ledger/FIT_INVENTORY.json',inventory)
    sessions=read_json(BASE/'SESSION.json');heartbeat=read_json(BASE/'runtime/HEARTBEAT.json')
    failures=[read_json(p) for p in (BASE/'jobs').glob('*/attempt_*/RESULT.json') if read_json(p)['status']=='FAILED']
    job_states=[read_json(p) for p in (BASE/'jobs').glob('*/STATE.json')]
    maintenance={'status':'ONGOING','created_utc':utc_now(),'wall_seconds':heartbeat['wall_seconds'],
        'productive_seconds':heartbeat['productive_seconds'],'unobserved_seconds':heartbeat['unobserved_seconds'],
        'planned_supervisor_restarts':len(list((BASE/'runtime/maintenance_updates').glob('SUPERVISOR_*_REPLACED.json'))),
        'supervisor_errors':len(list((BASE/'runtime/errors').glob('*.json'))),
        'registration_bugs_before_any_child_attempt':sum(s.get('attempt')==0 and s.get('status')=='QUARANTINED' and s.get('terminal_kind')!='SUPERSEDED_WITHOUT_EXECUTION' for s in job_states),
        'superseded_jobs_without_execution':sum(s.get('terminal_kind')=='SUPERSEDED_WITHOUT_EXECUTION' for s in job_states),
        'child_failed_attempts':len(failures),'resources':heartbeat['resources'],
        'last_completed_task_count':heartbeat['completed_count'],'modeling_blocked':sessions.get('modeling_blocked',False),
        'fit_inventory':{k:v for k,v in inventory.items() if k!='primitive_fit_receipt_sha256'},
        'old_experiment_ledger_unchanged':digest(ROOT/'EXPERIMENT_LEDGER.csv')==digest(BASE/'preflight/EXPERIMENT_LEDGER_BEFORE_NEXT3.csv')}
    atomic_json(ROOT/'MAINTENANCE_LOG.json',maintenance)
    atomic_json(ROOT/'SEAL_GUARD_NEXT3.json',{'status':'PASS','created_utc':utc_now(),
        'all_seals_unopened':True,'seals_opened':{name:False for name in guard.lock['partitions']},
        'DEV_series':len(guard.dev_ids),'seal_series':len(guard.seal_ids),'DEV_seal_overlap':len(guard.dev_ids&guard.seal_ids),
        'seal_lock_sha256':guard.lock_sha,'reduced_ledger_sha256':guard.ledger_sha,
        'reduced_new_usage':0,'public_submission_new':0,'cloud_submission_new':0,
        'boundary':'Application-level predicate/ID admission; not an OS sandbox'})
    progress('NEXT3 ongoing evidence exported',f'{len(rows)} candidates, {inventory["total_fit_completions"]} completed learner/gate fits; scientific role freeze is separate, final runtime delivery still pending')
    print('NEXT3 PROGRESS EXPORTED',len(rows),inventory['total_fit_completions'],'fits; NO FINAL DELIVERY CLAIM',flush=True)


if __name__=='__main__':main()
