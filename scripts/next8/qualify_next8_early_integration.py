"""Complete early-only inner procedure if the frozen normal M6 is used.

The normal eligibility receipts control unused computation only. No normal
eligibility bit is passed as an early eligibility decision or training input.
"""
from pathlib import Path
import argparse
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
import numpy as np
import joblib
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity,note
from src.next8.cache import FeatureCache,immutable_npz
from src.next8.data import outer_scope,inner_scope
from src.next8.nested_auxiliary import early_scope,selected_cross_branch_needed
from src.next8.joint_models import load_heldout_states
from src.next8.features import DETAILED,UNKNOWN
from src.next8.early_reference import fit_early_reference
from src.next8.early_integration import early_prediction_rows,predict_early_head,normal_integration_relevant
from src.next8.integration import fixed_blend,integration_eligible
from src.next8.paired_bootstrap import evaluate_partition,summarize_paired
from src.next6.data import reference_matrix


def stage(script,arguments,done_path):
    if done_path.exists():
        done=read_json(done_path)
        assert done['status']=='COMPLETE'
        for name,sha in done.get('files_sha256',{}).items():assert digest(done_path.parent/name)==sha
        return
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/next8'/script),*arguments],cwd=ROOT,check=True)
    assert read_json(done_path)['status']=='COMPLETE'


def inner_reference(cache,scope,rows,directory,lock):
    components=[];sources={}
    for kind in ('J102','C205'):
        p=directory/kind;p.mkdir(parents=True,exist_ok=True)
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json')
            assert done['scope']==scope.provenance()
            assert digest(p/'model.joblib')==done['model_sha256']
            assert digest(p/'PREDICTIONS.npz')==done['prediction_sha256']
            with np.load(p/'PREDICTIONS.npz',allow_pickle=False) as a:
                np.testing.assert_array_equal(a['global_row'],rows);prediction=a['prediction'].copy()
        else:
            if any(p.iterdir()):raise RuntimeError('Partial early-inner reference preserved')
            with activity(f'Early inner BL80 {scope.name} {kind}','EARLY_INNER_REFERENCE_FIT','B5_DOMAIN_GENERALIZATION'):
                x=reference_matrix(kind,cache.data)
                rp=ROOT/f'artifacts/next5/references/registrations/REF_{kind}.json'
                assert digest(rp)==lock['registration_sha256'][kind]
                model=fit_early_reference(x,cache.meta,scope,read_json(rp),cache.data.guard)
                model.save(p/'model.joblib');prediction=np.empty(len(rows),dtype=np.float64)
                for start in range(0,len(rows),50000):prediction[start:start+50000]=model.predict(x[rows[start:start+50000]])
                immutable_npz(p/'PREDICTIONS.npz',global_row=rows,prediction=prediction)
                atomic(p/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
                    'model_sha256':digest(p/'model.joblib'),'prediction_sha256':digest(p/'PREDICTIONS.npz'),
                    'training':model.training,'max_training_target_time':199,'max_prediction_time':199,'score_released':False})
                del x,model
        components.append(prediction)
        sources[kind]={'model_sha256':digest(p/'model.joblib'),'prediction_sha256':digest(p/'PREDICTIONS.npz')}
    return .8*components[0]+.2*components[1],sources


def main(fold):
    assert read_json(BASE/'EARLY_INTEGRATION_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock_path=BASE/'EARLY_INTEGRATION_IMPLEMENTATION_LOCK_NEXT8.json';lock=read_json(lock_path)
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    normal=[read_json(BASE/f'integration/outer_{f}/DONE.json') for f in range(5)]
    relevant=normal_integration_relevant(normal)
    cache=FeatureCache();outer=early_scope(outer_scope(fold,cache.data),True)
    directory=BASE/'early_integration'/outer.name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():raise RuntimeError('Completed early qualification cannot be rerun')
    gate_sources={f'integration/outer_{f}/DONE.json':digest(BASE/f'integration/outer_{f}/DONE.json') for f in range(5)}
    if not relevant:
        atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':outer.provenance(),
            'required':False,'eligible':False,'M6_status':'NOT_RUN_NORMAL_PROCEDURE_UNUSED',
            'normal_gate_source_sha256':gate_sources,'early_eligibility_inferred_from_normal_gate':False,
            'reason':'Every frozen primary M6 falls back to M0; no separate early M6 discovery or unused fits performed'})
        return
    points=[];samples=[];sources={}
    for inner in range(3):
        scope=early_scope(inner_scope(fold,inner,cache.data),True)
        arguments=['--fold',str(fold),'--meta-inner',str(inner),'--early']
        for script,parent,extra in (
            ('screen_next8_scope.py','selection',[]),
            ('select_next8_movement_parents.py','parent_selection',[]),
            ('audit_h_context_scope.py','h_context_selection',[]),
            ('fit_next8_graph_states.py','graph_states',['--mode','inference']),
            ('fit_next8_graph_states.py','meta_state_inputs',['--mode','meta-leaf']),
            ('fit_next8_models.py','models',[])):
            stage(script,arguments+extra,BASE/parent/scope.name/'DONE.json')
        p=directory/f'inner_{inner}';p.mkdir(parents=True,exist_ok=True)
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json');assert done['scope']==scope.provenance()
            for name,sha in done['files_sha256'].items():assert digest(p/name)==sha
            points.append(done['points'])
            with np.load(p/'BOOTSTRAP.npz',allow_pickle=False) as a:samples.append({k:a[k].copy() for k in a.files})
            sources[scope.name]=digest(p/'DONE.json');continue
        with activity(f'Early-only M6 inner qualification {scope.name}','EARLY_INTEGRATION_GATE','B5_DOMAIN_GENERALIZATION'):
            rows=early_prediction_rows(cache,scope)
            outer.admit_training(np.unique(cache.meta['dataset_id'][rows]))
            selection=read_json(BASE/'h_context_selection'/scope.name/'SELECTION.json')
            graph=read_json(BASE/'parent_selection'/scope.name/'GRAPH.json')
            if selected_cross_branch_needed(selection,graph):
                cross,state_receipt=load_heldout_states(cache,scope,rows,BASE/'graph_states'/scope.name)
            else:cross=np.full((len(rows),len(DETAILED)),UNKNOWN,dtype=np.int8);state_receipt={'required':False}
            mp=BASE/'models'/scope.name;md=read_json(mp/'DONE.json');actual=md['model_aliases']['M5']
            assert digest(mp/(actual+'.joblib'))==md['files_sha256'][actual+'.joblib']
            model=joblib.load(mp/(actual+'.joblib'))
            m5=predict_early_head(cache,scope,selection,model,actual,rows,cross)
            baseline,reference_sources=inner_reference(cache,scope,rows,p/'baseline',lock)
            predictions={'M0':baseline,'M5':m5,'M6':fixed_blend(baseline,m5)}
            immutable_npz(p/'PREDICTIONS.npz',global_row=rows,**predictions)
            groups=np.asarray([cache.data.guard.split.groups[int(s)] for s in cache.meta['dataset_id'][rows]])
            point,sample,receipt=evaluate_partition(cache.meta['target'][rows],predictions,cache.meta['time_online'][rows],
                groups,partition=f'outer{fold}_inner{inner}_EARLY_T199_M6',draws=2000)
            immutable_npz(p/'BOOTSTRAP.npz',**sample)
            atomic(p/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
                'points':point,'resampling':receipt,'state_provenance':state_receipt,'reference_sources':reference_sources,
                'M5_model_sha256':digest(mp/(actual+'.joblib')),'max_scored_time':199,
                'all_scored_labels_belong_to_outer_training':True,'complete_inner_selection_training_cutoff':199,
                'files_sha256':{name:digest(p/name) for name in ('PREDICTIONS.npz','BOOTSTRAP.npz')}})
            points.append(point);samples.append(sample);sources[scope.name]=digest(p/'DONE.json')
    comparison=summarize_paired(points,samples,'M0','M6');eligible=integration_eligible(comparison)
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':outer.provenance(),
        'required':True,'eligible':eligible,'comparison':comparison,
        'M6_status':'ELIGIBLE_EARLY_FIXED_95_5' if eligible else 'NOT_RUN_NO_EARLY_INDEPENDENT_INFO',
        'normal_gate_source_sha256':gate_sources,'normal_eligibility_reused_as_early_decision':False,
        'inner_source_sha256':sources,'all_feature_parent_state_weight_fit_and_eligibility_targets_max_time':199,
        'outer_validation_scores_opened':False,'implementation_lock_sha256':digest(lock_path)})
    note('Complete early-only integrated procedure',outer.name+'; no late labels or normal-gate decision reused','B5_DOMAIN_GENERALIZATION')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    main(p.parse_args().fold)
