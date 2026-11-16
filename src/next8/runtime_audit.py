"""Fixed case selection and label-free research-fold execution checks."""
import copy
import hashlib
import time
import numpy as np
import joblib
from src.next8.common import ROOT,BASE,read_json,digest
from src.next8.protocol import H_BUCKETS
from src.next8.streaming_runtime import load_prepared,IntegratedState
from src.next8.multihead_runtime import PreparedMultiHead
from src.next8.joint_models import joint_design,model_design
from src.next8.data import baseline_oof
from src.next5.model import FeatureModel
from src.next5.runtime import SharedResearchBaseline


def fixed_cases(cache,scope,per_H_bin=2):
    ids=cache.data.ids;length=np.diff(cache.data.series['historical_offsets']);records=[]
    for k,(lo,hi) in enumerate(H_BUCKETS):
        candidates=sorted(int(s) for s,n in zip(ids,length) if int(s) in scope.validation_ids and n>=lo and (hi is None or n<=hi))
        records.append({'H_bin':k,'available':len(candidates),'selected_ids':candidates[:per_H_bin]})
    return records


def observations(cache,sid):
    cache.data.guard.admit([int(sid)],purpose='NEXT8 label-free frozen runtime replay')
    k=cache.data.id_to_index[int(sid)];ho=cache.data.series['historical_offsets'];oo=cache.data.series['online_offsets']
    # Do not call ResearchArrays.get: its third return value is tau.
    return cache.data.historical[ho[k]:ho[k+1]],cache.data.online[oo[k]:oo[k+1]],np.arange(oo[k],oo[k+1],dtype=np.int64)


def load_scope_runtime(cache,scope,fold,*,early=False):
    directory=BASE/'models'/scope.name;done=read_json(directory/'DONE.json')
    assert done['scope']==scope.provenance()
    for name,sha in done['files_sha256'].items():assert digest(directory/name)==sha
    selection=BASE/'h_context_selection'/scope.name/'SELECTION.json'
    assert digest(selection)==read_json(selection.parent/'DONE.json')['selection_sha256']
    heads={method:load_prepared(directory,selection,BASE/'graph_states'/scope.name,method,resident=True) for method in done['methods']}
    if early:
        bp=BASE/'early_baseline'/scope.name;bd=read_json(bp/'DONE.json')
        assert bd['scope']==scope.provenance()
        paths=[bp/k/'model.joblib' for k in ('J102','C205')]
        for path in paths:assert digest(path)==read_json(path.parent/'DONE.json')['model_sha256']
        assert digest(bp/'BL80_PREDICTIONS.npz')==bd['prediction_sha256']
        with np.load(bp/'BL80_PREDICTIONS.npz',allow_pickle=False) as a:reference={k:a[k].copy() for k in a.files}
        integration=read_json(BASE/'early_integration'/scope.name/'DONE.json')
    else:
        audit=read_json(BASE/'BASELINE_PROVENANCE_NEXT8.json')
        paths=[ROOT/f'artifacts/next5/references/fits/PRIMARY/full/{k}/fold_{fold}/model.joblib' for k in ('J102','C205')]
        for path in paths:assert digest(path)==audit['files_sha256'][path.relative_to(ROOT).as_posix()]
        reference=baseline_oof(fold,data=cache.data)
        integration=read_json(BASE/'integration'/scope.name/'DONE.json')
    baseline=SharedResearchBaseline(*(FeatureModel.load(p) for p in paths))
    with np.load(directory/'PREDICTIONS.npz',allow_pickle=False) as a:prediction={k:a[k].copy() for k in a.files}
    gp=BASE/'graph_states'/scope.name;gd=read_json(gp/'DONE.json')
    assert digest(gp/'STATE_PREDICTIONS.npz')==gd['files_sha256']['STATE_PREDICTIONS.npz']
    with np.load(gp/'STATE_PREDICTIONS.npz',allow_pickle=False) as a:states={k:a[k].copy() for k in ('global_row','state')}
    return heads,baseline,reference,prediction,states,integration


def timed_path(factory,h,online):
    start=time.perf_counter_ns();state=factory(h);init=time.perf_counter_ns()-start
    prediction=np.empty(len(online));elapsed=np.empty(len(online),dtype=np.int64)
    for k,x in enumerate(online):
        start=time.perf_counter_ns();prediction[k]=state.predict_one(x);elapsed[k]=time.perf_counter_ns()-start
    return state,prediction,elapsed,init


def latency_summary(elapsed):
    elapsed=np.asarray(elapsed,dtype=np.float64)
    return {'points':len(elapsed),'median_microseconds':float(np.median(elapsed)/1000),
        'p99_microseconds':float(np.quantile(elapsed,.99)/1000),'mean_microseconds':float(elapsed.mean()/1000),
        'maximum_microseconds':float(elapsed.max()/1000),'measurement':'Warm process scalar API, perf_counter_ns; initialization excluded'}


def audit_case(cache,sid,heads,baseline,reference,saved,states,eligible,*,timings=True):
    h,online,rows=observations(cache,sid);prepared=PreparedMultiHead(heads)
    loc=np.searchsorted(states['global_row'],rows);np.testing.assert_array_equal(states['global_row'][loc],rows)
    cross=states['state'][loc]
    expected={};native={}
    for name,p in heads.items():
        design=model_design(p.model,joint_design(cache,rows,p.selection,p.method,cross))
        expected[name]=design.values;native[name]=p.model.predict_proba(design)[:,1]
    joint=prepared.new_state(h);actual=np.empty((len(rows),len(heads)),dtype=np.float32)
    raw=np.empty(actual.shape,dtype=np.float64);maximum_coordinate_error=0.;maximum_feature_error=0.
    used=np.asarray(joint.used_features,dtype=np.int32)
    for k,x in enumerate(online):
        actual[k]=joint.predict_one(x);raw[k]=joint.unrounded_probabilities
        for j,name in enumerate(heads):
            coordinates=joint.args[-1][1][j][0][-1]
            maximum_coordinate_error=max(maximum_coordinate_error,float(np.max(np.abs(coordinates-expected[name][k]))))
        if len(used):maximum_feature_error=max(maximum_feature_error,float(np.max(np.abs(joint.args[-1][2][k%32,used]-cache.values[rows[k],used]))))
    assert maximum_coordinate_error==maximum_feature_error==0.
    native_error={name:float(np.max(np.abs(raw[:,j]-native[name]))) for j,name in enumerate(heads)}
    assert max(native_error.values())<1e-12,native_error
    take=np.isin(rows,saved['global_row']);at=np.searchsorted(saved['global_row'],rows[take])
    errors={name:float(np.max(np.abs(actual[take,j].astype(float)-saved[name][at]))) if take.any() else None for j,name in enumerate(heads)}
    assert all(x is None or x<=np.finfo(np.float32).eps for x in errors.values()),errors
    # Exact fresh/chunk/snapshot reproduction does not refit any model/map.
    boundary=min(73,len(online));state=prepared.new_state(h);prefix=state.replay(online[:boundary]);snapshot=state.snapshot()
    suffix=state.replay(online[boundary:]);np.testing.assert_array_equal(suffix,snapshot.replay(online[boundary:]))
    np.testing.assert_array_equal(np.vstack([prefix,suffix]),actual)
    changed=online.copy();changed[boundary:]=-7*changed[boundary:]
    np.testing.assert_array_equal(prepared.new_state(h).replay(changed)[:boundary],actual[:boundary])
    result={'dataset_id':int(sid),'H_length':len(h),'online_points':len(online),'saved_prediction_points':int(take.sum()),
        'maximum_coordinate_error':maximum_coordinate_error,'maximum_normalized_feature_error':maximum_feature_error,
        'native_float64_error':native_error,'saved_float32_error':errors,'exact_saved_prefix_chunk_future_suffix':True,
        'prediction_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'shared_instance_array_bytes':joint.total_instance_array_bytes,
        'used_feature_count':len(used),'eligible_integrated':bool(eligible),'labels_or_tau_used':False}
    times={};initialization={};memory={}
    if timings:
        factories={'M0':baseline.new_state,**{name:p.new_state for name,p in heads.items()}}
        if eligible:factories['M6']=lambda H:IntegratedState(baseline,heads['M5'],H,True)
        paths={}
        for name,factory in factories.items():
            state,p,elapsed,init=timed_path(factory,h,online);times[name]=elapsed;initialization[name]=init;paths[name]=p
            memory[name]=getattr(state,'total_instance_array_bytes',None)
        pos=np.isin(rows,reference['global_row']);refat=np.searchsorted(reference['global_row'],rows[pos])
        baseline_error=float(np.max(np.abs(paths['M0'][pos]-reference['prediction'][refat]))) if pos.any() else None
        assert baseline_error is None or baseline_error<1e-12,baseline_error
        for j,name in enumerate(heads):np.testing.assert_array_equal(paths[name],actual[:,j])
        if eligible:np.testing.assert_array_equal(paths['M6'],.95*paths['M0']+.05*paths['M5'])
        result.update({'baseline_float64_error':baseline_error,'standalone_vs_shared_bitwise':True,'initialization_nanoseconds':initialization,'instance_array_bytes':memory})
    return result,times
