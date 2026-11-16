"""Apply frozen RP models to matched synthetic paths, without any refits."""
import copy
import numpy as np
import joblib
from src.next8.common import ROOT,BASE,read_json,digest
from src.next8.synthetic_controls import (SyntheticCache,matched_path,interventions,block_coordinates,INTERVENTIONS,ONLINE_LENGTH)
from src.next8.features import DETAILED,HORIZONS,UNKNOWN
from src.next8.graph_states import predict_graph_state
from src.next8.streaming_runtime import load_prepared
from src.next8.joint_models import joint_design,model_design
from src.next8.movement import own_design,mechanical_design,movement_target,forecast_delta
from src.next8.integration import fixed_blend
from src.next8.protocol import TIME_BUCKETS
from src.next5.model import FeatureModel

WINDOWS=(('ALL',0,None),)+tuple((f'TIME_{k}',a,b) for k,(a,b) in enumerate(TIME_BUCKETS))+(
    ('SHOCK_256_287',256,287),('SHOCK_DECAY_288_399',288,399))
AUX_METHODS=('D0','D1','D5','D5_CROSS')
SCORE_METHODS=('M0','M1','M2','M3','M4','M5','M6','BL95_M3_05_CONTROL','BL95_M4_05_CONTROL','BL95_M2_05_CONTROL')


def load_frozen_models(release):
    folds=[];audit=read_json(BASE/'BASELINE_PROVENANCE_NEXT8.json')
    def load(path):
        assert digest(path)==release['files_sha256'][path.relative_to(BASE).as_posix()]
        return joblib.load(path)
    for fold in range(5):
        name=f'outer_{fold}';mp=BASE/'models'/name;sp=BASE/'h_context_selection'/name/'SELECTION.json'
        heads={m:load_prepared(mp,sp,BASE/'graph_states'/name,m,resident=False) for m in ('M1','M2','M3','M4','M5')}
        for m in heads.values():
            actual=m.method;assert digest(mp/(actual+'.joblib'))==release['files_sha256'][(mp/(actual+'.joblib')).relative_to(BASE).as_posix()]
        baseline=[]
        for kind in ('J102','C205'):
            p=ROOT/f'artifacts/next5/references/fits/PRIMARY/full/{kind}/fold_{fold}/model.joblib'
            assert digest(p)==audit['files_sha256'][p.relative_to(ROOT).as_posix()];baseline.append(FeatureModel.load(p))
        auxiliary={}
        for j in DETAILED:
            for h in HORIZONS:
                p=BASE/'forecasts'/name/f'F{j:03d}_h{h}';done=read_json(p/'DONE.json');models={};donors={}
                for method in AUX_METHODS[1:]:
                    donor=done['method_aliases'][method]
                    if donor not in donors:donors[donor]=load(p/(donor+'.joblib'))['delta']
                    models[method]=donors[donor]
                auxiliary[j,h]={'models':models,'parents':done['parents'],'aliases':done['method_aliases']}
        ip=BASE/'integration'/name/'DONE.json';assert digest(ip)==release['files_sha256'][ip.relative_to(BASE).as_posix()]
        folds.append({'fold':fold,'heads':heads,'baseline':baseline,'auxiliary':auxiliary,'eligible':read_json(ip)['eligible']})
    return folds


def baseline_matrices(block,models):
    background,index,H_length,_=block_coordinates(block);h,online=matched_path(background,index,H_length)
    variants=interventions(online,index);result=[]
    for model in models:
        initial=model.make_state(h)
        result.append(np.concatenate([copy.deepcopy(initial).replay(values) for values in variants]))
    return result


def predict_block_heads(cache,heads,baseline_models,baseline_features,eligible):
    rows=np.arange(len(cache.values));cross=np.full((len(rows),len(DETAILED)),UNKNOWN,dtype=np.int8)
    prepared=heads['M5']
    for j,model in prepared.graph_models.items():cross[:,DETAILED.index(j)]=predict_graph_state(cache,rows,j,model)[0]
    predictions={'M0':.8*baseline_models[0].predict(baseline_features[0])+.2*baseline_models[1].predict(baseline_features[1])}
    aliases={};seen={}
    for method,prepared in heads.items():
        key=prepared.method
        if key in seen:predictions[method]=predictions[seen[key]];aliases[method]=seen[key];continue
        design=model_design(prepared.model,joint_design(cache,rows,prepared.selection,key,cross))
        predictions[method]=prepared.model.predict_proba(design)[:,1].astype(np.float32).astype(np.float64)
        seen[key]=method;aliases[method]=method
    predictions['M6']=fixed_blend(predictions['M0'],predictions['M5']) if eligible else predictions['M0']
    for method in ('M3','M4','M2'):
        predictions[f'BL95_{method}_05_CONTROL']=fixed_blend(predictions['M0'],predictions[method]) if eligible else predictions['M0']
    return np.column_stack([predictions[m] for m in SCORE_METHODS]),cross,aliases


def summarize_scores(scores):
    # [time rows, method] -> [intervention, window, method, mean/max/p>.5/p>.8].
    values=np.asarray(scores).reshape(len(INTERVENTIONS),ONLINE_LENGTH,-1)
    result=np.empty((len(INTERVENTIONS),len(WINDOWS),values.shape[-1],4),dtype=np.float64)
    t=np.arange(ONLINE_LENGTH)
    for w,(_,a,b) in enumerate(WINDOWS):
        keep=(t>=a)&(True if b is None else t<=b);v=values[:,keep,:]
        result[:,w,:,0]=v.mean(axis=1);result[:,w,:,1]=v.max(axis=1)
        result[:,w,:,2]=(v>.5).mean(axis=1);result[:,w,:,3]=(v>.8).mean(axis=1)
    return result


def auxiliary_errors(cache,auxiliary):
    shape=(len(DETAILED),len(HORIZONS),len(AUX_METHODS),len(INTERVENTIONS),len(WINDOWS),2)
    result=np.full(shape,np.nan,dtype=np.float64)
    counts=np.zeros((len(DETAILED),len(HORIZONS),len(INTERVENTIONS),len(WINDOWS)),dtype=np.int32)
    sampled=np.flatnonzero(cache.meta['time_online']%8==0)
    for ji,j in enumerate(DETAILED):
        for hi,h in enumerate(HORIZONS):
            rows=cache.forecast_rows(sampled,h);target=movement_target(cache,rows,j,h)['delta'];current=cache.current(rows,[j])[:,0]
            spec=auxiliary[j,h];pred={'D0':np.zeros(len(rows),dtype=np.float32)};seen={}
            for method in AUX_METHODS[1:]:
                donor=spec['aliases'][method]
                if donor in seen:pred[method]=pred[seen[donor]];continue
                design=own_design(cache,rows,j,h) if method=='D1' else mechanical_design(cache,rows,j,h,parents=spec['parents'] if method=='D5_CROSS' else ())
                pred[method]=forecast_delta(spec['models'][method],design,current).astype(np.float32)
                seen[donor]=method
            t=cache.meta['time_online'][rows];variant=cache.meta['intervention'][rows]
            known=cache.series['valid'][cache.series_index[rows],j]
            for vi in range(len(INTERVENTIONS)):
                for wi,(_,a,b) in enumerate(WINDOWS):
                    keep=(variant==vi)&known&(t>=a)&(True if b is None else t<=b);counts[ji,hi,vi,wi]=int(keep.sum())
                    if not keep.any():continue
                    for mi,method in enumerate(AUX_METHODS):
                        error=np.asarray(target[keep],dtype=float)-pred[method][keep]
                        result[ji,hi,mi,vi,wi]=[np.mean(error**2),np.mean(np.abs(error))]
    return result,counts
