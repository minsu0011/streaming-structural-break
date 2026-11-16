"""Matched innovation-group uncertainty for fixed synthetic counterexamples."""
import hashlib
from itertools import product
import numpy as np
from src.next8.common import BASE,read_json,digest
from src.next8.synthetic_controls import BACKGROUNDS,H_LENGTHS,INTERVENTIONS,block_coordinates
from src.next8.synthetic_evaluation import WINDOWS,SCORE_METHODS,AUX_METHODS
from src.next8.features import DETAILED,HORIZONS,NAMES,N_FEATURES

SCORE_METRICS=('mean_probability','max_probability','fraction_score_gt_.5','fraction_score_gt_.8')
ERROR_METRICS=('MSE','MAE')
MODEL_PAIRS=(('M0','M5'),('M2','M5'),('M3','M5'),('M4','M5'),('M0','M6'),
             ('BL95_M3_05_CONTROL','M6'),('BL95_M4_05_CONTROL','M6'),('BL95_M2_05_CONTROL','M6'))
ERROR_PAIRS=(('D0','D1'),('D1','D5'),('D5','D5_CROSS'))


def bootstrap_group_means(values,strata,*,seed=8101604,draws=2000):
    """One row per innovation group; resample within background strata.

    Columns may have unavailable H states. Such groups are excluded identically
    from point/replicates for that column and their support is reported. Exact
    duplicate columns and constants reuse mathematical bootstrap equivalence.
    """
    values=np.asarray(values,dtype=np.float64);strata=np.asarray(strata)
    if values.ndim!=2 or len(values)!=len(strata):raise ValueError('One row per independent innovation group required')
    finite=np.isfinite(values);count=finite.sum(axis=0)
    point=np.divide(np.where(finite,values,0).sum(axis=0),count,out=np.full(values.shape[1],np.nan),where=count>0)
    low=np.full(len(point),np.nan);high=low.copy();fraction=low.copy();supported=count>=64
    rng=np.random.default_rng(seed);weights=np.zeros((draws,len(values)),dtype=np.float64)
    for label in np.unique(strata):
        ix=np.flatnonzero(strata==label)
        weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=draws)
    unique={};aliases={};columns=[];constant_count=0
    for column in np.flatnonzero(supported):
        v=values[:,column];mask=finite[:,column];valid=v[mask]
        if np.all(valid==valid[0]):
            low[column]=high[column]=point[column]=valid[0];fraction[column]=float(valid[0]>0);constant_count+=1;continue
        canonical=np.where(mask,v,0).astype('<f8');signature=hashlib.sha256(mask.tobytes()+canonical.tobytes()).hexdigest()
        if signature in unique:
            donor=unique[signature]
            assert np.array_equal(finite[:,donor],mask) and np.array_equal(values[mask,donor],valid)
            aliases[column]=donor
        else:unique[signature]=column;columns.append(column)
    for start in range(0,len(columns),128):
        cols=columns[start:start+128];v=values[:,cols];ok=np.isfinite(v)
        numerator=weights@np.where(ok,v,0);denominator=weights@ok.astype(np.float64)
        samples=np.divide(numerator,denominator,out=np.full(numerator.shape,np.nan),where=denominator>0)
        for k,column in enumerate(cols):
            good=np.isfinite(samples[:,k])
            if good.sum()<1900:continue
            a=samples[good,k];low[column],high[column]=np.quantile(a,[.025,.975]);fraction[column]=np.mean(a>0)
    for column,donor in aliases.items():low[column]=low[donor];high[column]=high[donor];fraction[column]=fraction[donor]
    return {'point':point,'ci_low':low,'ci_high':high,'positive_resample_fraction':fraction,'groups':count,
        'computed_columns':len(columns),'constant_columns':constant_count,'exact_alias_columns':len(aliases),'draws':draws}


def finite_number(v):
    return float(v) if np.isfinite(v) else None


def load_synthetic_group_arrays():
    shapes={'score':(512,2,5,len(WINDOWS),len(SCORE_METHODS),4),
            'error':(512,2,len(DETAILED),len(HORIZONS),len(AUX_METHODS),5,len(WINDOWS),2),
            'error_count':(512,2,len(DETAILED),len(HORIZONS),5,len(WINDOWS)),
            'clip':(512,2,5,N_FEATURES)}
    output={k:np.full(shape,np.nan,dtype=np.float64) for k,shape in shapes.items()}
    expected={}
    for part in range(16):
        done=read_json(BASE/'synthetic_evaluation'/f'PART_{part}_DONE.json')
        assert done['status']=='COMPLETE' and done['parts']==16
        for r in done['records']:
            assert r['block'] not in expected;expected[r['block']]=r['done_sha256']
    assert set(expected)==set(range(1024))
    for block in range(1024):
        p=BASE/'synthetic_evaluation'/f'block_{block:04d}';assert digest(p/'DONE.json')==expected[block]
        done=read_json(p/'DONE.json');bg,index,H,group=block_coordinates(block);hi=H_LENGTHS.index(H)
        assert (done['background'],done['innovation_index'],done['H_length'],done['dependency_group'])==(bg,index,H,group)
        assert done['score_methods']==list(SCORE_METHODS) and done['auxiliary_methods']==list(AUX_METHODS)
        assert done['auxiliary_features']==list(DETAILED) and done['horizons']==list(HORIZONS)
        assert done['windows']==[list(w) for w in WINDOWS]
        assert digest(p/'SUMMARY.npz')==done['files_sha256']['SUMMARY.npz']
        with np.load(p/'SUMMARY.npz',allow_pickle=False) as a:
            score=a['score_summary'];error=a['movement_error'];counts=a['movement_count']
            assert score.shape[0]==error.shape[0]==counts.shape[0]==5
            assert np.isfinite(score).all() and np.all(counts==counts[0])
            # A fold with an unavailable value is never silently removed from
            # the five-procedure average. All five evaluate identical targets.
            assert np.all(np.isfinite(error)==np.isfinite(error[0]))
            output['score'][group,hi]=score.mean(axis=0)
            output['error'][group,hi]=error.mean(axis=0)
            output['error_count'][group,hi]=counts[0]
        feature=read_json(BASE/'synthetic_controls'/f'block_{block:04d}'/'DONE.json')
        assert feature['derived_features_sha256']==done['synthetic_features_sha256']
        output['clip'][group,hi]=np.asarray([r['boundary_fraction_per_feature'] for r in feature['summary']])
    return output


def tensor_rows(values,definitions,*,kind):
    """Retain every fixed dimension, with H-paired differences in one group."""
    assert values.shape[:2]==(512,2)
    labels=np.repeat(np.arange(len(BACKGROUNDS)),128);rows=[];operations=[]
    for background in ('ALL_BACKGROUNDS',*BACKGROUNDS):
        keep=np.ones(512,dtype=bool) if background=='ALL_BACKGROUNDS' else labels==BACKGROUNDS.index(background)
        for h_name,array in ((str(H_LENGTHS[0]),values[:,0]),(str(H_LENGTHS[1]),values[:,1]),
                             ('H2048_MINUS_H512_MATCHED',values[:,1]-values[:,0])):
            flat=array[keep].reshape(keep.sum(),-1);assert flat.shape[1]==len(definitions)
            evidence=bootstrap_group_means(flat,labels[keep])
            for column,definition in enumerate(definitions):
                row={'kind':kind,'background':background,'H_comparison':h_name,**definition,
                    'independent_innovation_groups':int(evidence['groups'][column]),
                    'estimate':finite_number(evidence['point'][column]),'ci_low':finite_number(evidence['ci_low'][column]),
                    'ci_high':finite_number(evidence['ci_high'][column]),
                    'positive_resample_fraction':finite_number(evidence['positive_resample_fraction'][column])}
                row['status']='SUPPORTED' if row['ci_low'] is not None else 'INSUFFICIENT_H_STATE_SUPPORT'
                rows.append(row)
            operations.append({'kind':kind,'background':background,'H_comparison':h_name,
                **{k:evidence[k] for k in ('computed_columns','constant_columns','exact_alias_columns','draws')}})
    return rows,operations


def synthetic_summary_tables(arrays):
    """Yield bounded tables so large all-window results need not coexist."""
    score=arrays['score'];error=arrays['error'];clip=arrays['clip']
    score_def=[{'intervention':INTERVENTIONS[v],'window':WINDOWS[w][0],'method':SCORE_METHODS[m],'metric':SCORE_METRICS[k]}
               for v,w,m,k in product(range(5),range(len(WINDOWS)),range(len(SCORE_METHODS)),range(4))]
    yield 'SCORE_PROBES',*tensor_rows(score,score_def,kind='ABSOLUTE_SCORE_PROBE')
    contrasts=np.stack([score[...,SCORE_METHODS.index(c),:]-score[...,SCORE_METHODS.index(r),:] for r,c in MODEL_PAIRS],axis=-2)
    definitions=[{'intervention':INTERVENTIONS[v],'window':WINDOWS[w][0],'reference':MODEL_PAIRS[m][0],
        'candidate':MODEL_PAIRS[m][1],'metric':SCORE_METRICS[k],'effect':'Candidate score probe minus reference; not TS-AUC improvement'}
        for v,w,m,k in product(range(5),range(len(WINDOWS)),range(len(MODEL_PAIRS)),range(4))]
    yield 'MODEL_CONTRASTS',*tensor_rows(contrasts,definitions,kind='MATCHED_MODEL_SCORE_DIFFERENCE')
    interventions=score[:,:,1:]-score[:,:,:1]
    definitions=[{'intervention':INTERVENTIONS[v+1],'reference_intervention':'none','window':WINDOWS[w][0],
        'method':SCORE_METHODS[m],'metric':SCORE_METRICS[k],'effect':'Intervention minus identical innovation null path'}
        for v,w,m,k in product(range(4),range(len(WINDOWS)),range(len(SCORE_METHODS)),range(4))]
    yield 'INTERVENTION_CONTRASTS',*tensor_rows(interventions,definitions,kind='MATCHED_INTERVENTION_RESPONSE')
    definitions=[{'feature':DETAILED[j],'name':NAMES[DETAILED[j]],'horizon':HORIZONS[h],'method':AUX_METHODS[m],
        'intervention':INTERVENTIONS[v],'window':WINDOWS[w][0],'metric':ERROR_METRICS[k]}
        for j,h,m,v,w,k in product(range(len(DETAILED)),range(len(HORIZONS)),range(len(AUX_METHODS)),range(5),range(len(WINDOWS)),range(2))]
    yield 'MOVEMENT_ERRORS',*tensor_rows(error,definitions,kind='FROZEN_FORECAST_ERROR')
    gains=np.stack([error[:,:,:,:,AUX_METHODS.index(r)]-error[:,:,:,:,AUX_METHODS.index(c)] for r,c in ERROR_PAIRS],axis=4)
    definitions=[{'feature':DETAILED[j],'name':NAMES[DETAILED[j]],'horizon':HORIZONS[h],'reference':ERROR_PAIRS[m][0],
        'candidate':ERROR_PAIRS[m][1],'intervention':INTERVENTIONS[v],'window':WINDOWS[w][0],'metric':ERROR_METRICS[k],
        'effect':'Reference error minus candidate error; positive is improvement'}
        for j,h,m,v,w,k in product(range(len(DETAILED)),range(len(HORIZONS)),range(len(ERROR_PAIRS)),range(5),range(len(WINDOWS)),range(2))]
    yield 'MOVEMENT_GAINS',*tensor_rows(gains,definitions,kind='MATCHED_FORECAST_LOSS_GAIN')
    definitions=[{'intervention':INTERVENTIONS[v],'feature':j,'name':NAMES[j],'metric':'fraction_abs_normalized_F_ge12'}
                 for v,j in product(range(5),range(N_FEATURES))]
    yield 'CLIPPING',*tensor_rows(clip,definitions,kind='H_NORMALIZATION_BOUNDARY_FRACTION')
