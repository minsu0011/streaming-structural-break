"""Nested relational discovery, cross-fitted heads, deferred outer score release.

Reuses prior RP split/scorer/learner; completed graph/head jobs are resumable.
"""
from pathlib import Path
import sys,os,argparse,csv,gc,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import joblib
from src.next6.common import BASE,read_json,save,digest,activity,checkpoint,ids_hash,utc_now
from src.next6.data import arrays
from src.next6.graph import fit_graph,GraphState
from src.next6.model import fit_head,EXTRA_COLUMNS
from src.scoring.ts_auc import ts_auc

def rows_for(data,ids):
    return np.concatenate([np.arange(*data.series['online_offsets'][data.id_to_index[int(i)]:data.id_to_index[int(i)]+2],dtype=np.int64) for i in sorted(ids)])

def graph_job(data,metadata,values,fold,tag,trainids,validids,topology='selected',selected=None):
    data.guard.partition(trainids,validids)
    d=BASE/f'fold_{fold}/graphs/{tag}';d.mkdir(parents=True,exist_ok=True)
    if (d/'DONE.json').exists():
        r=read_json(d/'DONE.json')
        if r['training_ids_sha256']!=ids_hash(trainids) or r['validation_ids_sha256']!=ids_hash(validids):raise RuntimeError('Graph reuse boundary mismatch')
        if r['source_sha256']!=digest(ROOT/'src/next6/graph.py'):raise RuntimeError('Graph reuse source mismatch')
        return joblib.load(d/'graph.joblib'),np.load(d/'rows.npy',allow_pickle=False),np.load(d/'features.npy',mmap_mode='r',allow_pickle=False)
    with activity(f'graph_fold{fold}_{tag}','RELATION_ANALYSIS'):
        if (d/'graph.joblib').exists():
            g=joblib.load(d/'graph.joblib')
            if g.provenance['training_ids_sha256']!=ids_hash(trainids):raise RuntimeError('Partial graph membership mismatch')
        else:
            g,edges=fit_graph(data,metadata,values,trainids,topology=topology,selected=selected)
            joblib.dump(g,d/'graph.joblib')
            save(str((d/'TRAIN_PROVENANCE.json').relative_to(BASE)),g.provenance)
            if edges:
                with (d/'EDGE_DISCOVERY.csv').open('x',encoding='utf-8-sig',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(edges[0]));w.writeheader();w.writerows(edges)
        rows=rows_for(data,validids);np.save(d/'rows.npy',rows,allow_pickle=False)
        out=np.lib.format.open_memmap(d/'features.npy',mode='w+',shape=(len(rows),24),dtype=np.float32)
        pos=0
        for count,sid in enumerate(sorted(validids),1):
            ix=data.id_to_index[int(sid)];a,b=data.series['online_offsets'][ix:ix+2]
            state=GraphState(g,values['H_states'][ix]);out[pos:pos+b-a]=state.replay(values['tabular'][a:b,101:],values['states'][a:b]);pos+=b-a
            if count%400==0:checkpoint('graph apply',f'fold{fold} {tag} {count}/{len(validids)}')
        assert pos==len(rows) and np.isfinite(out).all();out.flush()
        save(str((d/'DONE.json').relative_to(BASE)),{'status':'COMPLETE','training_ids_sha256':ids_hash(trainids),
            'validation_ids_sha256':ids_hash(validids),'source_sha256':digest(ROOT/'src/next6/graph.py'),
            'graph_sha256':digest(d/'graph.joblib'),'features_sha256':digest(d/'features.npy'),
            'train_valid_group_disjoint':True,'edge_count':len(g.selected),'validation_label_used':False})
        return g,rows,out

def join_graph_parts(target_rows,parts):
    out=np.empty((len(target_rows),24),dtype=np.float32);seen=np.zeros(len(target_rows),dtype=bool)
    for rows,features,ids,data in parts:
        wanted=rows_for(data,ids);location=np.searchsorted(rows,wanted)
        if not np.array_equal(rows[location],wanted):raise RuntimeError('Graph cache row coverage missing')
        at=np.searchsorted(target_rows,wanted)
        if not np.array_equal(target_rows[at],wanted) or np.any(seen[at]):raise RuntimeError('Head crossfit coordinate error')
        out[at]=features[location];seen[at]=True
    assert seen.all();return out

def head_job(data,metadata,values,fold,stage,method,trainrows,validrows,trainextra,validextra,graph,score_allowed):
    directory=BASE/f'fold_{fold}/{stage}/{method}';directory.mkdir(parents=True,exist_ok=True)
    done=directory/'DONE.json'
    if done.exists():
        record=read_json(done)
        if record['training_ids_sha256']!=ids_hash(np.unique(metadata['dataset_id'][trainrows])):raise RuntimeError('Head training IDs changed')
        if record['valid_rows_sha256']!=__import__('hashlib').sha256(validrows.tobytes()).hexdigest():raise RuntimeError('Head valid rows changed')
        return record
    extra=EXTRA_COLUMNS[method];nbase=101 if method=='BASE_ONLY' else 245
    with activity(f'head_design_{fold}_{stage}_{method}','MODEL_RESEARCH'):
        x=np.empty((len(trainrows),nbase+len(extra)),dtype=np.float32)
        # Copy in chunks to limit transient advanced-index copies.
        for a in range(0,len(trainrows),50000):x[a:a+50000,:nbase]=values['tabular'][trainrows[a:a+50000],:nbase]
        if extra:x[:,nbase:]=trainextra[:,extra]
        model=fit_head(x,metadata['target'][trainrows],metadata['time_online'][trainrows],metadata['dataset_id'][trainrows],
            data.guard,method,graph,np.unique(metadata['dataset_id'][validrows]))
        del x;gc.collect()
        model.save(directory/'model.joblib')
        prediction=np.empty(len(validrows),dtype=np.float64)
        for a in range(0,len(validrows),50000):
            r=validrows[a:a+50000];x=np.empty((len(r),nbase+len(extra)),dtype=np.float32);x[:,:nbase]=values['tabular'][r,:nbase]
            if extra:x[:,nbase:]=validextra[a:a+50000,extra]
            prediction[a:a+len(r)]=model.predict(x)
        if not np.isfinite(prediction).all():raise RuntimeError('Nonfinite head predictions')
        np.savez(directory/'predictions.npz',global_row=validrows,dataset_id=metadata['dataset_id'][validrows],prediction=prediction)
        record={'status':'COMPLETE','method':method,'stage':stage,'fold':fold,'created_utc':utc_now(),
            'training_ids_sha256':ids_hash(np.unique(metadata['dataset_id'][trainrows])),'training_series':len(np.unique(metadata['dataset_id'][trainrows])),
            'valid_rows_sha256':__import__('hashlib').sha256(validrows.tobytes()).hexdigest(),
            'model_sha256':digest(directory/'model.joblib'),'prediction_sha256':digest(directory/'predictions.npz'),
            'head_crossfit':bool(extra),'outer_score_released':False,'TS_AUC':float(ts_auc(metadata['target'][validrows],prediction,metadata['time_online'][validrows])) if score_allowed else None}
        save(str(done.relative_to(BASE)),record);checkpoint('head complete',f'fold{fold} {stage} {method}; '+('inner score='+str(record['TS_AUC']) if score_allowed else 'outer score deferred'))
        return record

def inner_stage(data,metadata,values,fold,protocol):
    assignments=protocol['inner_mapping'][str(fold)];sets=[{int(s) for s,k in assignments.items() if k==j} for j in range(3)]
    train=set.union(*sets);singles=[];pairs=[]
    for j in range(3):singles.append(graph_job(data,metadata,values,fold,'single_'+str(j),sets[j],train-sets[j]))
    for j in range(3):pairs.append(graph_job(data,metadata,values,fold,'pair_'+str(j),train-sets[j],sets[j]))
    results=[]
    for held in range(3):
        development=train-sets[held];tr=rows_for(data,development);va=rows_for(data,sets[held]);parts=[]
        for k in range(3):
            if k==held:continue
            donor=next(j for j in range(3) if j not in (held,k));g,rows,features=singles[donor]
            parts.append((rows,features,sets[k],data))
        cross=join_graph_parts(tr,parts);graph,vrows,vfeatures=pairs[held];assert np.array_equal(vrows,va)
        for method in ('M1','M2','M3','M4'):
            results.append(head_job(data,metadata,values,fold,'inner_'+str(held),method,tr,va,cross,vfeatures,graph if method!='M1' else None,True))
        del cross;gc.collect()
    means={m:float(np.mean([r['TS_AUC'] for r in results if r['method']==m])) for m in ('M1','M2','M3','M4')}
    m4=means['M4']>=max(means['M2'],means['M3'])+.0002
    path=f'fold_{fold}/INNER_FREEZE.json'
    if not (BASE/path).exists():save(path,{'status':'FROZEN_BEFORE_OUTER_SCORE','inner_scores':means,'M4_eligible':m4,
        'training_ids':sorted(train),'all_head_training_graph_features_crossfit':True,
        'inner_validation_labels_excluded_from_graph_and_head_training':True,
        'protocol_sha256':digest(BASE/'NEXT6_PROTOCOL_LOCK.json'),'procedure':'M1/M2/M3 always; M4 only if own fold inner gate passed'})

def outer_stage(data,metadata,values,fold,protocol):
    # All inner decisions are sealed before any outer learner is scored.
    for f in range(5):assert (BASE/f'fold_{f}/INNER_FREEZE.json').exists(),'Complete inner freezes for all folds first'
    assignments=protocol['inner_mapping'][str(fold)];sets=[{int(s) for s,k in assignments.items() if k==j} for j in range(3)]
    train=set.union(*sets);valid=set(data.ids)-train;tr=rows_for(data,train);va=rows_for(data,valid)
    pairs=[graph_job(data,metadata,values,fold,'pair_'+str(j),train-sets[j],sets[j]) for j in range(3)]
    cross=join_graph_parts(tr,[(r,x,sets[j],data) for j,(g,r,x) in enumerate(pairs)])
    graph,vrows,vfeatures=graph_job(data,metadata,values,fold,'outer_full',train,valid)
    assert np.array_equal(vrows,va)
    methods=['BASE_ONLY','M1','M2','M3']
    if read_json(BASE/f'fold_{fold}/INNER_FREEZE.json')['M4_eligible']:methods.append('M4')
    for method in methods:head_job(data,metadata,values,fold,'outer',method,tr,va,cross,vfeatures,graph if method not in ('M1','BASE_ONLY') else None,False)
    del cross;gc.collect()
    for kind,method in [('own','OWN'),('random','RANDOM')]:
        controls=[]
        for j in range(3):
            pg,_,_=pairs[j];controls.append(graph_job(data,metadata,values,fold,kind+'_pair_'+str(j),train-sets[j],sets[j],kind,pg.selected))
        cross=join_graph_parts(tr,[(r,x,sets[j],data) for j,(g,r,x) in enumerate(controls)])
        g,r,x=graph_job(data,metadata,values,fold,kind+'_outer_full',train,valid,kind,graph.selected)
        head_job(data,metadata,values,fold,'outer',method,tr,va,cross,x,g,False)
        del cross,controls;gc.collect()
    save(f'fold_{fold}/OUTER_PREDICTIONS_FROZEN.json',{'status':'PREDICTIONS_FROZEN_NOT_SCORED','models':methods+['OWN','RANDOM'],
        'training_ids_sha256':ids_hash(train),'validation_ids_sha256':ids_hash(valid),'timestamp':utc_now()})

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['inner','outer'],required=True);p.add_argument('--fold',type=int,choices=range(5));args=p.parse_args()
    assert read_json(BASE/'PREFLIGHT.json')['status']=='PASS'
    import xml.etree.ElementTree as ET
    t=ET.parse(BASE/'GRAPH_TESTS_REPAIRED.xml');assert not t.findall('.//failure') and not t.findall('.//error')
    sources={n:digest(ROOT/n) for n in ('src/next6/nodes.py','src/next6/graph.py','src/next6/model.py','scripts/run_next6_nested.py')}
    lock=BASE/'IMPLEMENTATION_LOCK.json'
    if not lock.exists():save('IMPLEMENTATION_LOCK.json',{'created_utc':utc_now(),'source_sha256':sources})
    else:assert read_json(lock)['source_sha256']==sources,'Implementation changed after fit began; requires explicit BUG repair receipt'
    data,metadata,values=arrays();assert np.all(np.diff(data.ids)>0),'Sorted source series required';protocol=read_json(BASE/'NEXT6_PROTOCOL_LOCK.json')
    for fold in ([args.fold] if args.fold is not None else range(5)):
        finished=BASE/f'fold_{fold}'/('INNER_FREEZE.json' if args.stage=='inner' else 'OUTER_PREDICTIONS_FROZEN.json')
        if finished.exists():checkpoint('reuse complete nested stage',f'{args.stage} fold{fold}');continue
        (inner_stage if args.stage=='inner' else outer_stage)(data,metadata,values,fold,protocol)
    data.guard.check_state()

if __name__=='__main__':main()
