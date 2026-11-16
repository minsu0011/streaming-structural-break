"""Frozen edge evaluation on each unseen outer fold; labels never select edges."""
from pathlib import Path
import sys,os,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from src.next6.common import BASE,read_json,save,activity,ids_hash,digest
from src.next6.data import arrays
from src.next6.graph import pack_sample,design,weights,ridge,owncols,state_contexts
from src.next6.evaluation import csv_write

def bh(p):
    p=np.asarray(p);order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty_like(q);out[order]=np.minimum(q,1);return out

def main():
    data,meta,v=arrays();provenance=[];edges=[];mechanical=[];results=[];selections=[];transitions=[]
    for f in range(5):
        directory=BASE/f'fold_{f}/graphs/outer_full';g=joblib.load(directory/'graph.joblib');tr=set(g.provenance['training_ids']);va=set(data.ids)-tr;data.guard.partition(tr,va)
        selections.extend([{'fold':f,**e} for e in g.selected]);provenance.append({'fold':f,'training_ids_sha256':ids_hash(tr),'heldout_ids_sha256':ids_hash(va),
            'graph_sha256':digest(directory/'graph.joblib'),'edge_count':len(g.selected),'no_outer_validation_selection':True})
        with (directory/'EDGE_DISCOVERY.csv').open(encoding='utf-8-sig',newline='') as file:discovery=list(csv.DictReader(file))
        q=bh([float(r['p_diagnostic'] or 1) for r in discovery])
        for r,z in zip(discovery,q):edges.append({'fold':f,**r,'BH_q_diagnostic':float(z),'multiplicity_scope':'4800 fixed phase/half/edge tests per outer training fold; descriptive only'})
        for e in g.selected:mechanical.append({'fold':f,**e,'shared_raw_input':True,'economic_causality_established':False,'filter_half_lives':'5,8,32; see lineage','null_check':'RAW_NULL_SELECTED_EDGES.csv'})
        with activity('outer_edge_stability_fold'+str(f),'RELATION_EVALUATION'),threadpool_limits(limits=2):
            train=pack_sample(data,meta,v,tr)
            # Identical subsampling for evaluation, explicitly separate from training admission.
            data.guard.admit(va,purpose='NEXT6 heldout edge evaluation only')
            rr=[]
            for sid in sorted(va):
                i=data.id_to_index[int(sid)];a,b=data.series['online_offsets'][i:i+2]
                for phase in (0,1):
                    r=np.arange(a,b,8);rr.extend(r[meta['target'][r]==phase][:128].tolist())
            rr=np.array(rr);xv=design(v['tabular'][rr,101:],meta['time_online'][rr]);yv=np.asarray(v['nodes'][rr]);vid=meta['dataset_id'][rr]
            current,previous,contexts=state_contexts(data,v,rr,g.parents)
            for phase in (0,1):
                a=train['phase']==phase;b=meta['target'][rr]==phase;xt=train['X'][a];yt=train['Y'][a];w=weights(train['ids'][a]);gram=xt.T@(xt*w[:,None]);xy=xt.T@(yt*w[:,None])
                unique,codes,count=np.unique(vid[b],return_inverse=True,return_counts=True);rng=np.random.default_rng(20260913+f*2+phase)
                bw=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=1000)
                for e in g.selected:
                    j=e['target'];own=owncols(j);cols=sorted(set(own+[e['source']+16*e['lag_index']]));bo=ridge(gram,xy[:,j],own);bc=ridge(gram,xy[:,j],cols)
                    eo=yv[b,j]-xv[b][:,own]@bo;ec=yv[b,j]-xv[b][:,cols]@bc
                    mse0=np.bincount(codes,weights=eo*eo)/count;mse1=np.bincount(codes,weights=ec*ec)/count;gain=(mse0-mse1)/np.maximum(mse0,.01);draw=bw@gain/len(gain)
                    lo,hi=np.quantile(draw,[.025,.975]);coef=float(bc[cols.index(e['source']+16*e['lag_index'])])
                    results.append({'fold':f,**e,'evaluation_phase':phase,'heldout_series':len(unique),'heldout_sample_rows':int(b.sum()),'heldout_mean_relative_MSE_gain':float(gain.mean()),
                        'heldout_positive_series_fraction':float((gain>0).mean()),'ci_2_5':float(lo),'ci_97_5':float(hi),'individual_cross_coefficient_train_only':coef,
                        'replicated_descriptive':bool(gain.mean()>=.005 and (gain>0).mean()>=.55 and lo>0),'outer_labels_evaluation_only':True,
                        'statistic':'single frozen selected edge + own five lags/common controls; train-only ridge; no edge reselection'})
                for node in range(16):
                    good=b&(contexts[:,node]>=0)&(current[:,node]<3)&(previous[:,node]<3)
                    if not good.any():continue
                    sid,cod,cnt=np.unique(vid[good],return_inverse=True,return_counts=True)
                    y=current[good,node];context=contexts[good,node];prev=previous[good,node]
                    pp=g.tables[phase,node,context,y];op=g.own_tables[phase,node,prev,y]
                    delta=np.log(pp)-np.log(op);byseries=np.bincount(cod,weights=delta)/cnt
                    boot=rng.multinomial(len(sid),np.full(len(sid),1/len(sid)),size=1000)@byseries/len(sid);lo,hi=np.quantile(boot,[.025,.975])
                    transitions.append({'fold':f,'phase':phase,'node':node,'selected_parents':len(g.parents[node]),'heldout_series':len(sid),'sample_rows':int(good.sum()),
                        'mean_parent_vs_own_log_probability_gain':float(byseries.mean()),'ci_2_5':float(lo),'ci_97_5':float(hi),
                        'positive_series_fraction':float((byseries>0).mean()),'mean_parent_NLL':float(np.mean(-np.log(pp))),
                        'context_fallback_fraction':float(np.mean(g.context_support[phase,node,context]<8)),
                        'higher_log_gain_better':True,'phase_label_used_for_evaluation_only':True})
    frequency={}
    for e in selections:
        key=(e['source'],e['target'],e['lag']);frequency[key]=frequency.get(key,0)+1
    for r in results:r['selected_outer_fold_count']=frequency[(r['source'],r['target'],r['lag'])]
    csv_write('LAGGED_RELATION_RESULTS.csv',edges);csv_write('MECHANICAL_EDGE_AUDIT.csv',mechanical);csv_write('EDGE_STABILITY_RESULTS.csv',results)
    csv_write('STATE_TRANSITION_TABLES/HELDOUT_LOG_SCORE_GAIN.csv',transitions)
    save('GRAPH_SELECTION_PROVENANCE.json',{'status':'PASS','folds':provenance,'all_crossfit_details':'fold_*/graphs/*/TRAIN_PROVENANCE.json',
        'heldout_reproduction_does_not_change_models':True,'unique_selected_edges':len(frequency),'edges_selected_all_5_folds':sum(v==5 for v in frequency.values()),
        'replicated_selected_phase_fold_edges':sum(r['replicated_descriptive'] and r['evaluation_phase']==r['selection_phase'] for r in results),
        'selected_fold_edges':len(selections),'heldout_CIs_conditional_on_selected_training_graph':True})

if __name__=='__main__':main()
