"""One fixed RP diagnostic replay; fit descriptive macro transitions on outer train."""
from pathlib import Path
import sys,os,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.fresh_state import State,categorize,STATES,COLUMNS
from src.next6.data import source
from src.scoring.ts_auc import ts_auc
from src.next5.bootstrap import PreparedAUC

def main():
    lock=BASE/'FRESH_RP_EVALUATION_LOCK.json'
    if not lock.exists():save(lock.name,{'created_utc':utc_now(),'source_sha256':digest(Path(__file__)),'threshold_receipt_sha256':digest(BASE/'calibration/fresh/DONE.json'),
        'score':'Raw persistent evidence, sign fixed, no tuning; diagnostic TS-AUC not a learned classifier or promotion candidate.',
        'case_selection':'fold0 sha256 NEXT6_EXTENSION_CASE:ID; first success(delay<=127), delayed(>=128), null_false_alarm, miss; no prealarm for successes.',
        'macro_tables':'Outer TRAIN descriptive P(current observable state | previous state,current official binary label). Equal series/phase total64 counts, Dirichlet.5. Both conditions available, never true-label routing.',
        'macro_visualization':'Jump-chain probabilities conditional on leaving a state; separately show ordinary self-transition probability, unique-series support; skip drawing cross links with fewer8 training series.',
        'adaptive_research_only':True,'Shadow_Seal_opened':False})
    assert read_json(lock)['source_sha256']==digest(Path(__file__))
    data=source();meta=data.rows();threshold=read_json(BASE/'calibration/fresh/DONE.json')['thresholds']['FRESH'];frozen=read_json(BASE/'FRESH_GRAPHS_FROZEN.json')
    cache=BASE/'rp_replay';cache.mkdir(exist_ok=True);n=len(meta['target']);score=np.lib.format.open_memmap(cache/'evidence.npy',mode='w+',dtype=np.float64,shape=(n,));states=np.lib.format.open_memmap(cache/'states.npy',mode='w+',dtype=np.int8,shape=(n,))
    per=[];counts=np.zeros((6400,2,7,7));support=np.zeros_like(counts,dtype=bool);phase_stats=[];case_candidates=[]
    block=np.load(BASE/'block_cache/node_z.npy',mmap_mode='r');blockids=np.load(BASE/'block_cache/dataset_id.npy',mmap_mode='r');blockoffsets=np.r_[0,np.cumsum([np.sum(blockids==sid) for sid in data.ids])]
    for fold in range(5):
        assert digest(BASE/f'fresh_graphs/fold_{fold}/graph.joblib')==frozen['graphs'][str(fold)]
        graph=joblib.load(BASE/f'fresh_graphs/fold_{fold}/graph.joblib')
        with activity(f'fixed_fresh_state_RP_replay_fold{fold}','EVALUATION'):
            for sid in data.ids[data.series['fold']==fold]:
                i=data.id_to_index[int(sid)];h,o,tau=data.get(int(sid));a,b=data.series['online_offsets'][i:i+2];out=State(h,threshold=threshold,graph=graph).replay(o)
                score[a:b]=out[:,1];states[a:b]=out[:,0];end=np.flatnonzero(out[:,4]);np.testing.assert_allclose(out[end,12:20],block[blockoffsets[i]:blockoffsets[i+1]],atol=5e-7,rtol=1e-7)
                y=meta['target'][a:b];s=out[:,0].astype(int)
                for phase in (0,1):
                    good=y[1:]==phase;cnt=int(good.sum())
                    if cnt:
                        raw=np.zeros((7,7));np.add.at(raw,(s[:-1][good],s[1:][good]),1);counts[i,phase]=raw*64/cnt;support[i,phase]=raw>0
                rel=np.arange(len(o))-tau
                masks={'no_break':np.ones(len(o),dtype=bool) if tau<0 else np.zeros(len(o),dtype=bool)}
                if tau>=0:
                    for name,low,high in [('pre',-32,-1),('onset',0,7),('earlyPost',8,31),('establishedPost',32,127),('laterPost',128,999999)]:masks[name]=(rel>=low)&(rel<=high)
                for phase,mask in masks.items():
                    if mask.any():phase_stats.append({'dataset_id':int(sid),'fold':fold,'phase':phase,'observed_points':int(mask.sum()),'state_fraction':[float(np.mean(s[mask]==j)) for j in range(7)],'mean_persistent_evidence':float(out[mask,1].mean())})
                alarm=np.flatnonzero(out[:,1]>threshold);first=int(alarm[0]) if len(alarm) else -1;category=None
                if tau<0 and first>=0:category='null_false_alarm'
                elif tau>=0 and first<0:category='miss'
                elif tau>=0 and first>=tau:category='success' if first-tau<=127 else 'delayed'
                per.append({'dataset_id':int(sid),'fold':fold,'tau':tau,'online_length_evaluation_only':len(o),'first_alarm':first,'prealarm':tau>=0 and 0<=first<tau,'category':category,'persistent_fraction':float(np.mean(s==4))})
                if fold==0 and category:case_candidates.append((hashlib.sha256(('NEXT6_EXTENSION_CASE:'+str(sid)).encode()).hexdigest(),int(sid),category))
    score.flush();states.flush();np.save(cache/'series_transition_counts.npy',counts,allow_pickle=False);np.save(cache/'series_transition_support.npy',support,allow_pickle=False)
    csv_write('FRESH_RP_SERIES_DIAGNOSTIC.csv',per);save('FRESH_RP_PHASE_SERIES.json',{'rows':phase_stats,'states':STATES,'phase_labels_evaluation_only':True})
    models=[];probrows=[]
    with activity('outer_training_macro_transition_tables','STATE_TRANSITION_DESCRIPTION'):
        for fold in range(5):
            tr=data.series['fold']!=fold;v=counts[tr].sum(axis=0);ss=support[tr].sum(axis=0);prob=(v+.5)/(v+.5).sum(axis=-1,keepdims=True);models.append(prob)
            for phase in (0,1):
                for previous in range(7):
                    leaving=v[phase,previous].copy();leaving[previous]=0;mass=leaving.sum()
                    for current in range(7):probrows.append({'fold':fold,'phase':phase,'previous':previous,'current':current,'previous_name':STATES[previous],'current_name':STATES[current],
                        'ordinary_probability_smoothed':float(prob[phase,previous,current]),'jump_probability_unsmoothed':float(leaving[current]/mass) if mass else None,'weighted_count':float(v[phase,previous,current]),
                        'unique_training_series':int(ss[phase,previous,current]),'source_state_training_series':int(np.any(support[tr,phase,previous],axis=1).sum()),'self_transition':current==previous,'descriptive_only':True})
        np.save(cache/'macro_probabilities.npy',models,allow_pickle=False);csv_write('MACRO_STATE_TRANSITIONS.csv',probrows)
    scores=[];draws=[]
    with activity('fixed_persistent_evidence_discrimination','EVALUATION'):
        for fold in range(5):
            use=meta['fold']==fold;_,codes=np.unique(meta['dataset_id'][use],return_inverse=True);nseries=len(np.unique(codes));rng=np.random.default_rng(2026091304+fold);w=np.vstack((np.ones((1,nseries),dtype=np.int32),rng.multinomial(nseries,np.full(nseries,1/nseries),size=1000)))
            value=float(ts_auc(meta['target'][use],score[use],meta['time_online'][use]));z=PreparedAUC.create(meta['target'][use],score[use],meta['time_online'][use],codes,nseries).evaluate(w);draws.append(z);lo,hi=np.quantile(z[1:],[.025,.975])
            scores.append({'fold':fold,'diagnostic_TS_AUC':value,'ci_2_5':lo,'ci_97_5':hi,'not_a_promotable_candidate':True})
        csv_write('FRESH_RP_DIAGNOSTIC_AUC.csv',scores);zz=np.mean(draws,axis=0)
    examples=[]
    for category in ('success','delayed','null_false_alarm','miss'):
        choices=sorted([c for c in case_candidates if c[2]==category])
        if not choices:examples.append({'category':category,'status':'NO_CASE'});continue
        sid=choices[0][1];h,o,tau=data.get(sid);graph=joblib.load(BASE/'fresh_graphs/fold_0/graph.joblib');out=State(h,threshold=threshold,graph=graph).replay(o)
        examples.append({'category':category,'dataset_id':sid,'tau':tau,'columns':COLUMNS,'curve':out.tolist(),'status':'SELECTED_BY_FROZEN_HASH_RULE'})
    save('FRESH_RP_CASES.json',{'threshold':threshold,'cases':examples,'tau_is_plot_annotation_only':True})
    save('FRESH_RP_EVALUATION.json',{'status':'COMPLETE','created_utc':utc_now(),'series':6400,'points':len(score),'block_cache_replay_parity':'PASS float32 storage tolerance','mean_diagnostic_TS_AUC':float(zz[0]),'conditional_CI':np.quantile(zz[1:],[.025,.975]).tolist(),
        'selected_discriminative_cross_edges':[len(joblib.load(BASE/f'fresh_graphs/fold_{f}/graph.joblib').selected) for f in range(5)],'source_lock_sha256':digest(BASE/'REFINEMENT_IMPLEMENTATION_LOCK.json'),
        'no_new_detector_fit':True,'no_graph_reselection_after_outer_evaluation':True,'not_calibrated_break_probability':True,'canonical_retained':True})
    data.guard.check_state()
if __name__=='__main__':main()
