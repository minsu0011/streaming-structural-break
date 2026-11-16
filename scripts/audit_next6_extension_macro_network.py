"""Uncertainty, held-out transport, and state-memory sufficiency diagnostics."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from src.next6_extension.common import *
from src.next6.data import source
from src.next6_extension.fresh_state import STATES

def main():
    save('MACRO_NETWORK_AUDIT_LOCK.json',{'created_utc':utc_now(),'source_sha256':digest(Path(__file__)),'bootstrap_replicates':500,
        'uncertainty':'Whole original group bootstrap of each original outer-training table. No topology or display threshold changes. Overlapping folds not independent.',
        'transport':'Original heldout empirical transition distribution under each official phase, descriptive calibration only. Labels not supplied to runtime.',
        'memory':'Report exit frequencies stratified by block clock and previous-state dwell age. No new conditioned predictor or candidate fitted.',
        'no_outer_score_tuning':True})
    data=source();meta=data.rows();counts=np.load(BASE/'rp_replay/series_transition_counts.npy');support=np.load(BASE/'rp_replay/series_transition_support.npy');models=np.load(BASE/'rp_replay/macro_probabilities.npy');folds=data.series['fold'];out=[];transport=[]
    with activity('learned_macro_network_group_uncertainty','STATE_NETWORK_AUDIT'):
        for fold in range(5):
            tr=folds!=fold;va=~tr;ids=data.ids[tr];g=np.array([data.guard.split.groups[int(s)] for s in ids]);unique,code=np.unique(g,return_inverse=True);groupcounts=np.zeros((len(unique),2,7,7));np.add.at(groupcounts,code,counts[tr]);ng=len(unique)
            rng=np.random.default_rng([2026091309,fold]);weight=rng.multinomial(ng,np.full(ng,1/ng),size=500);rep=(weight@groupcounts.reshape(ng,-1)).reshape(500,2,7,7);prob=(rep+.5)/(rep+.5).sum(axis=-1,keepdims=True)
            v=counts[tr].sum(axis=0);observed=counts[va].sum(axis=0);ss=support[tr].sum(axis=0);heldsupport=support[va].sum(axis=0)
            np.testing.assert_allclose((v+.5)/(v+.5).sum(axis=-1,keepdims=True),models[fold],rtol=1e-12,atol=1e-12)
            jump=rep.copy()
            for s in range(7):jump[:,:,s,s]=0
            jumpmass=jump.sum(axis=-1,keepdims=True);jump=np.divide(jump,jumpmass,out=np.full_like(jump,np.nan),where=jumpmass>0)
            for phase in (0,1):
                for previous in range(7):
                    source_support=int(np.any(support[tr,phase,previous],axis=1).sum());held_source=int(np.any(support[va,phase,previous],axis=1).sum());trainmass=v[phase,previous].sum();heldmass=observed[phase,previous].sum()
                    if heldmass>0:
                        observedprob=observed[phase,previous]/heldmass;tv=float(.5*np.abs(observedprob-models[fold,phase,previous]).sum());transport.append({'fold':fold,'phase':phase,'previous':previous,'previous_name':STATES[previous],'train_source_series':source_support,'heldout_source_series':held_source,'heldout_weighted_mass':heldmass,'total_variation_to_frozen_train_probability':tv,'descriptive_only':True})
                    for current in range(7):
                        interval=np.quantile(prob[:,phase,previous,current],[.025,.975]) if source_support else [None,None];j=jump[:,phase,previous,current];j=j[np.isfinite(j)];ji=np.quantile(j,[.025,.975]) if len(j) else [None,None]
                        leave=v[phase,previous].copy();leave[previous]=0
                        out.append({'fold':fold,'phase':phase,'previous':previous,'current':current,'previous_name':STATES[previous],'current_name':STATES[current],'frozen_probability':float(models[fold,phase,previous,current]),'bootstrap_lo':interval[0],'bootstrap_hi':interval[1],'training_edge_series':int(ss[phase,previous,current]),'training_source_series':source_support,'training_groups':ng,'jump_probability':float(leave[current]/leave.sum()) if leave.sum()>0 else None,'jump_bootstrap_lo':ji[0],'jump_bootstrap_hi':ji[1],'bootstrap_with_observed_exit':len(j),'heldout_empirical_probability':float(observed[phase,previous,current]/heldmass) if heldmass>0 else None,'heldout_edge_series':int(heldsupport[phase,previous,current]),'heldout_source_series':held_source,'pseudocount_only_row':source_support==0})
            checkpoint('macro network uncertainty',f'fold {fold}:500 group resamples; heldout transition transport')
    csv_write('MACRO_TRANSITION_UNCERTAINTY.csv',out);csv_write('MACRO_TRANSITION_HELDOUT_TRANSPORT.csv',transport)
    states=np.load(BASE/'rp_replay/states.npy',mmap_mode='r');clock=np.zeros((6400,2,7,16,2));dwell=np.zeros((6400,2,7,5,2));clock_support=np.zeros_like(clock,dtype=bool);dwell_support=np.zeros_like(dwell,dtype=bool)
    with activity('observable_state_hidden_memory_diagnostic','STATE_NETWORK_AUDIT'):
        for i,sid in enumerate(data.ids):
            a,b=data.series['online_offsets'][i:i+2];s=states[a:b];y=meta['target'][a:b];t=np.arange(1,len(s));previous=s[:-1];leave=(s[1:]!=previous).astype(int);start=np.maximum.accumulate(np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1][np.cumsum(np.r_[0,s[1:]!=s[:-1]])]) if False else None
            boundaries=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1];start_at=boundaries[np.cumsum(np.r_[0,s[1:]!=s[:-1]])];age=np.arange(len(s))-start_at+1;binid=np.searchsorted([2,16,32,64],age[:-1],side='right')
            for phase in (0,1):
                use=y[1:]==phase;n=int(use.sum())
                if not n:continue
                c=np.zeros((7,16,2));d=np.zeros((7,5,2));np.add.at(c,(previous[use],t[use]%16,leave[use]),1);np.add.at(d,(previous[use],binid[use],leave[use]),1);clock[i,phase]=c*64/n;dwell[i,phase]=d*64/n;clock_support[i,phase]=c>0;dwell_support[i,phase]=d>0
        rows=[]
        for partition,mask in [('outer_train_fold0',folds!=0),('outer_heldout_fold0',folds==0)]:
            for kind,values,ss,labels in [('clock',clock,clock_support,[str(t) for t in range(16)]),('dwell',dwell,dwell_support,['1','2-15','16-31','32-63','64+'])]:
                v=values[mask].sum(axis=0);supportsum=np.any(ss[mask],axis=-1).sum(axis=0)
                for phase in (0,1):
                    for previous in range(7):
                        for k,label in enumerate(labels):
                            den=v[phase,previous,k].sum();rows.append({'partition':partition,'official_phase':phase,'previous_state':STATES[previous],'stratifier':kind,'stratum':label,'exit_probability':float(v[phase,previous,k,1]/den) if den else None,'weighted_opportunities':float(den),'unique_series':int(supportsum[phase,previous,k]),'block_completes_at_clock':15,'not_a_fitted_predictor':True})
        csv_write('MACRO_STATE_CLOCK_AND_DWELL_DIAGNOSTIC.csv',rows)
    save('MACRO_NETWORK_AUDIT_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'uncertainty_rows':len(out),'bootstrap_replicates_per_fold':500,'group_unit_verified':True,'all_RP_groups':len(set(data.guard.split.groups[int(s)] for s in data.ids)),
        'interpretation':'Seven displayed states are an observable summary, not a complete first-order Markov state. The runtime also keeps block clock, four-block evidence memory, shock cooldown and recovery counter. These diagnostics do not change any table.'})
    data.guard.check_state()
if __name__=='__main__':main()
