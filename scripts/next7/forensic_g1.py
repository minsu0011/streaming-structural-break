"""Exact RP latch/pair/time diagnostics; no model fitting or candidate tuning."""
from pathlib import Path
import csv
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next7/numba_cache')
import numpy as np
from scipy.stats import spearmanr
from src.next7.common import BASE,save,read_json,digest,csv_write,utc_now,activity,require_guard,note
from src.next7.attribution import latch_pair_partition
from src.next7.model import Correction
from src.next7.features import NAMES
from src.next6.data import source


def main():
    guard=require_guard()
    dest=BASE/'forensic_g1'
    if (dest/'DONE.json').exists():
        raise RuntimeError('Preserve completed forensic results')
    data=source()
    meta=data.rows()
    state=np.load(BASE/'cache/g1/state.npy',mmap_mode='r',allow_pickle=False)
    scores=read_json(BASE/'G1_OUTER_SCORES.json')['scores']
    with activity('G1 target alignment and exact latch rank-pair forensic','BOTTLENECK_FORENSIC','F3_CONFIRMED_EVENT_MEMORY'):
        series=[]
        events=[]
        for j,sid in enumerate(data.ids):
            a,b=data.series['online_offsets'][j:j+2]
            v=state[a:b]
            tau=int(data.series['tau'][j])
            first=int(v[-1,21])
            ever=first>=0
            category='no_break' if tau<0 else 'break_observed' if tau<len(v) else 'break_unobserved'
            record={'dataset_id':int(sid),'fold':guard.split.assignment(int(sid))[1],
                'online_length_offline_only':len(v),'tau_offline_only':tau,'category':category,
                'first_latch':first,'latched':ever,'latched_before_break':ever and tau>=0 and first<tau,
                'latched_after_break':ever and tau>=0 and first>=tau,
                'first_shock':int(v[-1,19]),'first_confirmation':int(v[-1,20]),
                'event_strength_final':float(v[-1,3]),'event_persists_while_current_normal':bool(np.any((v[:,0]==1)&(v[:,3]>0))),
                'current_confirmed_points':int((v[:,0]==4).sum()),'event_positive_points':int((v[:,3]>0).sum()),
                'current_score_mean':float(v[:,1].mean()),'event_score_mean':float(v[:,3].mean()),
                'scale_current_mean':float(v[:,10].mean())}
            series.append(record)
            on=v[:,17]>.5
            for t in np.flatnonzero(on&~np.r_[False,on[:-1]]):
                oldest=int(v[t,23])
                events.append({'dataset_id':int(sid),'fold':record['fold'],'event_time':int(t),
                    'firstShockTime':int(v[t,19]),'firstConfirmationTime':int(v[t,20]),
                    'firstLatchTime':int(v[t,21]),'confirmationChannel':int(v[t,22]),
                    'confirmationBlocks':','.join(map(str,range(oldest,oldest+4))),
                    'confirmationStrength':float(v[t,2]),'eventDirection':int(v[t,8]),'eventAge':int(v[t,5]),
                    'tau_offline_only':tau,'target_at_event':int(meta['target'][a+t])})
        pairs=[]
        correlations=[]
        coefficients=[]
        for fold in range(5):
            directory=BASE/f'g1/fold_{fold}'
            with np.load(directory/'fixed_predictions.npz',allow_pickle=False) as f:
                rows=f['global_row'].copy()
                baseline=f['M0'].copy()
                candidate=f['M_EVENT'].copy()
                methods={k:f[k].copy() for k in f.files if k not in ('global_row','dataset_id','M0','AMPLITUDE_0.9_M0')}
            y=meta['target'][rows]
            t=meta['time_online'][rows]
            latch=state[rows,3]>0
            original,mass=latch_pair_partition(y,baseline,t,latch)
            updated,check=latch_pair_partition(y,candidate,t,latch)
            np.testing.assert_array_equal(mass,check)
            total=mass.sum()
            assert abs((updated-original).sum()/total-(scores['M_EVENT'][str(fold)]-scores['M0'][str(fold)]))<1e-12
            assert updated[0,0]==original[0,0]
            for pos in range(2):
                for neg in range(2):
                    pairs.append({'fold':fold,'positive_latched':pos,'negative_latched':neg,
                        'pairs':int(mass[pos,neg]),'fraction_all_pairs':float(mass[pos,neg]/total),
                        'baseline_wins':float(original[pos,neg]),'candidate_wins':float(updated[pos,neg]),
                        'baseline_AUC_in_cell':float(original[pos,neg]/mass[pos,neg]) if mass[pos,neg] else None,
                        'candidate_AUC_in_cell':float(updated[pos,neg]/mass[pos,neg]) if mass[pos,neg] else None,
                        'delta_contribution_to_fold_TS_AUC':float((updated[pos,neg]-original[pos,neg])/total),
                        'posthoc_pair_partition_not_causal_effect':True})
            for method in ('I3_CURRENT_EVENT','M_FULL','I4_EVENT_SCALE_TREE'):
                with np.load(directory/method/'predictions.npz',allow_pickle=False) as f:
                    np.testing.assert_array_equal(rows,f['global_row'])
                    methods[method]=f['prediction'].copy()
                model=Correction.load(directory/method/'model.joblib')
                if model.beta is not None:
                    for k,index in enumerate(model.columns):
                        coefficients.append({'fold':fold,'method':method,'feature':NAMES[index],
                            'standardized_coefficient':float(model.beta[k]),'training_center':float(model.center[k]),
                            'training_spread':float(model.spread[k]),'baseline_logit_offset_coefficient':1})
            order=np.argsort(t,kind='stable')
            ends=np.r_[0,np.flatnonzero(np.diff(t[order]))+1,len(t)]
            for method,p in methods.items():
                numerator=denominator=0.
                time_count=0
                for a,b in zip(ends[:-1],ends[1:]):
                    r=order[a:b]
                    npos=int(y[r].sum())
                    m=npos*(len(r)-npos)
                    if m and np.ptp(baseline[r])>0 and np.ptp(p[r])>0:
                        value=float(spearmanr(baseline[r],p[r]).statistic)
                        numerator+=m*value;denominator+=m;time_count+=1
                corr=float(np.corrcoef(baseline,p)[0,1]) if np.ptp(p)>0 else None
                correlations.append({'fold':fold,'method':method,'Pearson':corr,
                    'Spearman':float(spearmanr(baseline,p).statistic),
                    'within_time_Spearman_pair_weighted':numerator/denominator if denominator else None,
                    'within_time_supported_pair_mass':int(denominator),'within_time_supported_times':time_count,
                    'mean_absolute_correction':float(np.mean(np.abs(p-baseline))),
                    'fold_delta_TS_AUC':scores[method][str(fold)]-scores['M0'][str(fold)],
                    'no_independent_signal_claim_under_policy':corr is not None and corr>.995 and scores[method][str(fold)]-scores['M0'][str(fold)]<.0005})
            print(utc_now(),'G1 FORENSIC',fold,flush=True)
        csv_write('forensic_g1/SERIES_LATCH_AND_RECOVERY.csv',series)
        csv_write('EVENT_TIMING_FORENSIC_G1.csv',events)
        csv_write('forensic_g1/EXACT_LATCH_PAIR_DECOMPOSITION.csv',pairs)
        csv_write('forensic_g1/BL80_CORRELATION_AND_REDUNDANCY.csv',correlations)
        csv_write('forensic_g1/CORRECTION_COEFFICIENTS.csv',coefficients)
        summary=[]
        for category in sorted({r['category'] for r in series}):
            subset=[r for r in series if r['category']==category]
            row={'category':category,'series':len(subset)}
            for key in ('latched','latched_before_break','latched_after_break','event_persists_while_current_normal',
                'current_confirmed_points','event_positive_points','current_score_mean','event_score_mean'):
                row[key+'_mean']=float(np.mean([r[key] for r in subset]))
            summary.append(row)
        csv_write('TARGET_ALIGNMENT_RESULTS_G1.csv',summary)
        total_contribution=[]
        for pos in range(2):
            for neg in range(2):
                group=[r for r in pairs if r['positive_latched']==pos and r['negative_latched']==neg]
                total_contribution.append({'positive_latched':pos,'negative_latched':neg,
                    'mean_fold_delta_contribution':float(np.mean([r['delta_contribution_to_fold_TS_AUC'] for r in group])),
                    'mean_pair_share':float(np.mean([r['fraction_all_pairs'] for r in group]))})
        save('forensic_g1/DONE.json',{'status':'COMPLETE','created_utc':utc_now(),
            'source_sha256':digest(__file__),'series':len(series),'latch_episodes':len(events),
            'pair_decomposition_matches_exact_G1_delta':True,'mean_pair_contributions':total_contribution,
            'target_alignment':summary,'new_fits':0,'candidate_definitions_changed':False})
    guard.check_state()
    note('G1 event failure forensic complete','Exact pair losses and recovery/persistence are quantified without retuning','F4_NULL_GROWTH_CORRECTION')


if __name__=='__main__':
    main()
