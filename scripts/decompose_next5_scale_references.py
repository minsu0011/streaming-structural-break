"""S3: exact paired ordering decomposition and within-time feature geometry."""
from pathlib import Path
import argparse
import csv
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from numba import njit,prange,set_num_threads
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.data import prepare_research
from src.next5.references import component
from src.next5.cache import build_design
from src.next5.scheduler import register_job
from src.scoring.ts_auc import ts_auc


@njit(cache=False,parallel=True)
def joint_ordering(y,p,cuts):
    result=np.zeros((len(cuts)-1,3,3,3),dtype=np.int64)
    for block in prange(len(cuts)-1):
        start=cuts[block];stop=cuts[block+1]
        for i in range(start,stop):
            if y[i]!=1:continue
            for j in range(start,stop):
                if y[j]!=0:continue
                a=2 if p[i,0]>p[j,0] else 1 if p[i,0]==p[j,0] else 0
                b=2 if p[i,1]>p[j,1] else 1 if p[i,1]==p[j,1] else 0
                c=2 if p[i,2]>p[j,2] else 1 if p[i,2]==p[j,2] else 0
                result[block,a,b,c]+=1
    return result


def time_groups(times):
    order=np.argsort(times,kind='stable');cuts=np.r_[0,np.flatnonzero(np.diff(times[order]))+1,len(order)]
    return order,cuts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    out=BASE/'diagnostics/S3_SCALE_DECOMPOSITION';policy_path=out/'POLICY.json'
    if args.register:
        atomic_json(policy_path,{'created_utc':utc_now(),'status':'FROZEN_BEFORE_DECOMPOSITION','source_sha256':digest(Path(__file__)),
            'new_fits':0,'selection_authority':False,'pair_sampling':False,
            'pair_rule':'Every positive-negative pair at equal online time within its original ResearchPool OOF fold, exact ties worth 0.5.',
            'feature_rule':'Exact B2 last16 and D2 36 feature coordinates; Pearson and Spearman after centering within original ResearchPool fold/time.',
            'interpretation':'Error complementarity is descriptive, not an attainable oracle classifier or new candidate.'},immutable=True)
        register_job({'id':'S3_REFERENCE_INFORMATION','kind':'validation','priority':46,
            'depends_on_success':['REF_SUMMARY_PRIMARY'],'argv':['scripts/decompose_next5_scale_references.py'],
            'description':'Exact B2/D2 pair-error overlap and within-time feature correlations on matched ResearchPool references','retry_safe':False})
        return
    policy=read_json(policy_path);assert policy['source_sha256']==digest(Path(__file__))
    atomic_json(out/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(policy_path),'new_fits':0},immutable=True)
    set_num_threads(2);data=prepare_research();metadata=data.rows();sources=[];components={}
    for kind in ('J102','C205','B2_augmented_J','D2_expert'):
        components[kind],receipt=component(kind,'full','PRIMARY',metadata,data.guard);sources.append(receipt)
    base=.8*components['J102']+.2*components['C205'];b2=.8*components['B2_augmented_J']+.2*components['C205'];d2=.9*base+.1*components['D2_expert']
    probabilities=np.column_stack([base,b2,d2]);joint_rows=[];conditional_rows=[];fold_rows=[];score_correlations=[]
    score_cross=np.zeros((3,3));score_squares=np.zeros(3);all_joint=np.zeros((3,3,3),dtype=np.int64)
    for fold in range(5):
        rows=np.flatnonzero(metadata['fold']==fold);times=metadata['time_online'][rows];target=metadata['target'][rows]
        order,cuts=time_groups(times);p=probabilities[rows][order];y=target[order]
        progress('S3 exact pair ordering','ResearchPool fold '+str(fold)+'; all same-time positive-negative pairs, no sampling')
        per_time=joint_ordering(y,p,cuts);joint=per_time.sum(axis=0);all_joint+=joint;pair_mass=int(joint.sum())
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    joint_rows.append({'fold':fold,'BL80_credit':a/2,'B2_credit':b/2,'D2_credit':c/2,'pairs':int(joint[a,b,c]),'pair_fraction':float(joint[a,b,c]/pair_mass)})
        credits=np.indices((3,3,3),dtype=float)/2.
        values=[float(np.sum(joint*credits[k])/pair_mass) for k in range(3)]
        for k,name in enumerate(('BL80','B2','D2')):
            assert abs(values[k]-ts_auc(target,probabilities[rows,k],times))<1e-14
        gain_b=credits[1]-credits[0];gain_d=credits[2]-credits[0]
        fold_rows.append({'fold':fold,'pairs':pair_mass,'BL80_TS_AUC':values[0],'B2_TS_AUC':values[1],'D2_TS_AUC':values[2],
            'B2_delta':values[1]-values[0],'D2_delta':values[2]-values[0],
            'B2_positive_improvement_mass':float(np.sum(joint*np.maximum(gain_b,0))/pair_mass),
            'B2_damage_mass':float(np.sum(joint*np.maximum(-gain_b,0))/pair_mass),
            'D2_positive_improvement_mass':float(np.sum(joint*np.maximum(gain_d,0))/pair_mass),
            'D2_damage_mass':float(np.sum(joint*np.maximum(-gain_d,0))/pair_mass),
            'shared_rescue_pair_fraction':float(joint[(gain_b>0)&(gain_d>0)].sum()/pair_mass),
            'B2_only_rescue_pair_fraction':float(joint[(gain_b>0)&(gain_d<=0)].sum()/pair_mass),
            'D2_only_rescue_pair_fraction':float(joint[(gain_d>0)&(gain_b<=0)].sum()/pair_mass)})
        for baseline_credit in range(3):
            selected=joint[baseline_credit];mass=int(selected.sum());b_credit=np.arange(3)[:,None]/2;c_credit=np.arange(3)[None,:]/2
            conditional_rows.append({'fold':fold,'BL80_pair_credit':baseline_credit/2,'pairs':mass,
                'B2_conditional_pair_accuracy':float((selected*b_credit).sum()/mass) if mass else None,
                'D2_conditional_pair_accuracy':float((selected*c_credit).sum()/mass) if mass else None})
        for start,stop in zip(cuts[:-1],cuts[1:]):
            ranks=(rankdata(p[start:stop],axis=0,method='average')-.5)/(stop-start)-.5
            score_cross+=ranks.T@ranks;score_squares+=np.sum(ranks*ranks,axis=0)
        print('NEXT5 S3 EXACT PAIRS',fold,pair_mass,flush=True)
    corr=score_cross/np.sqrt(score_squares[:,None]*score_squares[None,:])
    for i,left in enumerate(('BL80','B2','D2')):
        for j,right in enumerate(('BL80','B2','D2')):score_correlations.append({'left':left,'right':right,'within_fold_time_Spearman':float(corr[i,j])})
    reg_b=read_json(BASE/'references/registrations/REF_B2_augmented_J.json');reg_d=read_json(BASE/'references/registrations/REF_D2_expert.json')
    xb,_,cb=build_design(data,reg_b);xd,_,cd=build_design(data,reg_d);assert xb.shape[1]==93 and xd.shape[1]==36
    totals={kind:{'xy':np.zeros((16,36)),'xx':np.zeros(16),'yy':np.zeros(36)} for kind in ('Pearson','Spearman')}
    with threadpool_limits(limits=2):
        for fold in range(5):
            rows=np.flatnonzero(metadata['fold']==fold);order,cuts=time_groups(metadata['time_online'][rows]);rows=rows[order]
            for block,(start,stop) in enumerate(zip(cuts[:-1],cuts[1:])):
                ix=rows[start:stop];left=np.asarray(xb[ix,77:],dtype=float);right=np.asarray(xd[ix],dtype=float)
                for kind in ('Pearson','Spearman'):
                    a=left if kind=='Pearson' else (rankdata(left,axis=0,method='average')-.5)/len(ix)
                    b=right if kind=='Pearson' else (rankdata(right,axis=0,method='average')-.5)/len(ix)
                    a=a-a.mean(axis=0);b=b-b.mean(axis=0)
                    totals[kind]['xy']+=a.T@b;totals[kind]['xx']+=np.sum(a*a,axis=0);totals[kind]['yy']+=np.sum(b*b,axis=0)
                if block%200==0:progress('S3 within-time feature geometry','fold '+str(fold)+' time block '+str(block)+'; B2 semitail16 x D2 likelihood36')
            print('NEXT5 S3 FEATURE GEOMETRY',fold,flush=True)
    feature_rows=[]
    for i,name_b in enumerate(cb['feature_names'][77:]):
        for j,name_d in enumerate(cd['feature_names']):
            row={'B2_feature':name_b,'D2_feature':name_d}
            for kind,stats in totals.items():
                denominator=np.sqrt(stats['xx'][i]*stats['yy'][j]);row['within_fold_time_'+kind]=float(stats['xy'][i,j]/denominator) if denominator>0 else None
            feature_rows.append(row)
    tables={'JOINT_PAIR_ORDER.csv':joint_rows,'PAIR_DECOMPOSITION.csv':fold_rows,'CONDITIONAL_PAIR_ACCURACY.csv':conditional_rows,
        'SCORE_CORRELATIONS.csv':score_correlations,'FEATURE_CORRELATIONS.csv':feature_rows}
    for name,table in tables.items():
        with (out/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    atomic_json(out/'RESULTS.json',{'created_utc':utc_now(),'status':'PASS','scope':'S3 ResearchPool matched OOF only','new_fits':0,
        'shadow_rows':0,'seal_rows':0,'reference_sources':sources,'feature_cache_keys':[cb['key'],cd['key']],
        'exact_pair_mass':int(all_joint.sum()),'exact_pair_score_parity':True,'pair_sampling':False,
        'feature_pairs':len(feature_rows),'within_fold_time_score_correlations':score_correlations,
        'mean_B2_gain':float(np.mean([r['B2_delta'] for r in fold_rows])),'mean_D2_gain':float(np.mean([r['D2_delta'] for r in fold_rows])),
        'selection_authority':False,'conditional_descriptive_not_an_oracle_candidate':True,'policy_sha256':digest(policy_path),
        'files_sha256':{name:digest(out/name) for name in tables}},immutable=True)
    data.guard.check_state();progress('S3 exact information decomposition complete','All same-time pairs and 576 feature correlations; error overlap does not by itself prove recoverable new signal')


if __name__=='__main__':main()
