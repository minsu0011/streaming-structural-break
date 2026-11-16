"""Paired 2000-series bootstrap against J102, with unchanged 24 DEV slices."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import sys
import json
import time
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.next.bootstrap import PreparedAUC,paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from scripts.forensic_next_candidate import rank_within_time
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def masks_for(meta,y,t):
    age=t-meta.tau.to_numpy()
    masks={'all_time':np.ones(len(y),dtype=bool)}
    for name,a,b in [('0-9',0,10),('10-24',10,25),('25-49',25,50),('50-99',50,100),('100-199',100,200),('200-399',200,400),('400+',400,100000)]:
        masks['time_'+name]=(t>=a)&(t<b)
    for name,a,b in [('0-4',0,5),('5-9',5,10),('10-19',10,20),('20-49',20,50),('50+',50,100000)]:
        masks['age_'+name]=(y==0)|((y==1)&(age>=a)&(age<b))
    for regime in range(5):
        masks['regime_'+str(regime)]=meta.historical_regime.to_numpy()==regime
    for signature in ['MEAN_DOMINANT','SCALE_DOMINANT','DEPENDENCE_DOMINANT','TAIL_DOMINANT','MIXED','WEAK_OR_AMBIGUOUS']:
        masks['mechanism_'+signature]=np.isin(meta.signature.to_numpy(),[signature,'NO_BREAK_CONTROL'])
    return masks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    args=parser.parse_args()
    set_num_threads(2)
    data=prepare_dev(ROOT)
    c,original,metadata,pred,sources=load_full_oof(data,args.candidate)
    _,_,refmeta,reference,refsources=load_full_oof(data,'J102_SERIAL_VARIANCE_512')
    for k in metadata:
        np.testing.assert_array_equal(metadata[k],refmeta[k])
    out=ROOT/'artifacts/next2/forensic'/args.candidate
    out.mkdir(parents=True,exist_ok=True)
    meta_path=ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    regime_lock=json.loads((meta_path.parent/'REGIME_ASSIGNMENT_LOCK.json').read_text(encoding='utf-8'))
    assert sha256(meta_path)==regime_lock['files_sha256'][meta_path.name]
    smeta=pd.read_parquet(meta_path).set_index('dataset_id')
    data.guard.admit(smeta.index.to_numpy(),purpose='NEXT2 DEV-only diagnostic signatures')
    refdir=ROOT/'artifacts/next/forensic/J102_SERIAL_VARIANCE_512'
    refreceipt=json.loads((refdir/'RESULTS.json').read_text(encoding='utf-8'))
    policy={'candidate':c,'source_sha256':sources,'reference_sha256':refsources,
        'diagnostic_metadata_sha256':sha256(meta_path),'replicates':2000,'seed_base':202609094,
        'next2_policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),
        'script_sha256':sha256(Path(__file__)),'original_status':original['status']}
    if (out/'POLICY.json').exists():
        assert json.loads((out/'POLICY.json').read_text(encoding='utf-8'))==policy
    else:
        write_json(out/'POLICY.json',policy)
    if (out/'RESULTS.json').exists():
        r=json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
        for name,digest in r['files_sha256'].items():
            assert sha256(out/name)==digest
        print('NEXT2 FORENSIC REUSED',args.candidate,flush=True)
        return
    started=time.perf_counter()
    candidates,references,supports,folds,correlations={},{},[],[],[]
    for fold in range(5):
        rows=np.flatnonzero(metadata['fold']==fold)
        ids,y,t=(metadata[k][rows] for k in ('dataset_id','target','time_online'))
        p=pred[rows]; ref=reference[rows]
        meta=smeta.reindex(ids)
        code,counts=draw_group_multiplicities(ids,2000,202609094+fold)
        for scope,mask in masks_for(meta,y,t).items():
            name=f'fold_{fold}_{scope}.npy'
            receipt=out/(name+'.json')
            if (out/name).exists() and receipt.exists():
                assert sha256(out/name)==json.loads(receipt.read_text())['sha256']
                values=np.load(out/name,allow_pickle=False)
            else:
                values=PreparedAUC.create(y[mask],p[mask],t[mask],code[mask],counts.shape[1]).evaluate(counts)
                np.save(out/name,values,allow_pickle=False)
                write_json(receipt,{'sha256':sha256(out/name)})
            assert sha256(refdir/name)==refreceipt['files_sha256'][name]
            reference_values=np.load(refdir/name,allow_pickle=False)
            candidates.setdefault(scope,[]).append(values)
            references.setdefault(scope,[]).append(reference_values)
            supports.append({'scope':scope,'fold':fold,'break_series':len(np.unique(ids[mask&(meta.tau.to_numpy()>=0)])),
                'null_series':len(np.unique(ids[mask&(meta.tau.to_numpy()<0)])),'rows':int(mask.sum())})
            folds.append({'candidate':args.candidate,'fold':fold,'scope':scope,'ts_auc':values[0],
                'reference_ts_auc':reference_values[0],'delta':values[0]-reference_values[0]})
        p_rank,r_rank=rank_within_time(p,t),rank_within_time(ref,t)
        corr={'candidate':args.candidate,'fold':fold,'pearson':float(np.corrcoef(p,ref)[0,1]),
            'within_time_spearman':float(np.corrcoef(p_rank,r_rank)[0,1]),
            'mean_absolute_rank_disagreement':float(np.abs(p_rank-r_rank).mean())}
        for fraction in (.01,.05):
            a,b=p_rank>=1.-fraction,r_rank>=1.-fraction
            corr['top_'+str(fraction)+'_jaccard']=float((a&b).sum()/max((a|b).sum(),1))
            corr['top_'+str(fraction)+'_null_disagreement']=float(np.mean((a!=b)[y==0]))
        correlations.append(corr)
        checkpoint('Paired NEXT2 forensic '+args.candidate+' fold '+str(fold),
            ['Complete 24 paired scopes','Check fixed role gates'])
        print('NEXT2 FORENSIC',args.candidate,fold,round(time.perf_counter()-started,1),flush=True)
    totals=pd.DataFrame(supports).groupby('scope')[['break_series','null_series']].sum()
    rows=[]
    for scope in candidates:
        new=np.asarray(candidates[scope]).T; ref=np.asarray(references[scope]).T
        row={'candidate':args.candidate,'reference':'J102_SERIAL_VARIANCE_512','scope':scope,
            'candidate_mean':float(new[0].mean()),'reference_mean':float(ref[0].mean()),
            'supported':bool(totals.loc[scope,'break_series']>=100 and (not scope.startswith('regime_') or totals.loc[scope,'null_series']>=100)),
            **totals.loc[scope].astype(int).to_dict(),**paired_summary(ref,new),
            'probability_delta_above_minus_0005':float(np.mean((new.mean(axis=1)-ref.mean(axis=1))[1:]>-.0005))}
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'PAIRED_FRACTURES.csv',index=False)
    pd.DataFrame(folds).to_csv(out/'FOLD_SCORES.csv',index=False)
    pd.DataFrame(supports).to_csv(out/'SUPPORT.csv',index=False)
    pd.DataFrame(correlations).to_csv(out/'DISAGREEMENT.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','candidate':args.candidate,'created_utc':utc_now(),
        'all_time_paired_bootstrap':rows[0],'comparisons':len(rows),'seal_rows':0,
        'elapsed_seconds':time.perf_counter()-started,
        'files_sha256':{p.name:sha256(p) for p in list(out.glob('*.npy'))+list(out.glob('*.csv'))},
        'limitations':'Fixed fitted OOF conditional bootstrap; adaptive nomination, model fitting and multiple comparisons are not corrected.'})
    print('NEXT2 FORENSIC COMPLETE',args.candidate,rows[0],flush=True)


if __name__=='__main__':
    main()
