"""Exact feature differences, H-only correlation correction and OOF disagreements."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next.candidate_io import candidate_design
from src.next2.evaluation import load_full_oof
from src.next2.pruned_runtime import CompactPairState
from scripts.forensic_next_candidate import rank_within_time
from src.utils.artifacts import write_json,sha256,utc_now
from src.next2.session import checkpoint

J='J102_SERIAL_VARIANCE_512'
Y='Y001_LONG_VARIANCE_ARCH1'


def main():
    data=prepare_dev(ROOT)
    jc,jr,meta,jp,js=load_full_oof(data,J)
    yc,yr,ym,yp,ys=load_full_oof(data,Y)
    x,xm,names,xs,rows=candidate_design(data,jc)
    z,zm,ynames,zs,zrows=candidate_design(data,yc)
    np.testing.assert_array_equal(rows,zrows)
    if [n.replace('serial_variance__','conditional_variance__') for n in names]!=ynames:
        raise RuntimeError('Serial and uncorrected feature contracts differ')
    for key in meta:
        np.testing.assert_array_equal(meta[key],ym[key])
        np.testing.assert_array_equal(xm[key],zm[key])
    out=ROOT/'artifacts/next2/serial_correction'
    out.mkdir(parents=True,exist_ok=True)
    identity={'script_sha256':sha256(Path(__file__)),'j_oof_sources':js,'y_oof_sources':ys,
        'j_feature_sources':xs,'y_feature_sources':zs,'comparison':'Matched old-family 51 plus raw-energy/forecast 26; original five-fold OOF'}
    write_json(out/'POLICY.json',identity)
    started=time.perf_counter()
    feature_rows=[]
    unchanged_expected=0
    for k,name in enumerate(names):
        a,b=x[:,k],z[:,k]
        equal=np.array_equal(a,b)
        changed=np.count_nonzero(a!=b)
        delta=a.astype(float)-b.astype(float)
        correlation=1. if equal and np.std(a)>0 else float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
        local=name.startswith('arch8__') or name.endswith(('_current','_ewma8','_ewma32','_ewma128','_cusum_positive','_cusum_negative'))
        if local:
            unchanged_expected+=1
            if not equal:
                raise AssertionError('Serial correction changed a nominally untouched local/base feature')
        feature_rows.append({'feature':name,'serial_correction_targets_this_statistic':not local,'bitwise_equal':equal,
            'changed_points':int(changed),'changed_fraction':changed/len(a),'pearson':correlation,
            'mean_signed_difference':float(delta.mean()),'mean_absolute_difference':float(np.abs(delta).mean()),
            'p99_absolute_difference':float(np.quantile(np.abs(delta),.99)),'max_absolute_difference':float(np.abs(delta).max())})
    pd.DataFrame(feature_rows).to_csv(out/'FEATURE_DIFFERENCES.csv',index=False)
    signature=ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    slock=json.loads((signature.parent/'REGIME_ASSIGNMENT_LOCK.json').read_text())
    if sha256(signature)!=slock['files_sha256'][signature.name]:
        raise RuntimeError('H diagnostic metadata changed')
    smeta=pd.read_parquet(signature).set_index('dataset_id')
    data.guard.admit(smeta.index.to_numpy(),purpose='Serial DEV H strata')
    rho_rows=[]
    ho=data.series['historical_offsets']
    for k,sid in enumerate(data.ids):
        state=CompactPairState(data.historical[ho[k]:ho[k+1]],jc['extension']['settings'],'serial')
        rho=(state.age_variances[:,1]-2.)/2.
        row={'dataset_id':int(sid),'fold':data.guard.split.assignment(int(sid))[1],
            'historical_regime':int(smeta.loc[int(sid),'historical_regime'])}
        for channel,name in enumerate(('raw_energy','forecast')):
            row[name+'_rho']=float(rho[channel])
            row[name+'_variance512_over_512']=float(state.age_variances[channel,-1]/512.)
        rho_rows.append(row)
    rho_frame=pd.DataFrame(rho_rows)
    rho_frame.to_csv(out/'H_CORRELATION_DIAGNOSTICS.csv',index=False)
    corr_rows,case_rows=[],[]
    for fold in range(5):
        ix=np.flatnonzero(meta['fold']==fold)
        p,q,t=jp[ix],yp[ix],meta['time_online'][ix]
        pr,qr=rank_within_time(p,t),rank_within_time(q,t)
        ids=meta['dataset_id'][ix]
        hmeta=smeta.reindex(ids)
        masks={'all':np.ones(len(ix),dtype=bool),'negative_points':meta['target'][ix]==0,
            'no_break_controls':hmeta.tau.to_numpy()<0}
        masks.update({'regime_'+str(r):hmeta.historical_regime.to_numpy()==r for r in range(5)})
        for scope,mask in masks.items():
            if not mask.any():
                continue
            r={'fold':fold,'scope':scope,'points':int(mask.sum()),'series':len(np.unique(ids[mask])),
                'pearson':float(np.corrcoef(p[mask],q[mask])[0,1]),
                'within_time_spearman':float(np.corrcoef(pr[mask],qr[mask])[0,1]),
                'mean_absolute_rank_disagreement':float(np.abs(pr[mask]-qr[mask]).mean())}
            for fraction in (.01,.05):
                a,b=pr[mask]>=1-fraction,qr[mask]>=1-fraction
                r['top_'+str(fraction)+'_jaccard']=float((a&b).sum()/max((a|b).sum(),1))
                r['top_'+str(fraction)+'_disagreement_fraction']=float((a!=b).mean())
                r['top_'+str(fraction)+'_j_only_fraction']=float((a&~b).mean())
                r['top_'+str(fraction)+'_y_only_fraction']=float((~a&b).mean())
            corr_rows.append(r)
        cases=pd.DataFrame({'dataset_id':ids,'absolute_rank_difference':np.abs(pr-qr),'absolute_probability_difference':np.abs(p-q)})
        grouped=cases.groupby('dataset_id').agg(mean_absolute_rank_difference=('absolute_rank_difference','mean'),
            max_absolute_rank_difference=('absolute_rank_difference','max'),mean_absolute_probability_difference=('absolute_probability_difference','mean'))
        selected=grouped.sort_values('mean_absolute_rank_difference',ascending=False).head(20).reset_index()
        selected['fold']=fold
        case_rows.extend(selected.to_dict('records'))
    pd.DataFrame(corr_rows).to_csv(out/'OOF_DISAGREEMENT.csv',index=False)
    pd.DataFrame(case_rows).merge(rho_frame,on=['dataset_id','fold'],how='left').to_csv(out/'LARGEST_DISAGREEMENT_SERIES.csv',index=False)
    summary=[]
    for regime,frame in rho_frame.groupby('historical_regime'):
        for name in ('raw_energy_rho','forecast_rho','raw_energy_variance512_over_512','forecast_variance512_over_512'):
            summary.append({'historical_regime':int(regime),'quantity':name,'series':len(frame),
                'q05':float(frame[name].quantile(.05)),'median':float(frame[name].median()),'q95':float(frame[name].quantile(.95))})
    pd.DataFrame(summary).to_csv(out/'H_CORRECTION_REGIMES.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'identity':identity,'dev_series':len(data.ids),
        'points':len(jp),'features':len(names),'expected_untouched_features':unchanged_expected,
        'bitwise_equal_features':sum(r['bitwise_equal'] for r in feature_rows),'new_fits':0,
        'quality_delta_j_minus_y':jr['full']['mean']-yr['full']['mean'],'elapsed_seconds':time.perf_counter()-started,
        'files_sha256':{p.name:sha256(p) for p in out.glob('*.csv')},'seal_rows':0,
        'interpretation':'H-fitted AR(1) correction changes age-based evidence. This feature analysis alone does not establish incremental predictive value or non-inferiority of removing correction.'})
    checkpoint('Serial feature-level and OOF disagreement complete',['Matched component bootstrap synthesis','Independent runtime profiles'],
        artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    print('NEXT2 SERIAL ANALYSIS COMPLETE',unchanged_expected,flush=True)


if __name__=='__main__':
    main()
