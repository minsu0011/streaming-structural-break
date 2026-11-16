"""Fixed historical-structure slices of existing DEV OOF predictions only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.validation.splits import FrozenSplit,assert_development_ids
from src.validation.group_bootstrap import draw_group_multiplicities,weighted_ts_auc_replicates
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json,sha256,utc_now
from scripts.bootstrap_real_oof import interval

CANDIDATES=('M7_cross_order_AR4_LGB_AR8_HGB_equal','M4_ar_residual_input_ABCD_S3_9f68be316ea3','M3_histgb_AR8_ARCH1_ABCD_S3_1cb5e51ac7ae')
BINS={'historical_length':(2000,3500),'historical_lag1':(-.5,.5),'historical_excess_kurtosis':(0.,10.),'ar8_squared_acf1':(.05,.2)}
REPLICATES=400


def cohort_codes(frame):
    """Only four explicitly whitelisted historical columns enter membership."""
    result={}
    for name,bounds in BINS.items():
        values=frame[name].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():raise ValueError('Historical cohort input is not finite')
        code=np.searchsorted(np.asarray(bounds),values,side='right')
        result[name]=code
    return result


def main():
    directory=ROOT/'artifacts/real/historical_cohorts'
    if (directory/'RESULTS.json').exists():raise RuntimeError('Completed diagnostic cannot be overwritten')
    split=FrozenSplit(ROOT/'artifacts/validation');ids=split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'].tolist()
    policy={'timestamp':utc_now(),'candidates':CANDIDATES,'historical_bins':BINS,'boundary_rule':'Left closed/right open intervals; values equal to a boundary enter the upper bin.',
        'membership':'DEV historical length, lag-one correlation, excess kurtosis, and previously audited AR8 historical squared-innovation lag-one correlation. No outcome, tau, online length or OOF prediction enters membership.',
        'status':'FIXED_BEFORE_THIS_COHORT_SCORE_DIAGNOSTIC_AFTER_MODEL_SELECTION','replicates':REPLICATES,'seed':20260908,
        'bootstrap_unit':'whole frozen duplicate group, stratified by original validation fold and historical cohort; same multiplicities across candidates',
        'interval_scope':'Marginal 95% percentile intervals conditional on fitted OOF predictions. Neither model-selection adjusted nor simultaneous across cohorts.',
        'score_scope':'Five-fold mean of within-cohort TS-AUC. Cross-cohort positive/negative pairs are excluded, so the cohort scores are not additive contributions to the overall TS-AUC.',
        'small_support_flag':'Fewer than 100 validation series or fewer than 25 positive/negative series in any fold; reported, not silently omitted.',
        'no_new_model_or_promotion':True,'split_sha256':split.digest,'script_sha256':sha256(__file__),'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    if (directory/'POLICY.json').exists():
        existing=json.loads((directory/'POLICY.json').read_text())
        comparable=lambda value:json.loads(json.dumps({k:v for k,v in value.items() if k not in ('timestamp','script_sha256')}))
        if comparable(existing)!=comparable(policy):raise RuntimeError('Existing diagnostic policy cannot be reselected')
        policy=existing
    else:write_json(directory/'POLICY.json',policy)
    columns=['dataset_id','historical_length','historical_lag1','historical_excess_kurtosis','has_break']
    metadata=pd.read_parquet(ROOT/'artifacts/audits/SERIES_METADATA.parquet',columns=columns,filters=[('dataset_id','in',ids)]).set_index('dataset_id')
    residual=pd.read_csv(ROOT/'artifacts/real/nonlinear_residual_audit/SERIES_DIAGNOSTICS.csv')
    residual=residual.loc[residual['transform']=='squared'].set_index('dataset_id')
    if not residual.index.is_unique or set(residual.index)!=set(ids):raise AssertionError('Historical residual coverage mismatch')
    metadata['ar8_squared_acf1']=residual.acf_1.reindex(metadata.index)
    if set(metadata.index)!=set(ids):raise AssertionError('Historical metadata DEV coverage mismatch')
    codes=cohort_codes(metadata);members={'all':set(ids)}
    for name,values in codes.items():
        for code in range(3):members[f'{name}__bin_{code}']=set(metadata.index[values==code])
    records=[];samples={cohort:{eid:[] for eid in CANDIDATES} for cohort in members};started=time.perf_counter()
    manifests={eid:json.loads((ROOT/'data/processed/oof'/eid/'MANIFEST.json').read_text()) for eid in CANDIDATES}
    if any(v['split_sha256']!=split.digest for v in manifests.values()):raise AssertionError('Candidate split differs')
    for fold in range(5):
        reference=None;predictions={}
        for eid in CANDIDATES:
            cache=ROOT/manifests[eid]['cache_directory'];manifest=json.loads((cache/'MANIFEST.json').read_text());path=cache/f'fold_{fold}.parquet'
            if sha256(path)!=manifest['fold_sha256'][str(fold)]:raise AssertionError('Frozen feature cache bytes changed')
            frame=pd.read_parquet(path,columns=['dataset_id','time_online','target'])
            if reference is None:reference=frame
            else:pd.testing.assert_frame_equal(frame,reference)
            prediction=np.load(ROOT/'data/processed/oof'/eid/f'fold_{fold}.npy',allow_pickle=False)
            if len(prediction)!=len(reference):raise AssertionError('OOF row count differs')
            predictions[eid]=prediction
        assert_development_ids(reference.dataset_id.unique(),split)
        expected=set(split.frame.loc[(split.frame.role=='DEVELOPMENT_POOL')&(split.frame.fold==fold),'dataset_id'])
        if set(reference.dataset_id.unique())!=expected:raise AssertionError('Incomplete OOF fold coverage')
        target=reference.target.to_numpy();online_time=reference.time_online.to_numpy()
        for number,(cohort,cohort_ids) in enumerate(members.items()):
            present=expected&cohort_ids;mask=reference.dataset_id.isin(present).to_numpy()
            groups=reference.loc[mask,'dataset_id'].map(split.groups).to_numpy()
            code,counts=draw_group_multiplicities(groups,REPLICATES,20260908+fold+100*number)
            negative_series=int((~metadata.loc[sorted(present),'has_break']).sum());positive_series=len(present)-negative_series
            time_counts=np.bincount(online_time[mask]);positive_counts=np.bincount(online_time[mask],weights=target[mask],minlength=len(time_counts))
            pairs=int(np.dot(positive_counts,time_counts-positive_counts))
            for eid in CANDIDATES:
                sample=weighted_ts_auc_replicates(target[mask],predictions[eid][mask],online_time[mask],code,counts)
                np.testing.assert_allclose(sample[0],ts_auc(target[mask],predictions[eid][mask],online_time[mask]),atol=1e-14,rtol=0)
                samples[cohort][eid].append(sample)
                records.append({'cohort':cohort,'candidate':eid,'fold':fold,'series':len(present),'break_series':positive_series,'no_break_series':negative_series,
                    'rows':int(mask.sum()),'within_cohort_pairs':pairs,'ts_auc':float(sample[0]),'small_support':len(present)<100 or min(positive_series,negative_series)<25})
        print('HISTORICAL COHORTS FOLD',fold,'complete',flush=True)
    summaries=[]
    for cohort,by_candidate in samples.items():
        primary=np.array(by_candidate[CANDIDATES[0]]).mean(axis=0)
        for eid,values in by_candidate.items():
            matrix=np.asarray(values);mean=matrix.mean(axis=0);support=[r for r in records if r['cohort']==cohort and r['candidate']==eid]
            summaries.append({'cohort':cohort,'candidate':eid,'series':sum(r['series'] for r in support),'fold_min_series':min(r['series'] for r in support),
                'small_support':any(r['small_support'] for r in support),'mean':float(mean[0]),'median':float(np.median(matrix[:,0])),'worst_fold':float(matrix[:,0].min()),
                'mean_interval':interval(mean),'difference_from_primary':interval(mean-primary)})
    pd.DataFrame(records).to_csv(directory/'FOLDS.csv',index=False)
    flattened=[]
    for row in summaries:
        item={k:v for k,v in row.items() if not isinstance(v,dict)}
        item.update({f'{name}_{key}':value for name in ('mean_interval','difference_from_primary') for key,value in row[name].items()});flattened.append(item)
    pd.DataFrame(flattened).to_csv(directory/'SUMMARY.csv',index=False)
    report={'status':'PASS','timestamp':utc_now(),'policy':policy,'execution_script_sha256':sha256(__file__),'summaries':summaries,'folds':records,'elapsed_seconds':time.perf_counter()-started,
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False,'models_changed':False,'group_membership_source_sha256':sha256(ROOT/'artifacts/real/nonlinear_residual_audit/SERIES_DIAGNOSTICS.csv')}
    write_json(directory/'RESULTS.json',report);print('HISTORICAL COHORTS COMPLETE',len(members),'cohorts',report['elapsed_seconds'],'seconds',flush=True)


if __name__=='__main__':main()
