"""Single release of fully frozen outer predictions; fixed paired inference."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np
from numba import set_num_threads
from src.next6.common import BASE,read_json,save,digest,activity,utc_now,checkpoint
from src.next6.data import arrays
from src.next6.evaluation import frozen_predictions,csv_write,METHODS
from src.next5.bootstrap import PreparedAUC,paired_summary
from src.next5.data import admitted_npz
from src.next5.fractures import masks,pair_mass
from src.scoring.ts_auc import ts_auc

def main():
    data,meta,values=arrays();pred,receipts=frozen_predictions(data,meta);y=meta['target'];t=meta['time_online'];folds=meta['fold']
    if not (BASE/'OUTER_SCORE_RELEASE.json').exists():
        save('OUTER_SCORE_RELEASE.json',{'created_utc':utc_now(),'status':'ALL_PREDICTIONS_FROZEN_BEFORE_RELEASE','receipt':receipts,
            'inner_freeze_hashes':{str(f):digest(BASE/f'fold_{f}/INNER_FREEZE.json') for f in range(5)},'protocol_sha256':digest(BASE/'NEXT6_PROTOCOL_LOCK.json'),
            'no_adaptive_refit_allowed':True})
    scores={};nested=[]
    with activity('fixed_outer_scoring','EVALUATION'):
        for m,p in pred.items():
            scores[m]=[]
            for f in range(5):
                ix=(folds==f)&np.isfinite(p)
                if not ix.any():scores[m].append(None);continue
                score=float(ts_auc(y[ix],p[ix],t[ix]));scores[m].append(score)
                nested.append({'stage':'outer','method':m,'fold':f,'TS_AUC':score,'series':len(np.unique(meta['dataset_id'][ix])),
                    'rows':int(ix.sum()),'positive_negative_time_pairs':pair_mass(y[ix],t[ix])})
        for f in range(5):
            for held in range(3):
                for m in ('M1','M2','M3','M4'):
                    r=read_json(BASE/f'fold_{f}/inner_{held}/{m}/DONE.json')
                    nested.append({'stage':'inner','method':m,'fold':f,'inner_holdout':held,'TS_AUC':r['TS_AUC']})
        if not (BASE/'NESTED_CV_NEXT6.csv').exists():csv_write('NESTED_CV_NEXT6.csv',nested)
        comparison=[]
        for m,s in scores.items():
            valid=np.array([v is not None for v in s]);a=np.array([v for v in s if v is not None]);b=np.array(scores['M0'])[valid];c=np.array(scores['M1'])[valid];r=np.array(scores['RANDOM'])[valid]
            delta=a-b
            comparison.append({'method':m,'outer_folds':len(a),'full_5_fold_candidate':len(a)==5,'mean_TS_AUC':float(a.mean()),'median_TS_AUC':float(np.median(a)),
                'worst_TS_AUC':float(a.min()),'delta_matched_M0':float(delta.mean()),'delta_same_input_M1':float((a-c).mean()),
                'delta_same_size_RANDOM':float((a-r).mean()),'positive_matched_folds':int((delta>0).sum()),'worst_matched_fold_delta':float(delta.min()),
                'interpretation':'partial M4 eligibility; subset scores cannot support full-fold promotion' if len(a)<5 else 'fixed 5-fold ResearchPool DEV evaluation'})
        if not (BASE/'MODEL_COMPARISON_NEXT6.csv').exists():csv_write('MODEL_COMPARISON_NEXT6.csv',comparison)
        if not (BASE/'OUTER_SCORES.json').exists():save('OUTER_SCORES.json',scores)
    set_num_threads(min(8,os.cpu_count() or 1));boot={};rng=np.random.default_rng(20260913)
    weights=[]
    for f in range(5):
        n=int(np.sum(data.series['fold']==f));weights.append(np.vstack((np.ones((1,n),dtype=np.int32),rng.multinomial(n,np.full(n,1/n),size=1000))))
    directory=BASE/'evaluation_cache';directory.mkdir(exist_ok=True)
    for m,p in pred.items():
        path=directory/(m+'_bootstrap.npy')
        if path.exists():boot[m]=np.load(path);continue
        with activity('paired_bootstrap_'+m,'UNCERTAINTY'):
            out=np.full((1001,5),np.nan)
            for f in range(5):
                ix=(folds==f)&np.isfinite(p)
                if not ix.any():continue
                _,codes=np.unique(meta['dataset_id'][ix],return_inverse=True);n=weights[f].shape[1]
                prepared=PreparedAUC.create(y[ix],p[ix],t[ix],codes,n);out[:,f]=prepared.evaluate(weights[f])
                assert abs(out[0,f]-scores[m][f])<1e-12
                checkpoint('paired bootstrap',f'{m} fold{f}')
            np.save(path,out,allow_pickle=False);boot[m]=out
    if not (BASE/'PAIRED_UNCERTAINTY_NEXT6.csv').exists():
        rows=[]
        for m in pred:
            valid=np.isfinite(boot[m][0])
            for ref in ('M0','M1','RANDOM','OWN'):
                if m==ref:continue
                r=paired_summary(boot[ref][:,valid],boot[m][:,valid]);rows.append({'candidate':m,'reference':ref,'folds':int(valid.sum()),
                    'conditional_on_fixed_predictions':True,'resampling_unit':'whole series within frozen outer fold','adaptive_DEV_reuse_unaccounted':True,**r})
        csv_write('PAIRED_UNCERTAINTY_NEXT6.csv',rows)
    if not (BASE/'MECHANISM_RESULTS_NEXT6.csv').exists():
        with activity('fixed_time_age_mechanism_diagnostics','EVALUATION'):
            d=ROOT/'artifacts/next5/diagnostics';lock=read_json(d/'RESEARCH_DIAGNOSTIC_LOCK.json')
            assert digest(d/'RESEARCH_SIGNATURES.npz')==lock['files_sha256']['RESEARCH_SIGNATURES.npz']
            signature=admitted_npz(d/'RESEARCH_SIGNATURES.npz',data.guard,'NEXT6 posthoc mechanism diagnostics only')
            si=np.argsort(signature['dataset_id']);lookup=np.searchsorted(signature['dataset_id'][si],meta['dataset_id']);srow=si[lookup]
            assert np.array_equal(signature['dataset_id'][srow],meta['dataset_id'])
            scopes=masks(y,t,signature['tau'][srow],signature['historical_regime'][srow],signature['signature'][srow],signature['observed_scale_direction'][srow])
            # H-only quartiles fit separately on each outer training set; no outer-bin fitting.
            for col,name in [(0,'H_length'),(1,'H_scale'),(2,'H_tail'),(3,'H_dependence')]:
                code=np.empty(len(data.ids),dtype=np.int8)
                for f in range(5):
                    tr=data.series['fold']!=f;va=~tr;q=np.quantile(values['H_descriptors'][tr,col],[.25,.5,.75]);code[va]=np.searchsorted(q,values['H_descriptors'][va,col],side='right')
                rowcode=code[np.searchsorted(data.ids,meta['dataset_id'])]
                for b in range(4):scopes[name+'_quartile_'+str(b)]=rowcode==b
            timeline=[];mechanism=[]
            for scope,mask in scopes.items():
                dest=timeline if scope=='all_time' or scope.startswith(('time_','age_')) else mechanism
                for f in range(5):
                    use=mask&(folds==f);mass=pair_mass(y[use],t[use])
                    if not mass:continue
                    p0=float(ts_auc(y[use],pred['M0'][use],t[use]))
                    for m,p in pred.items():
                        if not np.isfinite(p[use]).all():continue
                        score=float(ts_auc(y[use],p[use],t[use]));dest.append({'scope':scope,'fold':f,'method':m,'TS_AUC':score,'delta_M0':score-p0,
                            'series':len(np.unique(meta['dataset_id'][use])),'rows':int(use.sum()),'positive_negative_time_pairs':mass,
                            'diagnostic_only':True,'legacy_regime_historical_transductive':scope.startswith('regime_')})
            csv_write('TIME_AND_BREAK_AGE_NEXT6.csv',timeline);csv_write('MECHANISM_RESULTS_NEXT6.csv',mechanism)
    data.guard.check_state()

if __name__=='__main__':main()
