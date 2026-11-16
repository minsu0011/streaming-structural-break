"""Equal-series rank/centered/partial relation maps on one outer TRAIN only."""
from pathlib import Path
import sys,os,csv,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np
from scipy.stats import rankdata
from scipy.spatial import cKDTree
from src.next6.common import BASE,read_json,save,digest,activity,checkpoint,ids_hash
from src.next6.data import arrays
from src.next6.nodes import NAMES,KOREAN,LAGS

def correlation(a,b,partial=False):
    a=rankdata(a,axis=0);b=rankdata(b,axis=0);a-=a.mean(axis=0);b-=b.mean(axis=0)
    if partial:
        # Remove within-window linear trend and common marginal amplitude.
        trend=np.linspace(-1,1,len(a));common=np.mean(abs(a),axis=1);common-=common.mean()
        control=np.column_stack((trend,common));coef=np.linalg.lstsq(control,np.column_stack((a,b)),rcond=None)[0]
        residual=np.column_stack((a,b))-control@coef;a,b=residual[:,:16],residual[:,16:]
    denom=np.sqrt(np.sum(a*a,axis=0)[:,None]*np.sum(b*b,axis=0)[None,:])
    return np.divide(a.T@b,denom,out=np.full((16,16),np.nan),where=denom>1e-12)

def main():
    p=BASE/'CORRELATION_ATLAS/TRAIN_ATLAS.json'
    if p.exists():print('Existing atlas retained');return
    with activity('outer0_train_correlation_atlas','RELATION_ANALYSIS'):
        data,metadata,arrays_=arrays();protocol=read_json(BASE/'NEXT6_PROTOCOL_LOCK.json');x=arrays_['nodes'];hdesc=arrays_['H_descriptors']
        train=[int(i) for i in data.ids if protocol['outer_mapping'][str(int(i))]!=0];data.guard.admit(train,purpose='fold0 training atlas')
        tau=data.series['tau'];offset=data.series['online_offsets'];trainix=[data.id_to_index[i] for i in train]
        negative=[i for i in trainix if tau[i]<0];means=np.mean(hdesc[trainix,:4],axis=0);spread=np.maximum(np.std(hdesc[trainix,:4],axis=0),1e-6)
        # Match H descriptors fitted exclusively in the displayed training fold; reuse allowed and reported.
        tree=cKDTree((hdesc[negative,:4]-means)/spread)
        stats={};support=[];matched=[]
        for count,ix in enumerate(trainix):
            sid=int(data.ids[ix]);a,b=offset[ix:ix+2];v=np.asarray(x[a:b]);n=b-a;time=np.arange(n);tv=int(tau[ix]);windows={}
            if tv<0:windows['normal_no_break']=time
            else:
                windows['normal_not_yet']=time[time<tv]
                for name,(low,high) in protocol['phase_intervals'].items():
                    rel=time-tv;windows[name]=time[(rel>=low)&(rel<=high if high is not None else True)]
                distance,at=tree.query((hdesc[ix,:4]-means)/spread);ni=negative[int(at)];na,nb=offset[ni:ni+2]
                matched.append({'event_id':sid,'control_id':int(data.ids[ni]),'H_distance':float(distance),'same_absolute_time':True})
                for name in ('pre','onset','earlyPost','establishedPost','laterPost'):
                    times=windows[name];available=times[times<nb-na]
                    support.append({'series_id':sid,'phase':name,'available':len(times),'matched_control_available':len(available),'matched_censored':len(times)-len(available)})
                    if len(available)>=4:
                        vv=np.asarray(x[na+available]);stats.setdefault(('matched_normal_'+name,0,'rank'),[]).append(correlation(vv,vv))
            for name,times in windows.items():
                if len(times)<4:continue
                for lag in (0,*LAGS):
                    use=times[times>=lag]
                    if len(use)<4:continue
                    for mode in ('rank','partial'):
                        stats.setdefault((name,lag,mode),[]).append(correlation(v[use-lag],v[use],mode=='partial'))
            if count%500==0:checkpoint('training atlas',f'{count}/{len(train)} series; no outer-validation maps')
        out=BASE/'CORRELATION_ATLAS';out.mkdir(exist_ok=True)
        results=[];rng=np.random.default_rng(20260913)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            for (phase,lag,mode),values in stats.items():
                stack=np.asarray(values);mean=np.nanmean(stack,axis=0);support_matrix=np.sum(np.isfinite(stack),axis=0)
                # Series are the resampling unit; no pointwise iid p-values.
                boot=[]
                for _ in range(200):boot.append(np.nanmean(stack[rng.integers(len(stack),size=len(stack))],axis=0))
                lo,hi=np.nanquantile(boot,[.025,.975],axis=0)
                results.append({'phase':phase,'lag':lag,'mode':mode,'series':len(values),
                    'mean':np.nan_to_num(mean).tolist(),'support':support_matrix.tolist(),
                    'CI_low':np.nan_to_num(lo).tolist(),'CI_high':np.nan_to_num(hi).tolist()})
        save('CORRELATION_ATLAS/TRAIN_ATLAS.json',{'status':'COMPLETE','scope':'OUTER_FOLD0_TRAIN_DISCOVERY_ONLY',
            'training_ids_sha256':ids_hash(train),'training_series':len(train),'feature_names':list(NAMES),'korean_meaning':list(KOREAN),
            'method':'mean of within-series rank correlations; partial removes linear time and common rank amplitude; equal series weights',
            'uncertainty':'200 whole-series resamples; overlapping matched controls mean descriptive matched-map CI only, not independent causal evidence',
            'results':results,'mechanical':'All nodes derived from same univariate raw stream; mechanical connections identified in lineage',
            'sample_count_note':'per-cell independent-series support is explicit; absent windows not padded'})
        for name,rows in [('MATCHED_CONTROL_SUPPORT.csv',support),('MATCHED_CONTROL_IDS.csv',matched)]:
            with (out/name).open('x',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
