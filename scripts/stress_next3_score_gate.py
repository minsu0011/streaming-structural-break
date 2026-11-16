"""Evaluate the score-only causal gate from immutable paired synthetic components."""
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.router import heuristic_predict
from src.next2.counterexamples import policy
from src.scoring.ts_auc import ts_auc
from src.next.bootstrap import PreparedAUC,paired_summary


def load_pair(kind,scenario):
    left=ROOT/'artifacts/next2/counterexamples'/kind/scenario
    right=ROOT/'artifacts/next2/finalist_counterexamples'/kind/scenario
    a=read_json(left/'RESULTS.json');b=read_json(right/'RESULTS.json')
    assert a['status']==b['status']=='PASS'
    names=[('J102',left,'J102_SERIAL_VARIANCE_512.npy',a),
        ('C205',right,'C205_DEPENDENCE.npy',b),('BL80',right,'BL2_C205_DEPENDENCE_80.npy',b)]
    arrays={}
    sources={}
    for role,folder,name,receipt in names:
        assert digest(folder/name)==receipt['files_sha256'][name]
        arrays[role]=np.load(folder/name,mmap_mode='r',allow_pickle=False)
        sources[(folder/name).relative_to(ROOT).as_posix()]=receipt['files_sha256'][name]
    for folder,receipt in ((left,a),(right,b)):
        assert digest(folder/'synthetic_tau.npy')==receipt['files_sha256']['synthetic_tau.npy']
    tau=np.load(left/'synthetic_tau.npy',allow_pickle=False)
    np.testing.assert_array_equal(tau,np.load(right/'synthetic_tau.npy',allow_pickle=False))
    shape=arrays['J102'].shape
    assert all(x.shape==shape for x in arrays.values())
    inputs=np.zeros((shape[0]*shape[1],6),dtype=np.float32)
    inputs[:,0]=arrays['J102'].ravel();inputs[:,1]=arrays['C205'].ravel()
    arrays['R1_SCORE_GAP']=heuristic_predict(inputs,'R1_SCORE_GAP')[0].reshape(shape)
    return arrays,tau,sources


def threshold(maxima,alpha=.05):
    rank=math.ceil((len(maxima)+1)*(1-alpha))
    return float(np.sort(maxima)[rank-1]) if rank<=len(maxima) else 1.


def main():
    set_num_threads(2)
    hyp_path=BASE/'hypotheses/R1_SCORE_GAP_SYNTHETIC.json'
    hyp=read_json(hyp_path);assert digest(ROOT/'src/next3/router.py')==hyp['router_sha256']
    out=BASE/'counterexamples/R1_SCORE_GAP';out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULTS.json').exists():raise RuntimeError('Preserve completed synthetic diagnostic')
    lock=policy();scenarios={s['id']:s for s in lock['scenarios']}
    mapping=read_json(ROOT/'artifacts/next2/counterexamples/CALIBRATION_MAPPING.json')['backgrounds']
    calibration={};sources={}
    for item in mapping:
        name=item['calibration_scenario_seed']
        arrays,tau,evidence=load_pair('calibration',name);sources.update(evidence)
        calibration[(item['phi'],item['arch'],item['t_df'])]={role:threshold(arrays[role].max(axis=1)) for role in ('R1_SCORE_GAP','BL80')}
    rows=[];bootstrap=[];long=[]
    rng=np.random.default_rng(hyp['bootstrap_seed'])
    draws=np.asarray([np.bincount(indices,minlength=512) for indices in rng.integers(0,512,(2000,512))],dtype=np.int32)
    counts=np.vstack([np.ones((1,512),dtype=np.int32),draws])
    for name,spec in scenarios.items():
        arrays,tau,evidence=load_pair('counterexample',name);sources.update(evidence)
        candidates={role:arrays[role] for role in ('R1_SCORE_GAP','BL80')}
        shape=arrays['BL80'].shape;n,no=shape
        time=np.broadcast_to(np.arange(no),(n,no)).ravel()
        target=((np.arange(no)[None,:]>=tau[:,None])&(tau[:,None]>=0)).astype(np.uint8)
        thresholds=calibration[(spec['phi_before'],spec['arch_before'],spec['t_df_before'])]
        scores={};events={}
        for role,p in candidates.items():
            hit=p>thresholds[role];crossed=hit.any(axis=1)
            first=np.where(crossed,hit.argmax(axis=1),-1);changed=tau>=0;null=~changed
            score=ts_auc(target.ravel(),p.ravel(),time) if changed.any() else None
            rows.append({'candidate':role,'scenario':name,'ts_auc':score,'threshold_alpha05':thresholds[role],
                'null_eventual_alarm_rate':float(crossed[null].mean()) if null.any() else None,
                'clean_first_detection_rate':float(((first>=tau)&crossed)[changed].mean()) if changed.any() else None,
                'prebreak_alarm_rate':float(((first>=0)&(first<tau))[changed].mean()) if changed.any() else None,
                'outlier_excess':float(crossed[512:].mean()-crossed[:512].mean()) if name=='O1' else None,
                'pairs':512,'new_fits':0,'synthetic_only':True})
            events[role]=crossed
            if changed.any():
                code=np.repeat(np.tile(np.arange(512),2),no)
                scores[role]=PreparedAUC.create(target.ravel(),p.ravel(),time,code,512).evaluate(counts)
        if scores:
            result=paired_summary(scores['BL80'][:,None],scores['R1_SCORE_GAP'][:,None])
            bootstrap.append({'candidate':'R1_SCORE_GAP','reference':'BL80','scenario':name,**result})
        if name=='O1':
            difference=(events['R1_SCORE_GAP'][512:].astype(float)-events['R1_SCORE_GAP'][:512])-(events['BL80'][512:].astype(float)-events['BL80'][:512])
            vector=counts@difference/512
            atomic_json(out/'OUTLIER_UNCERTAINTY.json',{'status':'PASS','estimate_excess_alarm_delta':float(vector[0]),
                'ci_2_5':float(np.quantile(vector[1:],.025)),'ci_97_5':float(np.quantile(vector[1:],.975)),
                'replicates':2000,'calibration_thresholds_treated_fixed':True},immutable=True)
        progress('Score-gate fixed counterexamples',f'{name}: paired score/alarm diagnostics complete',experiment='R1_SCORE_GAP')
        print('NEXT3 SCORE GATE SYNTHETIC',name,flush=True)
    for name in lock['long_null']['regimes']:
        arrays,tau,evidence=load_pair('long_null',name);sources.update(evidence);spec=scenarios[name]
        thresholds=calibration[(spec['phi_before'],spec['arch_before'],spec['t_df_before'])]
        for role in ('R1_SCORE_GAP','BL80'):
            for horizon in lock['long_null']['horizons']:
                maxima=arrays[role][:,:horizon].max(axis=1)
                long.append({'candidate':role,'background':name,'horizon':horizon,'series':len(maxima),
                    'threshold_alpha05':thresholds[role],'eventual_alarm_rate':float(np.mean(maxima>thresholds[role])),
                    'mean_maximum':float(maxima.mean()),'synthetic_only':True})
        progress('Score-gate fixed long-null diagnostic',name,experiment='R1_SCORE_GAP')
    pd.DataFrame(rows).to_csv(out/'COUNTEREXAMPLES.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(out/'PAIRED_SCORE_BOOTSTRAP.csv',index=False)
    pd.DataFrame(long).to_csv(out/'LONG_NULL.csv',index=False)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'hypothesis_sha256':digest(hyp_path),
        'script_sha256':digest(Path(__file__)),'source_sha256':sources,'new_fits':0,'new_synthetic_base_replays':0,
        'synthetic_score_only_for_diagnostics':True,'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    print('NEXT3 SCORE GATE SYNTHETIC COMPLETE',flush=True)


if __name__=='__main__':main()
