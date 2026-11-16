"""First fixed robustness gate: two original ARCH variance-decrease scenarios."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.gate_runtime import replay_gate_features
from src.next3.gate_serialization import descriptor_predict
from src.next2.shared_inference import load_artifact
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
from src.next2.counterexamples import generate,policy
from src.next2.blend import combine
from src.scoring.ts_auc import ts_auc
from src.next.bootstrap import PreparedAUC,paired_summary
from numba import set_num_threads


def main():
    set_num_threads(2)
    hyp=BASE/'hypotheses/R3_DEPTH2_GATE_SYNTHETIC.json';registration=read_json(hyp)
    for name,sha in registration['sources_sha256'].items():assert digest(ROOT/name)==sha
    fitted=BASE/'refits/R3_DEPTH2_GATE/research_v1';receipt=read_json(fitted/'RESULTS.json')
    assert receipt['status']=='PASS' and digest(fitted/'gate.json')==receipt['files_sha256']['gate.json']
    gate=read_json(fitted/'gate.json')
    model,_=load_artifact(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1')
    prepared=PreparedSharedBlendPredictor(model,resident=False)
    out=BASE/'counterexamples/R3_DEPTH2_GATE/arch_gate';out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULTS.json').exists():raise RuntimeError('Preserve complete ARCH diagnostic')
    lock=policy();n=512;no=lock['online_length'];rows=[];bootstrap=[];files={}
    rng=np.random.default_rng(registration['bootstrap_seed'])
    draws=np.asarray([np.bincount(row,minlength=n) for row in rng.integers(0,n,(2000,n))],dtype=np.int32)
    counts=np.vstack([np.ones((1,n),dtype=np.int32),draws])
    for scenario in ('V3','V4'):
        old=ROOT/'artifacts/next2/finalist_counterexamples/counterexample'/scenario
        oldreceipt=read_json(old/'RESULTS.json')
        oldpred=old/'BL2_C205_DEPENDENCE_80.npy';oldtau=old/'synthetic_tau.npy'
        assert digest(oldpred)==oldreceipt['files_sha256'][oldpred.name]
        assert digest(oldtau)==oldreceipt['files_sha256'][oldtau.name]
        ref=np.load(oldpred,allow_pickle=False);tau=np.load(oldtau,allow_pickle=False)
        predictions=np.empty_like(ref);prefix_checks=0;frozen_parity=0
        for seed in range(n):
            for control,index in ((True,seed),(False,n+seed)):
                h,o,actual_tau=generate(scenario,seed,control=control,lock=lock)
                assert actual_tau==tau[index]
                x=replay_gate_features(o,prepared.new_state(h).call_args)
                np.testing.assert_array_equal(combine(x[:,0],x[:,1],.8),ref[index]);frozen_parity+=len(o)
                predictions[index]=descriptor_predict(gate,x)
            np.testing.assert_array_equal(predictions[seed,:tau[n+seed]],predictions[n+seed,:tau[n+seed]])
            prefix_checks+=1
            if (seed+1)%64==0:
                progress('Depth2 robustness ARCH replay',f'{scenario}: {seed+1}/512 paired seeds; frozen score exact',experiment='R3_DEPTH2_GATE')
                print('NEXT3 R3 ARCH',scenario,seed+1,flush=True)
        time=np.broadcast_to(np.arange(no),predictions.shape).ravel()
        y=((np.arange(no)[None,:]>=tau[:,None])&(tau[:,None]>=0)).astype(np.uint8).ravel()
        code=np.repeat(np.tile(np.arange(n),2),no)
        a=PreparedAUC.create(y,ref.ravel(),time,code,n).evaluate(counts)
        b=PreparedAUC.create(y,predictions.ravel(),time,code,n).evaluate(counts)
        score=ts_auc(y,predictions.ravel(),time);reference=ts_auc(y,ref.ravel(),time)
        np.testing.assert_allclose([score,reference],[b[0],a[0]],rtol=0,atol=1e-12)
        row={'candidate':'R3_DEPTH2_GATE','scenario':scenario,'ts_auc':score,'reference_ts_auc':reference,
            'delta':score-reference,'threshold':.01,'pass':bool(score-reference>=.01),'pairs':n,
            'prefix_checks':prefix_checks,'frozen_score_exact_points':frozen_parity,'synthetic_only':True}
        rows.append(row);bootstrap.append({'candidate':'R3_DEPTH2_GATE','reference':'BL80','scenario':scenario,**paired_summary(a[:,None],b[:,None])})
        np.save(out/f'{scenario}_predictions.npy',predictions,allow_pickle=False)
        np.save(out/f'{scenario}_bootstrap.npy',np.column_stack([a,b]),allow_pickle=False)
        for name in (f'{scenario}_predictions.npy',f'{scenario}_bootstrap.npy'):files[name]=digest(out/name)
        print('NEXT3 R3 ARCH RESULT',row,flush=True)
    pd.DataFrame(rows).to_csv(out/'COUNTEREXAMPLES.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(out/'PAIRED_SCORE_BOOTSTRAP.csv',index=False)
    files.update({name:digest(out/name) for name in ('COUNTEREXAMPLES.csv','PAIRED_SCORE_BOOTSTRAP.csv')})
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'hypothesis_sha256':digest(hyp),
        'new_fits':0,'arch_gate_pass':all(r['pass'] for r in rows),'scenarios':rows,'files_sha256':files,
        'disposition':'CONTINUE_ROBUSTNESS_VALIDATION' if all(r['pass'] for r in rows) else 'NEGATIVE_RESULT_FIXED_ARCH_GATE',
        'no_alarm_calibration_here':True},immutable=True)


if __name__=='__main__':main()
