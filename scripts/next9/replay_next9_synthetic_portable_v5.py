"""Fixed synthetic all-model fixture, or fresh extracted-payload reproduction.

No competition data are needed. Model training and candidate discovery are
never performed here. A verification run consumes cases in reversed order.
"""
from pathlib import Path
import sys
import os
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'nc') if os.environ.get('NEXT9_PORTABLE_REPLAY')=='1' else os.environ['NUMBA_CACHE_DIR']
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

def main(fold,mode,verify,original):
    if verify:
        if not original:raise ValueError('Fresh payload verification requires original workspace denial')
        from src.next8.rawfree_boundary import install
        denied,opened=install(ROOT,original=original)
        assert read_json(BASE/'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json')['status']=='PASS'
    else:
        from src.next9.boundary import install_workspace_boundary
        denied=install_workspace_boundary();opened=set()
    import numpy as np
    import pickle
    import copy
    import hashlib
    from src.next9.negative_controls_v2 import BACKGROUNDS,INTERVENTIONS,SEEDS,H_LENGTHS,generate
    from src.next9.runtime_loading_v5 import load_heads,load_baseline
    from src.next9.streaming import ResearchStream
    from src.next9.integrated_selected import IntegratedStream
    from src.next5.runtime import ScalarDetector
    from src.next9.io import save_npz
    heads=load_heads(fold,mode);baseline=load_baseline(fold,mode)
    static_before=hashlib.sha256(pickle.dumps(heads,protocol=5)).hexdigest()
    expected_root=BASE/'portable_fixture'/f'outer_{fold}'/mode
    root=BASE/('portable_verification' if verify else 'portable_fixture')/f'outer_{fold}'/mode
    if (root/'DONE.json').exists():raise FileExistsError('Do not credit a no-op rerun')
    cases=[(b,i,s,h) for b in BACKGROUNDS for i in INTERVENTIONS for s in SEEDS for h in H_LENGTHS]
    if verify:cases=cases[::-1]
    records=[]
    for background,intervention,seed,h_length in cases:
        name=f'{background}_{intervention}_{seed}_H{h_length}'
        out=root/name
        if (out/'DONE.json').exists():
            done=read_json(out/'DONE.json')
            if not verify:assert digest(out/'PREDICTIONS.npz')==done['prediction_sha256']
            records.append(done);continue
        kind='FINAL_AUDIT' if verify else 'PORTABLE_FIXTURE_RESEARCH'
        with activity(f'Synthetic raw-free {fold} {mode} {name} '+('fresh ZIP verification' if verify else 'prospective fixture'),kind,'B9_NORMALIZATION_STATE_AVAILABILITY'):
            h,o,metadata=generate(background,intervention,seed,h_length)
            stream=ResearchStream(heads,h);base=baseline.new_state(h);integrated=IntegratedStream(heads,baseline,h)
            output={};dedicated=[];snapshot_bytes=0
            for t,point in enumerate(o):
                prediction=stream.predict_one(float(point),base.predict_one(float(point)))
                dedicated.append(integrated.predict_one(float(point)))
                for method,value in prediction.items():output.setdefault(method,[]).append(value)
                if not verify and t==73:
                    state=pickle.dumps((stream.features,stream.bank,copy.deepcopy(base.args)),protocol=5)
                    snapshot_bytes=len(state)
                    stream.features,stream.bank,args=pickle.loads(state)
                    base=ScalarDetector(base.kernel,args,base.prepared)
                    integrated=IntegratedStream.restore(pickle.loads(pickle.dumps(integrated.snapshot(),protocol=5)))
            output={k:np.asarray(v,np.float64) for k,v in output.items()}
            for values in output.values():assert np.isfinite(values).all() and np.all((values>=0)&(values<=1))
            np.testing.assert_array_equal(output['M7'],dedicated)
            if heads.correction is None:np.testing.assert_array_equal(output['M7'],output['M0'])
            prefix=IntegratedStream(heads,baseline,h)
            changed=np.r_[o[:31],-8.*o[31:]+1.,np.array([999.,-999.])]
            prefix_prediction=np.asarray([prefix.predict_one(float(x)) for x in changed[:31]])
            np.testing.assert_array_equal(prefix_prediction,output['M7'][:31])
            if verify:
                expected=expected_root/name
                frozen=read_json(expected/'DONE.json');assert digest(expected/'PREDICTIONS.npz')==frozen['prediction_sha256']
                with np.load(expected/'PREDICTIONS.npz') as z:
                    assert set(z.files)==set(output)
                    for method,values in output.items():np.testing.assert_array_equal(values,z[method])
                sha=frozen['prediction_sha256']
            else:
                save_npz(out/'PREDICTIONS.npz',**output);sha=digest(out/'PREDICTIONS.npz')
            done=dict(status='PASS' if verify else 'COMPLETE',timestamp=utc_now(),fold=fold,mode=mode,
                synthetic_metadata=metadata,points=len(o),methods=list(output),prediction_sha256=sha,
                actual_scalar_execution=True,dedicated_M7_exact=True,M7_fallback_exact=heads.correction is None,
                altered_extended_suffix_prefix31_exact=True,mutable_snapshot_bytes=snapshot_bytes,
                save_restore_at74=not verify,reverse_case_order=verify,uninterrupted_second_execution=verify,
                exact_against_saved_fixture=verify,real_observations_or_labels_read=False)
            atomic(out/'DONE.json',done);records.append(done)
    static_after=hashlib.sha256(pickle.dumps(heads,protocol=5)).hexdigest()
    assert static_before==static_after and not denied
    atomic(root/'DONE.json',dict(status='PASS' if verify else 'COMPLETE',timestamp=utc_now(),fold=fold,mode=mode,
        cases=len(records),points=sum(r['points'] for r in records),methods=records[0]['methods'],
        model_pickle_sha256=static_after,no_fit_or_model_update=True,no_real_data=True,
        actual_fresh_payload=verify,reverse_order_exact=verify,opened_payload_files=sorted(opened),denied=denied,
        case_receipts_sha256={str((root/(f'{r["synthetic_metadata"]["background"]}_{r["synthetic_metadata"]["intervention"]}_{r["synthetic_metadata"]["seed"]}_H{r["synthetic_metadata"]["H_length"]}')/'DONE.json').relative_to(root)):digest(root/(f'{r["synthetic_metadata"]["background"]}_{r["synthetic_metadata"]["intervention"]}_{r["synthetic_metadata"]["seed"]}_H{r["synthetic_metadata"]["H_length"]}')/'DONE.json') for r in records}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--mode',choices=['NORMAL','EARLY'],required=True);p.add_argument('--verify',action='store_true');p.add_argument('--original');a=p.parse_args();main(a.fold,a.mode,a.verify,a.original)
