"""Final audit of the dedicated M7 production path on every admitted RP series."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz

def main(fold,mode,part):
    assert read_json(BASE/'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json')['status']=='PASS'
    violations=install_workspace_boundary()
    with target_free_npz() as accesses:
        import numpy as np
        import pickle
        from src.next9.runtime_loading_v3 import load_heads,load_baseline
        from src.next9.streaming import IntegratedStream
        from src.next9.data import FeatureBank
        bank=FeatureBank();heads=load_heads(fold,mode);baseline=load_baseline(fold,mode)
        pre=read_json(BASE/'PREFLIGHT_NEXT9.json');directory=ROOT/pre['RP_cache'];manifest=read_json(directory/'MANIFEST.json')
        for name in ['historical.npy','online.npy']:assert digest(directory/name)==manifest['files_sha256'][name]
        historical=np.load(directory/'historical.npy',mmap_mode='r');online=np.load(directory/'online.npy',mmap_mode='r')
        indices=np.flatnonzero(bank.folds==fold)[part::2][::-1]
        root=BASE/'audit/integrated_RP'/f'outer_{fold}'/mode/f'part_{part}'
        if (root/'DONE.json').exists():raise FileExistsError('Already audited; no redundant credit')
        records=[]
        for s in indices:
            sid=int(bank.ids[s]);out=root/f'S{sid}.json'
            if out.exists():records.append(read_json(out));continue
            with activity(f'Dedicated M7 final raw RP audit fold{fold} {mode} series{sid}','FINAL_AUDIT','B9_NORMALIZATION_STATE_AVAILABILITY'):
                lo,hi=map(int,bank.offsets[s:s+2]);hlo,hhi=map(int,bank.historical_offsets[s:s+2])
                h=np.asarray(historical[hlo:hhi]);o=np.asarray(online[lo:hi]);state=IntegratedStream(heads,baseline,h)
                source=BASE/'live_replay'/f'outer_{fold}'/mode/f'part_{part}'/f'S{sid}'
                frozen=read_json(source/'DONE.json');assert digest(source/'PREDICTIONS.npz')==frozen['prediction_sha256']
                with np.load(source/'PREDICTIONS.npz') as z:expected=z['M7'];expected_rows=z['global_row']
                values=[]
                for t,point in enumerate(o):
                    value=state.predict_one(float(point))
                    if mode=='NORMAL' or t>=400:values.append(value)
                    if t==30:state=IntegratedStream.restore(pickle.loads(pickle.dumps(state.snapshot(),protocol=5)))
                values=np.asarray(values,np.float64);np.testing.assert_array_equal(values,expected)
                r=dict(status='PASS',timestamp=utc_now(),dataset_id=sid,fold=fold,mode=mode,points=len(o),compared_points=len(values),
                    dedicated_M7_vs_all_model_scalar_exact=True,save_restore_at31=True,M7_exact_BL_bypass=heads.correction is None,
                    source_scalar_prediction_sha256=frozen['prediction_sha256'],no_labels_or_future_length=True)
                atomic(out,r);records.append(r)
        assert not violations
        atomic(root/'DONE.json',dict(status='PASS',timestamp=utc_now(),fold=fold,mode=mode,part=part,series=len(records),
            series_ids=[r['dataset_id'] for r in records],points=sum(r['points'] for r in records),compared_points=sum(r['compared_points'] for r in records),
            all_dedicated_M7_bitwise=True,save_restore_dedicated_state_at31=True,NPZ_members=sorted(set(accesses))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5));p.add_argument('--mode',required=True,choices=['NORMAL','EARLY']);p.add_argument('--part',type=int,required=True,choices=[0,1]);a=p.parse_args();main(a.fold,a.mode,a.part)
