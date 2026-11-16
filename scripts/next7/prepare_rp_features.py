"""Causal G1 outputs for the admitted RP; labels never enter streaming APIs."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE,save,read_json,digest,csv_write,utc_now,activity,require_guard,note
from src.next7.event import CombinedState,COLUMNS
from src.next7.features import from_state,NAMES
from src.next7.calibration import load_calibration,cutoff
from src.next6.data import source


def main():
    guard=require_guard()
    assert read_json(BASE/'G1_CANDIDATE_RESERVATION.json')['status']=='FROZEN'
    calibration=load_calibration()
    lock=read_json(BASE/'G1_IMPLEMENTATION_LOCK.json')
    for path,sha in lock['source_sha256'].items():
        assert digest(ROOT/path)==sha,path
    dest=BASE/'cache/g1'
    if (dest/'MANIFEST.json').exists():
        print('G1 RP feature cache complete; preserve')
        return
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError('Partial G1 cache requires explicit repair receipt; never overwrite')
    dest.mkdir(parents=True,exist_ok=True)
    with activity('causal G1 RP feature extraction','FEATURE_RESEARCH','F3_CONFIRMED_EVENT_MEMORY'):
        data=source()
        count=int(data.series['online_offsets'][-1])
        states=np.lib.format.open_memmap(dest/'state.npy',mode='w+',dtype=np.float64,shape=(count,30))
        features=np.lib.format.open_memmap(dest/'head_features.npy',mode='w+',dtype=np.float64,shape=(count,10))
        hrows=[]
        causal_receipts=[]
        for index,sid in enumerate(data.ids):
            h,o,_=data.get(int(sid))
            # Offline tau returned by the data adapter is deliberately discarded.
            a,b=data.series['online_offsets'][index:index+2]
            cut=cutoff(calibration,len(h),0,1)
            state=CombinedState(h,core_cut=calibration['core_cut'],scale_cut=cut,scale_index=0,rule=1)
            out=state.replay(o)
            assert out.shape==(b-a,30) and np.isfinite(out).all()
            assert not np.any((out[:,3]>0)&(out[:,20]<0))
            assert np.all(np.diff(out[:,3])>=0)
            states[a:b]=out
            features[a:b]=from_state(out)
            hrows.append({'dataset_id':int(sid),'fold':int(data.guard.split.assignment(int(sid))[1]),
                **state.scale.calibration,'scale_cut':cut,
                'core_cut':calibration['core_cut'],'state_mutable_bytes':state.bounded_mutable_state_bytes})
            # Fixed index modulo64, selected before looking at any prediction or label.
            if index%64==0:
                split=min(len(o),max(1,len(o)//2))
                replay=CombinedState(h,core_cut=calibration['core_cut'],scale_cut=cut)
                actual=np.array([replay.predict_one(x) for x in o[:split]])
                np.testing.assert_array_equal(out[:split],actual)
                causal_receipts.append({'dataset_id':int(sid),'prefix_points':split,'scalar_prefix_exact':True})
            if index%800==799:
                states.flush()
                features.flush()
                print(utc_now(),'G1 RP',index+1,'/6400',flush=True)
        states.flush()
        features.flush()
        del states,features
        csv_write('H_CALIBRATION_RP_G1.csv',hrows)
        save('audit/G1_RP_PREFIX_SPOT_CHECKS.json',{'status':'PASS','created_utc':utc_now(),
            'fixed_selection':'sorted RP index modulo64==0','series':len(causal_receipts),'results':causal_receipts})
        receipt={'status':'COMPLETE','created_utc':utc_now(),'series':6400,'rows':count,
            'state_columns':list(COLUMNS),'head_feature_names':list(NAMES),
            'files_sha256':{p.name:digest(p) for p in dest.glob('*.npy')},
            'source_sha256':{n:digest(ROOT/n) for n in ('scripts/next7/prepare_rp_features.py',
                'src/next7/features.py','src/next7/event.py','src/next7/scale.py','src/next7/calibration.py')},
            'calibration_sha256':digest(BASE/'SCALE_CALIBRATION_FROZEN.json'),
            'RP_manifest_sha256':read_json(BASE/'PREFLIGHT_NEXT7.json')['cache_manifest_sha256'],
            'research_split_sha256':guard.split.digest,'tau_or_future_length_passed_to_state':False,
            'outer_score_computed':False,'H_calibration_only_no_online_refitting':True,
            'large_cache_and_full_real_point_outputs_excluded_from_handoff':True}
        save('cache/g1/MANIFEST.json',receipt)
    guard.check_state()
    note('G1 RP causal features ready','Frozen signal and labels kept separate; train small nested correction heads next','F5_CAUSAL_NESTED_INTEGRATION')


if __name__=='__main__':
    main()
