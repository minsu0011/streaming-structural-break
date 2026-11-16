"""Identical fitted methods, opposite series order, uninterrupted scalar replay."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(fold,mode,part):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'REVERSE_ORDER_IMPLEMENTATION_LOCK_V3_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    with target_free_npz() as accesses:
        import numpy as np
        from src.next9.data import FeatureBank,outer_scope
        from src.next9.runtime_loading_v3 import load_heads,load_baseline
        from src.next9.streaming import ResearchStream
        bank=FeatureBank();scope=outer_scope(bank,fold);source=BASE/'live_replay'/scope.name/mode/f'part_{part}'
        prior=read_json(source/'DONE.json');assert prior['status']=='COMPLETE'
        root=BASE/'reverse_order_replay'/scope.name/mode/f'part_{part}'
        if (root/'DONE.json').exists():return
        heads=load_heads(fold,mode);baseline=load_baseline(fold,mode)
        directory=ROOT/read_json(BASE/'PREFLIGHT_NEXT9.json')['RP_cache'];manifest=read_json(directory/'MANIFEST.json')
        for name in ['historical.npy','online.npy']:assert digest(directory/name)==manifest['files_sha256'][name]
        historical=np.load(directory/'historical.npy',mmap_mode='r');online=np.load(directory/'online.npy',mmap_mode='r')
        id_to_index={int(s):k for k,s in enumerate(bank.ids)};records=[]
        for sid in prior['series_ids'][::-1]:
            out=root/f'S{sid}.json'
            if out.exists():records.append(read_json(out));continue
            with activity(f'{scope.name} {mode} series{sid} identical-method opposite-order uninterrupted replay','FULL_RP_ORDER_DETERMINISM','B9_NORMALIZATION_STATE_AVAILABILITY'):
                s=id_to_index[sid];lo,hi=map(int,bank.offsets[s:s+2]);hlo,hhi=map(int,bank.historical_offsets[s:s+2]);scope.admit_validation([sid])
                h=np.asarray(historical[hlo:hhi]);o=np.asarray(online[lo:hi]);ref=source/f'S{sid}'
                receipt=read_json(ref/'DONE.json');assert digest(ref/'PREDICTIONS.npz')==receipt['prediction_sha256']
                with np.load(ref/'PREDICTIONS.npz') as z:
                    rows=z['global_row'];expected={k:z[k] for k in z.files if k!='global_row'}
                stream=ResearchStream(heads,h);base=baseline.new_state(h);counter=0
                for t,p in enumerate(o):
                    result=stream.predict_one(float(p),base.predict_one(float(p)))
                    if mode=='EARLY' and t<400:continue
                    assert rows[counter]==lo+t
                    for name,value in result.items():
                        assert np.float64(value).tobytes()==np.float64(expected[name][counter]).tobytes(),(sid,mode,name,t)
                    counter+=1
                assert counter==len(rows)
                record=dict(status='PASS',timestamp=utc_now(),dataset_id=sid,mode=mode,fold=fold,points=len(o),compared_points=counter,
                    reversed_series_order=True,uninterrupted_vs_saved_restored_bitwise=True,all_methods=list(expected),
                    scalar_bitwise_not_merely_tolerance=True,source_scalar_prediction_sha256=receipt['prediction_sha256'],break_labels_or_tau_used=False)
                atomic(out,record);records.append(record)
        assert not violations
        atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),fold=fold,mode=mode,part=part,
            original_series=len(records),series_ids=[r['dataset_id'] for r in records],points=sum(r['points'] for r in records),
            compared_points=sum(r['compared_points'] for r in records),source_replay_sha256=digest(source/'DONE.json'),
            all_method_opposite_order_bitwise=True,uninterrupted_vs_resumed_bitwise=True,NPZ_members=sorted(set(accesses))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));p.add_argument('--mode',required=True,choices=['NORMAL','EARLY']);p.add_argument('--part',required=True,type=int,choices=[0,1]);a=p.parse_args();main(a.fold,a.mode,a.part)
