"""Assemble completed label-free shards once, binding all file identities."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
import os
import numpy as np
from src.next8.common import BASE, atomic, read_json, digest, save, utc_now, activity
from src.next8.data import source
from src.next8.cache import directory, SHARD_SIZE, immutable_npz
from src.next8.features import N_FEATURES, NAMES
from src.next5.data import admitted_npz


def main():
    data=source();out,ident=directory(data)
    for part in range(2):assert read_json(out/f'PART_{part}_DONE.json')['status']=='COMPLETE'
    assert not (out/'MANIFEST.json').exists()
    with activity('Assemble and hash all RP H-only feature shards','FEATURE_GENERATION','B7_NORMALIZATION_AND_LINEAGE'):
        target=out/'values.npy';tmp=out/'values.assembling.npy'
        if target.exists() or tmp.exists():raise RuntimeError('Preserve partial assembly; explicit repair required')
        total=int(data.series['online_offsets'][-1]);x=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(total,N_FEATURES))
        parts=[];receipts={}
        for shard,start in enumerate(range(0,len(data.ids),SHARD_SIZE)):
            p=out/f'shard_{shard:03d}';done=read_json(p/'DONE.json')
            assert done['status']=='COMPLETE' and done['identity']==ident
            for name,sha in done['files_sha256'].items():assert digest(p/name)==sha
            a=admitted_npz(p/'series.npz',data.guard,'NEXT8 feature shard assembly')
            stop=min(start+SHARD_SIZE,len(data.ids));np.testing.assert_array_equal(a['dataset_id'],data.ids[start:stop])
            z=np.load(p/'values.npy',mmap_mode='r',allow_pickle=False)
            lo,hi=map(int,data.series['online_offsets'][[start,stop]])
            assert (lo,hi)==(done['first_global_row'],done['last_global_row_exclusive'])
            x[lo:hi]=z;parts.append(a);receipts[f'shard_{shard:03d}/DONE.json']=digest(p/'DONE.json')
        x.flush();del x;os.rename(tmp,target)
        immutable_npz(out/'series.npz',**{k:np.concatenate([a[k] for a in parts]) for k in parts[0]})
        atomic(out/'MANIFEST.json',{'status':'COMPLETE','timestamp':utc_now(),'identity':ident,
            'rows':total,'series':len(data.ids),'features':list(NAMES),'shards_sha256':receipts,
            'files_sha256':{n:digest(out/n) for n in ('values.npy','series.npz')}})
        save('FEATURE_CACHE_NEXT8.json',{'status':'PASS','timestamp':utc_now(),'directory':out.relative_to(ROOT).as_posix(),
             'manifest_sha256':digest(out/'MANIFEST.json'),'row_count':total,'feature_count':N_FEATURES,
             'normalization_population_fit':False,'label_association_scores_computed':False})


if __name__=='__main__':main()
