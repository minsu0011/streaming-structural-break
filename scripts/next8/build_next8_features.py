"""Two bounded workers; immutable 64-series checkpoints; no label access in engine."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
import gc
import numpy as np
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity, note
from src.next8.data import source
from src.next8.features import FeatureState, N_FEATURES
from src.next8.cache import directory, SHARD_SIZE, immutable_npy, immutable_npz


def main(part,parts):
    assert read_json(BASE/'FEATURE_CORE_TEST_GATE_NEXT8.json')['status']=='PASS'
    assert read_json(BASE/'NEXT8_PROTOCOL_LOCK.json')['status']=='FROZEN'
    data=source();out,ident=directory(data)
    out.mkdir(parents=True,exist_ok=True)
    offsets=data.series['online_offsets']
    for shard,start in enumerate(range(0,len(data.ids),SHARD_SIZE)):
        if shard%parts!=part:continue
        stop=min(start+SHARD_SIZE,len(data.ids));p=out/f'shard_{shard:03d}'
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json');assert done['identity']==ident
            for name,sha in done['files_sha256'].items():assert digest(p/name)==sha
            continue
        if p.exists() and any(p.iterdir()):
            raise RuntimeError('Partial shard preserved; explicit repair receipt and new attempt required')
        with activity(f'H-only causal bank shard{shard} series{start}:{stop}', 'FEATURE_GENERATION', 'B7_NORMALIZATION_AND_LINEAGE'):
            p.mkdir(parents=True,exist_ok=False)
            atomic(p/'START.json',{'status':'RUNNING','timestamp':utc_now(),'identity':ident,'start':start,'stop':stop})
            values=[];metadata=[]
            for ix in range(start,stop):
                sid=int(data.ids[ix]);data.guard.admit([sid],purpose='NEXT8 label-free causal feature generation')
                ho=data.series['historical_offsets'];oo=data.series['online_offsets']
                h=data.historical[ho[ix]:ho[ix+1]];o=data.online[oo[ix]:oo[ix+1]]
                # tau and row targets are not requested or passed to this engine.
                engine=FeatureState(h)
                seed_values=engine.history.copy()
                f,_=engine.replay(o)
                assert f.shape==(len(o),N_FEATURES) and np.isfinite(f).all()
                assert engine.calibration_reference_sha256==engine.online_reference_sha256
                values.append(f)
                metadata.append({'dataset_id':sid,'low':engine.low,'high':engine.high,'valid':engine.valid,
                    'epsilon':engine.epsilon,'seed_values':seed_values,'context':engine.context,
                    'center':engine.center,'spread':engine.spread,'fit_length':engine.fit_length,
                    'H_length':len(h),'calibration_clip_count':engine.calibration_clip_count,
                    'online_bound_count':(np.abs(f)>=12).sum(axis=0),
                    'reference_sha256':engine.normalizer_fit_sha256})
            matrix=np.concatenate(values);assert len(matrix)==offsets[stop]-offsets[start]
            immutable_npy(p/'values.npy',matrix)
            immutable_npz(p/'series.npz',**{k:np.asarray([m[k] for m in metadata]) for k in metadata[0]})
            atomic(p/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'identity':ident,
                'start':start,'stop':stop,'first_global_row':int(offsets[start]),'last_global_row_exclusive':int(offsets[stop]),
                'rows':len(matrix),'series':stop-start,'population_fit':False,'label_input':False,
                'files_sha256':{n:digest(p/n) for n in ('values.npy','series.npz')}})
            print(utc_now(),'FEATURE SHARD',shard,'COMPLETE',len(matrix),flush=True)
            del values,matrix,metadata,engine;gc.collect()
    atomic(out/f'PART_{part}_DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'part':part,'parts':parts,'identity':ident})
    note('Feature cache partition complete',f'part{part}/{parts}; independent H-only transforms, no label associations','B7_NORMALIZATION_AND_LINEAGE')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--part',type=int,required=True);parser.add_argument('--parts',type=int,default=2)
    args=parser.parse_args();assert 0<=args.part<args.parts<=2
    main(args.part,args.parts)
