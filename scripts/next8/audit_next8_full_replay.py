"""Final-audit-only full RP replay, two disjoint 64-series shard workers."""
from pathlib import Path
import sys
import time
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,digest,atomic,utc_now,activity,note
from src.next8.runtime_accounting import audit_admitted
from src.next8.cache import FeatureCache
from src.next8.full_replay_audit import load_audit_models,audit_series


def main(part):
    admission=read_json(BASE/'runtime/FINAL_AUDIT_ADMISSION.json');heartbeat=read_json(BASE/'runtime/HEARTBEAT.json')
    assert audit_admitted(admission,heartbeat['productive_seconds'],digest(BASE/'SESSION_NEXT8.json'),36000)
    lock=read_json(BASE/'FULL_REPLAY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    directory=BASE/'full_replay_audit';directory.mkdir(parents=True,exist_ok=True)
    with activity(f'Full fresh RP audit load part{part}','FINAL_AUDIT','B7_NORMALIZATION_AND_LINEAGE'):
        cache=FeatureCache(verify_values=True);bundles,matrices=load_audit_models(cache)
    records=[];started=time.monotonic()
    for shard,start in enumerate(range(0,len(cache.data.ids),64)):
        if shard%2!=part:continue
        path=directory/f'SHARD_{shard:03d}.json'
        if path.exists():
            receipt=read_json(path);assert receipt['status']=='PASS' and receipt['implementation_lock_sha256']==digest(BASE/'FULL_REPLAY_IMPLEMENTATION_LOCK_NEXT8.json')
        else:
            with activity(f'Full fresh H/features/heldout models shard{shard}','FINAL_AUDIT','B7_NORMALIZATION_AND_LINEAGE'):
                results=[audit_series(cache,int(sid),bundles,matrices) for sid in cache.data.ids[start:start+64]]
                receipt={'status':'PASS','timestamp':utc_now(),'shard':shard,'part':part,'start':start,'stop':min(start+64,len(cache.data.ids)),
                    'series':len(results),'points':sum(r['online_points'] for r in results),'results':results,
                    'implementation_lock_sha256':digest(BASE/'FULL_REPLAY_IMPLEMENTATION_LOCK_NEXT8.json'),
                    'admission_sha256':digest(BASE/'runtime/FINAL_AUDIT_ADMISSION.json')}
                atomic(path,receipt)
        records.append({'shard':shard,'path':path.name,'sha256':digest(path),'series':receipt['series'],'points':receipt['points']})
    atomic(directory/f'PART_{part}_DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'part':part,'parts':2,'shards':records,
        'series':sum(r['series'] for r in records),'points':sum(r['points'] for r in records),'elapsed_seconds':time.monotonic()-started,
        'time_accounting':'Parent scheduler counts observed work union; part durations are not summed research/audit hours'})
    note('Complete full fresh RP replay part'+str(part),'All153 scalar bank and original full-H references; no model refits','B7_NORMALIZATION_AND_LINEAGE',phase='FINAL_AUDIT')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,choices=(0,1),required=True)
    main(p.parse_args().part)
