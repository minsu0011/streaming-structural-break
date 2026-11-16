"""Complement-specific admission using standalone information and correlation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
from pathlib import Path
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from scripts.run_next2_research import run_scope,export
from scripts.forensic_next_candidate import rank_within_time
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def correlation(eid,scope,data):
    root=ROOT/'artifacts/next2/experiments'/eid/scope
    ref=ROOT/'artifacts/next/experiments/J102_SERIAL_VARIANCE_512'/scope
    rows=[]
    metadata=data.rows()
    for fold in ([0,4] if scope=='screen' else range(5)):
        coords=np.load(root/f'fold_{fold}_global_rows.npy',allow_pickle=False)
        expected=np.load(ref/f'fold_{fold}_global_rows.npy',allow_pickle=False)
        np.testing.assert_array_equal(coords,expected)
        receipt=json.loads((ref/f'fold_{fold}.json').read_text(encoding='utf-8'))
        for name,digest in receipt['files_sha256'].items():
            assert sha256(ref/name)==digest
        p=np.load(root/f'fold_{fold}.npy',allow_pickle=False)
        r=np.load(ref/f'fold_{fold}.npy',allow_pickle=False)
        t=metadata['time_online'][coords]
        a,b=rank_within_time(p,t),rank_within_time(r,t)
        rows.append({'candidate':eid,'scope':scope,'fold':fold,'pearson':float(np.corrcoef(p,r)[0,1]),
            'within_time_rank_correlation':float(np.corrcoef(a,b)[0,1])})
    return rows


def main():
    queue=json.loads((ROOT/'configs/next2/COMPLEMENT_CANDIDATES_V1.json').read_text(encoding='utf-8'))
    if queue['policy_sha256']!=sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'):
        raise RuntimeError('Complement gate policy changed')
    data=prepare_dev(ROOT)
    all_correlations=[]
    for c in queue['candidates']:
        eid=c['experiment_id']
        out=ROOT/'artifacts/next2/experiments'/eid
        out.mkdir(parents=True,exist_ok=True)
        path=out/'RESULTS.json'
        if path.exists():
            previous=json.loads(path.read_text(encoding='utf-8'))
            assert previous['candidate']==c
            if previous['status']!='BUG':
                all_correlations.extend(previous.get('correlations',[]))
                print('COMPLEMENT RETAINED',eid,previous['status'],flush=True)
                continue
        r={'candidate':c,'status':'BUG','reason':'incomplete','created_utc':utc_now()}
        tick=time.perf_counter()
        try:
            screen=run_scope(data,c,'screen')
            r['screen']=screen
            correlations=correlation(eid,'screen',data)
            rankcorr=float(np.mean([v['within_time_rank_correlation'] for v in correlations]))
            r['correlations']=correlations
            if screen['mean']<=.54 or rankcorr>=.95 or not np.isfinite(rankcorr):
                r.update(status='COMPLEMENT_SCREEN_FAIL',reason='Standalone screen <= .54 or matched within-time rank correlation >= .95')
            else:
                full=run_scope(data,c,'full')
                r['full']=full
                r['delta']=full['mean']-.62163175031248
                fullcorr=correlation(eid,'full',data)
                r['correlations']+=fullcorr
                rankcorr=float(np.mean([v['within_time_rank_correlation'] for v in fullcorr]))
                r['full_rank_correlation']=rankcorr
                eligible=full['mean']>.54 and rankcorr<.95 and np.isfinite(rankcorr)
                r.update(status='COMPLEMENT_ELIGIBLE_FIXED_BLEND' if eligible else 'COMPLEMENT_FULL_FAIL',
                    reason='Standalone information and correlation criteria passed; fixed blends require independent full gates' if eligible else 'Full standalone information/correlation gate failed')
        except Exception as exc:
            r.update(status='BUG',reason=type(exc).__name__+': '+str(exc))
            (out/'FAILURE.log').write_text(traceback.format_exc(),encoding='utf-8')
        r['elapsed_seconds']=time.perf_counter()-tick
        write_json(path,r)
        all_correlations.extend(r.get('correlations',[]))
        pd.DataFrame(all_correlations).to_csv(ROOT/'COMPLEMENT_CORRELATION.csv',index=False)
        export()
        checkpoint('Standalone complement '+eid,['Finish bounded complement menu','Evaluate eligible fixed-weight blends'],
            artifacts=[path.relative_to(ROOT).as_posix()])
        print('COMPLEMENT RESULT',eid,r['status'],r.get('full',{}).get('mean'),r.get('full_rank_correlation'),flush=True)


if __name__=='__main__':
    main()
