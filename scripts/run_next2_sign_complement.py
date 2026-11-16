"""One registered amplitude-free dependence follow-up under unchanged complement gates."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import sys
import json
import copy
import time
import traceback
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next.registered_extensions import register_extension,registration_hash
from src.next.extensions import extension_hash
from src.next2.sign_dependence import feature_names
from src.next.data import prepare_dev
from scripts.run_next2_research import run_scope,export
from scripts.run_next2_complements import correlation
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    source=json.loads((ROOT/'configs/next2/COMPLEMENT_CANDIDATES_V1.json').read_text())
    c=copy.deepcopy(next(c for c in source['candidates'] if c['experiment_id']=='C205_DEPENDENCE'))
    extension={'kind':'next2_sign_dependence','settings':{'version':'sign_dependence_v1','order':8}}
    registration={'kind':extension['kind'],'module':'next2_sign_dependence'};register_extension(extension,registration)
    c.update(experiment_id='C210_SIGN_DEPENDENCE',parent_id='C205_DEPENDENCE',generation='NEXT2_SIGN_COMPLEMENT_V1',
        hypothesis='Remove amplitude from conditional-normalized lag products to reduce the C205 scale-down counterexample while retaining dependence information.',
        change='Sign conversion before the same four centered lag products; unchanged 24 evidence features and fixed learner.',
        counterexample_target='dependence versus scale decrease and isolated outlier',groups=['NEXT2_SIGN_DEPENDENCE'],
        extension=extension,extension_registration=registration,registration_source_sha256=registration_hash(),
        extension_frozen_hash=extension_hash(extension),keep_names=list(feature_names(extension['settings'])))
    c.pop('next2_registration_sha256')
    identity={'candidate':c,'policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),
        'hypothesis_sha256':sha256(ROOT/'docs/NEXT2_SIGN_DEPENDENCE_HYPOTHESIS.md'),'script_sha256':sha256(Path(__file__)),
        'prior_counterexample_fit_sensitivity_sha256':sha256(ROOT/'artifacts/next2/finalist_fit_sensitivity/MATCHED_FIT_CONTRASTS.csv'),
        'adaptive_nomination':True,'new_candidate_predictions_inspected_before_freeze':False}
    nomination=ROOT/'configs/next2/SIGN_COMPLEMENT_CANDIDATE_V1.json'
    if nomination.exists():assert json.loads(nomination.read_text())['identity']==identity
    else:write_json(nomination,{'created_utc':utc_now(),'identity':identity})
    data=prepare_dev(ROOT);eid=c['experiment_id'];out=ROOT/'artifacts/next2/experiments'/eid;out.mkdir(parents=True,exist_ok=True)
    path=out/'RESULTS.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['candidate']==c
        if old['status']!='BUG':print('NEXT2 SIGN COMPLEMENT RETAINED',old['status'],flush=True);return
        raise RuntimeError('Retain and diagnose the earlier failed attempt before retrying')
    r={'candidate':c,'created_utc':utc_now(),'nomination_sha256':sha256(nomination)};tick=time.perf_counter()
    try:
        screen=run_scope(data,c,'screen');r['screen']=screen;r['correlations']=correlation(eid,'screen',data)
        rankcorr=float(np.mean([v['within_time_rank_correlation'] for v in r['correlations']]))
        if screen['mean']<=.54 or rankcorr>=.95 or not np.isfinite(rankcorr):
            r.update(status='COMPLEMENT_SCREEN_FAIL',reason='Unchanged standalone screen <= .54 or matched rank correlation >= .95')
        else:
            full=run_scope(data,c,'full');r['full']=full;r['delta']=full['mean']-.62163175031248
            fullcorr=correlation(eid,'full',data);r['correlations']+=fullcorr
            rankcorr=float(np.mean([v['within_time_rank_correlation'] for v in fullcorr]));r['full_rank_correlation']=rankcorr
            passed=full['mean']>.54 and rankcorr<.95 and np.isfinite(rankcorr)
            r.update(status='COMPLEMENT_ELIGIBLE_FIXED_BLEND' if passed else 'COMPLEMENT_FULL_FAIL',
                reason='Original standalone information/correlation gate; any fixed blend must independently pass the unchanged full performance gates')
    except Exception as error:
        r.update(status='BUG',reason=type(error).__name__+': '+str(error));write_json(path,r)
        (out/'FAILURE.log').write_text(traceback.format_exc(),encoding='utf-8');raise
    r['elapsed_seconds']=time.perf_counter()-tick;write_json(path,r)
    correlations=[]
    for prior_path in sorted((ROOT/'artifacts/next2/experiments').glob('C*/RESULTS.json')):
        correlations.extend(json.loads(prior_path.read_text()).get('correlations',[]))
    pd.DataFrame(correlations).to_csv(ROOT/'COMPLEMENT_CORRELATION.csv',index=False)
    export();checkpoint('Single sign-dependence hypothesis '+r['status'],['Fixed four blends only if independently eligible','Retain original diagnostic and timing results'])
    print('NEXT2 SIGN COMPLEMENT',r['status'],r.get('full',{}).get('mean'),r.get('full_rank_correlation'),flush=True)


if __name__=='__main__':main()
