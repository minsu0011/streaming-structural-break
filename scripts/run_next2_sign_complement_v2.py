"""Repair only the empty name selector; preserve the zero-fit C210 adapter failure."""
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
from src.next.registered_extensions import register_extension
from src.next.extensions import names_and_groups
from src.next.data import prepare_dev
from scripts.run_next2_research import run_scope,export
from scripts.run_next2_complements import correlation
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    original_path=ROOT/'configs/next2/SIGN_COMPLEMENT_CANDIDATE_V1.json'
    original=json.loads(original_path.read_text())['identity'];c=copy.deepcopy(original['candidate'])
    failure=ROOT/'artifacts/next2/experiments/C210_SIGN_DEPENDENCE'
    assert json.loads((failure/'RESULTS.json').read_text())['status']=='BUG'
    assert not list(failure.glob('*/*.joblib')) and not list(failure.glob('*/fold_?.npy'))
    register_extension(c['extension'],c['extension_registration'])
    canonical,groups=names_and_groups(c['extension'])
    assert len(canonical)==24 and not set(canonical).intersection(c['keep_names'])
    c.update(experiment_id='C211_SIGN_DEPENDENCE',keep_names=canonical,
        adapter_repair_of='C210_SIGN_DEPENDENCE',adapter_repair='Use extension-prefixed canonical names; feature mathematics and all gate settings unchanged.')
    identity={'candidate':c,'policy_sha256':original['policy_sha256'],'original_nomination_sha256':sha256(original_path),
        'original_failure_sha256':sha256(failure/'RESULTS.json'),'original_script_sha256':sha256(ROOT/'scripts/run_next2_sign_complement.py'),
        'script_sha256':sha256(Path(__file__)),'hypothesis_sha256':original['hypothesis_sha256'],'prior_training_events':0,
        'prior_candidate_predictions_available':False,'new_statistical_hypotheses':0,'repair_scope':'Canonical feature-name selection only'}
    nomination=ROOT/'configs/next2/SIGN_COMPLEMENT_CANDIDATE_V2.json'
    if nomination.exists():assert json.loads(nomination.read_text())['identity']==identity
    else:write_json(nomination,{'created_utc':utc_now(),'identity':identity})
    write_json(failure/'ADAPTER_DIAGNOSIS.json',{'status':'DIAGNOSED_BEFORE_ANY_FIT','created_utc':utc_now(),
        'cause':'names_and_groups prepends extension kind; raw module feature_names were mistakenly used as keep_names.',
        'selected_columns':0,'completed_fits':0,'candidate_predictions':0,'replacement':'C211_SIGN_DEPENDENCE',
        'preserved_original_result_sha256':sha256(failure/'RESULTS.json'),'repair_nomination_sha256':sha256(nomination)})
    eid=c['experiment_id'];out=ROOT/'artifacts/next2/experiments'/eid;out.mkdir(parents=True,exist_ok=True);path=out/'RESULTS.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['candidate']==c
        if old['status']!='BUG':print('NEXT2 SIGN REPAIRED RETAINED',old['status'],flush=True);return
        raise RuntimeError('Do not retry an unexplained failure')
    data=prepare_dev(ROOT);r={'candidate':c,'created_utc':utc_now(),'nomination_sha256':sha256(nomination)};tick=time.perf_counter()
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
                reason='Original standalone information/correlation gate; fixed blends still require unchanged primary gates')
    except Exception as error:
        r.update(status='BUG',reason=type(error).__name__+': '+str(error));write_json(path,r)
        (out/'FAILURE.log').write_text(traceback.format_exc(),encoding='utf-8');raise
    r['elapsed_seconds']=time.perf_counter()-tick;write_json(path,r)
    correlations=[]
    for prior in sorted((ROOT/'artifacts/next2/experiments').glob('C*/RESULTS.json')):
        correlations.extend(json.loads(prior.read_text()).get('correlations',[]))
    pd.DataFrame(correlations).to_csv(ROOT/'COMPLEMENT_CORRELATION.csv',index=False)
    export();checkpoint('Repaired sign-dependence '+r['status'],['Evaluate only admitted four fixed blends','Retain the zero-fit adapter failure'])
    print('NEXT2 SIGN REPAIRED',r['status'],r.get('full',{}).get('mean'),r.get('full_rank_correlation'),flush=True)


if __name__=='__main__':main()
