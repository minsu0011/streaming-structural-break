"""Explain a limitation of frozen age correction without fitting or selecting it."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.canonical_model import load
from src.next2.score_envelope import probability_upper_bound,no_alarm_after,mapped_upper
from src.utils.artifacts import write_json,sha256,utc_now
from src.next2.session import checkpoint


def main():
    out=ROOT/'artifacts/next2/score_envelope';out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    paths={r['experiment_id']:ROOT/'artifacts/next2/canonical_deployment/references'/r['experiment_id']/'model.sbnext2' for r in frozen['references'].values()}
    paths['BL2_C205_DEPENDENCE_80']=ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_compact_v1/model.sbnext2'
    thresholds=pd.read_csv(ROOT/'artifacts/next2/age_alarm_diagnostics/LONG_NULL.csv',float_precision='round_trip')
    rows=[];bounds={}
    for eid,path in paths.items():
        model=load(path);bound=probability_upper_bound(model);bounds[eid]={'model_sha256':sha256(path),**bound}
        upper=bound['probability_upper_bound']
        selected=thresholds[(thresholds.candidate==eid)&(thresholds.horizon==16384)]
        for _,r in selected.iterrows():
            cutoff=no_alarm_after(upper,r.threshold,r.beta)
            rows.append({'candidate':eid,'regime':r.regime,'beta':r.beta,'alpha':r.alpha,'threshold':r.threshold,
                'score_upper_bound':upper,'sufficient_no_alarm_after_online_index':cutoff,
                **{'mapped_score_upper_at_'+str(t):mapped_upper(upper,t,r.beta) for t in (4096,16384,49152,65536,100000)},
                'interpretation':'Conservative sufficient bound, not an achievable maximum or an exact detector lifetime.'})
    pd.DataFrame(rows).to_csv(out/'BOUNDS.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'models':bounds,'table_sha256':sha256(out/'BOUNDS.csv'),
        'script_sha256':sha256(Path(__file__)),'source_sha256':sha256(ROOT/'src/next2/score_envelope.py'),
        'threshold_source_sha256':sha256(ROOT/'artifacts/next2/age_alarm_diagnostics/LONG_NULL.csv'),
        'conclusion':'For bounded probabilities below 1, fixed positive age-corrected thresholds eventually cannot fire. Rank preservation does not imply late-alarm sensitivity.',
        'selected_beta':None,'new_fits':0})
    checkpoint('Computed conservative age-map alarm envelope',['Compare locked 65,536-point late-break cohorts','Keep rank and alarm utility distinct'])
    print('NEXT2 SCORE ENVELOPES',bounds,flush=True)


if __name__=='__main__':
    main()
