"""Freeze exact decomposition and deletion questions, not additional candidates."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    xml=BASE/'maintenance/INFLUENCE_UNIT_TESTS.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==2 and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    names=['src/next4/influence.py','scripts/influence_next4_candidate.py','src/next4/frozen.py','src/next/bootstrap.py','src/scoring/ts_auc.py']
    candidates=['B1_SPLIT_ENERGY_BRANCHES','B2_LOG_ENERGY_SEMITAIL'];path=BASE/'diagnostic_policy/INFLUENCE_POLICY.json'
    atomic_json(path,{'created_utc':utc_now(),'candidates':candidates,'sources_sha256':{name:digest(ROOT/name) for name in names},
        'unit_test_xml_sha256':digest(xml),'scopes':['all_time','SCALE'],
        'deletions':'Exactly every admitted held-out series present in each scope, one at a time, original five-fold average and frozen models.',
        'time_decomposition':'Seven preregistered time bins, exact same-time pair-mass weighted contributions to each fold and the unweighted five-fold mean.',
        'new_fits':0,'replacement_CI':False,'gate_changes':False,'no_influential_series_exclusion_from_official_score':True,
        'stop_rule':'One exact fixed-OOF decomposition and deletion pass per candidate; no trimming, filtering or re-fitting based on influence.'},immutable=True)
    for i,candidate in enumerate(candidates):
        register_job({'id':f'IF{i+1:03d}_'+candidate,'kind':'validation','priority':130,
            'depends_on_success':[f'B{i+301:03d}_FORENSIC'],'argv':['scripts/influence_next4_candidate.py','--candidate',candidate],
            'description':'Exact official time-pair credit and one-series deletion sensitivity; no fits, trimming or new CIs',
            'retry_safe':False,'policy_sha256':digest(path)})
    progress('B1/B2 influence questions preregistered','Exact pair-mass credit and fixed-model deletion sensitivity; no new thresholds or exclusions')
    print('NEXT4 INFLUENCE DIAGNOSTICS QUEUED',flush=True)


if __name__=='__main__':main()
