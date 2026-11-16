"""Freeze a small control budget separately from selection candidates."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    xml=BASE/'maintenance/RUNTIME_CONTROLS_UNIT_TESTS.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==4 and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    sources=['scripts/negative_controls_next4.py','src/next4/negative_controls.py','src/next4/model.py',
        'src/next4/cache.py','src/next4/frozen.py','src/next/bootstrap.py','src/validation/group_bootstrap.py',
        'src/scoring/ts_auc.py','scripts/verify_next2_shared_public_api.py']
    path=BASE/'diagnostic_policy/NEGATIVE_CONTROL_POLICY.json'
    atomic_json(path,{'status':'FROZEN_BEFORE_REAL_CONTROL_FIT','created_utc':utc_now(),
        'registration_sha256':digest(BASE/'registrations/B2_LOG_ENERGY_SEMITAIL.json'),
        'sources_sha256':{name:digest(ROOT/name) for name in sources},'unit_test_xml_sha256':digest(xml),
        'seed':20260908,'real_control_fits':1,'scope':'Original screen 3000 IDs, fold 0, unchanged B2 feature/learner; train labels permuted only within train online-time strata.',
        'constant_control':'Official held-out TS-AUC of a fixed 0.5 output is exactly 0.5.',
        'invalid_future_control':'Explicitly detect future-length, array materialization and one-step lookahead; do not score or nominate an invalid future model.',
        'series_identifier_control':'Unit protocol renames and reverses series without passing identifier to the model. Real reordered-series runtime parity is handled in B2 verification.',
        'paired_replicates':2000,'stop_rule':'One fixed permutation, one fit. No alternate seed, parameter, model or split after scores; no selection or promotion authority.'},immutable=True)
    register_job({'id':'NC001_LIMITED_NEGATIVE_CONTROLS','kind':'validation','priority':120,
        'depends_on_success':['B002_SCREEN'],'argv':['scripts/negative_controls_next4.py'],
        'description':'One training-only label permutation fit, constant output and deliberately invalid future controls',
        'retry_safe':False,'policy_sha256':digest(path)})
    progress('Limited controls preregistered','Exactly one label-permutation fit and deterministic invalid-future/constant controls; no selection candidates')
    print('NEXT4 NEGATIVE CONTROLS QUEUED',flush=True)


if __name__=='__main__':main()
