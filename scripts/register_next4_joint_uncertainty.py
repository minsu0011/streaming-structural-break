"""Freeze the closed full-candidate snapshot before any joint max calculation."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    xml=BASE/'maintenance/JOINT_UNCERTAINTY_UNIT_TESTS_CORRECTED.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==2 and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    candidates=[];deps=[]
    for path in sorted((BASE/'experiments').glob('*/full/RESULTS.json')):
        full=read_json(path);assert full['status']=='COMPLETE';candidate=full['candidate'];candidates.append(candidate)
        menu=read_json(BASE/'families'/(full['family']+'_MENU_LOCK.json'));idx=menu['candidate_ids'].index(candidate)
        deps.append(f'{full["family"]}{idx+201:03d}_BOOTSTRAP')
    assert len(candidates)==14
    names=['src/next4/joint_uncertainty.py','scripts/joint_uncertainty_next4.py','src/next/bootstrap.py']
    path=BASE/'diagnostic_policy/JOINT_UNCERTAINTY_POLICY.json'
    atomic_json(path,{'created_utc':utc_now(),'candidates':candidates,'sources_sha256':{name:digest(ROOT/name) for name in names},
        'original_research_policy_sha256':digest(ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json'),'unit_test_xml_sha256':digest(xml),
        'new_fits':0,'new_draws':0,'statistic':'For each candidate: bootstrap (delta_star - observed_delta) / bootstrap standard deviation. Empirical 95th percentile of replicate-wise maximum across the fixed candidate list.',
        'report':'Observed minus common critical value times its bootstrap SD; empirical maximum tail fraction; best-score frequency; correlation participation ratio.',
        'limits':'Post-result diagnostic, fixed fitted OOF only. No full adaptive-selection/FWER claim, no stepdown implementation, no replacement of original two-sided 95% CI gates.',
        'method_context_source':'https://www.econ.uzh.ch/dam/jcr%3Affffffff-935a-b0d6-ffff-ffffd823d949/jasa.pdf',
        'stop_rule':'Exactly this 14-candidate snapshot and the original 2000 paired draws. No seed, draw-count, centering, confidence-level or candidate-subset variants after the joint calculation.'},immutable=True)
    register_job({'id':'JU001_CONDITIONAL_JOINT_UNCERTAINTY','kind':'validation','priority':128,'depends_on_success':deps,
        'argv':['scripts/joint_uncertainty_next4.py'],'description':'Joint dependence and centered-bootstrap sensitivity of the fixed full-candidate menu, no new draws',
        'retry_safe':False,'policy_sha256':digest(path)})
    progress('Conditional joint uncertainty preregistered','14 closed full recipes; reused 2000 shared paired draws and explicit limits on selection claims')
    print('NEXT4 JOINT UNCERTAINTY QUEUED',flush=True)


if __name__=='__main__':main()
