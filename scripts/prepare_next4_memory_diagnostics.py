"""Run diagnostic-oracle tests and freeze analysis before synthetic replay starts."""
from pathlib import Path
import sys
import subprocess
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    assert not list((BASE/'synthetic/candidates').glob('*/START.json'))
    xml=BASE/'maintenance/MEMORY_DIAGNOSTIC_UNIT_TESTS.xml'
    assert not xml.exists()
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_next4_memory_diagnostics.py',
        '--junitxml='+str(xml)],cwd=ROOT,check=True)
    suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==4
    assert all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    nominations={p.stem:digest(p) for p in sorted((BASE/'synthetic_nominations').glob('*.json'))}
    assert len(nominations)==12
    path=BASE/'diagnostic_policy/MEMORY_DIAGNOSTIC_POLICY.json'
    names=['src/next4/memory_diagnostics.py','scripts/diagnose_next4_memory.py','src/next4/synthetic.py',
        'src/next2/counterexamples.py']
    atomic_json(path,{'created_utc':utc_now(),'status':'FROZEN_BEFORE_FIRST_NEXT4_FITTED_SYNTHETIC_REPLAY',
        'sources_sha256':{name:digest(ROOT/name) for name in names},'unit_tests_sha256':digest(xml),
        'nominations':nominations,'synthetic_policy_sha256':digest(BASE/'diagnostic_policy/SYNTHETIC_POLICY.json'),
        'outlier_scenarios':{'O1':1,'OB16':16},'post_intervention_window':256,
        'quiet_run_length':32,'positive_excess_tolerance':.01,'null_segment_length':1024,
        'new_fits':0,'new_model_predictions':0,'new_seeds':0,'promotion_authority':False,
        'stop_rule':'Exactly one descriptive pass over completed original synthetic cohorts; no threshold, feature, cohort or learner changes from these diagnostics.'},immutable=True)
    for identifier in nominations:
        register_job({'id':'MD_'+identifier,'kind':'validation','priority':155,
            'depends_on_success':['SYN_'+identifier],
            'argv':['scripts/diagnose_next4_memory.py','--nomination',identifier],
            'description':'Fixed synthetic prediction memory diagnosis: outlier recovery, null segment drift and paired alarm uncertainty',
            'retry_safe':False,'policy_sha256':digest(path)})
    progress('Memory diagnostic definitions frozen','Four independent oracle tests passed; local recovery and null drift separated before fitted synthetic scores')
    print('NEXT4 MEMORY DIAGNOSTICS PREPARED',flush=True)


if __name__=='__main__':main()
