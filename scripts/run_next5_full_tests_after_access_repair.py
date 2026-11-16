"""Complete suite in the separate audit phase, with real-data read denial."""
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import re
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+',args.tag):raise ValueError('Owned audit tag required')
    session=read_json(BASE/'SESSION.json')
    if not session.get('audit_started_utc'):raise RuntimeError('Complete suite belongs to the separate final audit')
    out=BASE/'final_audit'/args.tag/'full_tests';out.mkdir(parents=True,exist_ok=False)
    temporary=(out/'pytest_tmp').resolve()
    assert temporary.is_relative_to((BASE/'final_audit').resolve()) and not temporary.exists()
    os.environ['NEXT5_TEST_AUDIT_DIRECTORY']=str(out);os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache');os.environ['NUMBA_NUM_THREADS']='2'
    atomic_json(out/'START.json',{'created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'source_sha256':digest(Path(__file__)),'access_guard_source_sha256':digest(ROOT/'src/next5/final_test_compatibility.py')},immutable=True)
    progress('Complete final test suite','All project tests; canonical raw and historical per-series arrays forbidden, temporary synthetic fixtures allowed',phase='FINAL_AUDIT',counts_as_research=False)
    import pytest
    status=pytest.main(['-q','tests','-p','src.next5.final_test_compatibility','--junitxml='+str(out/'FULL_TESTS.xml'),
        '--basetemp='+str(temporary),'-o','cache_dir='+str(out/'pytest_cache')])
    tree=ET.parse(out/'FULL_TESTS.xml');cases=tree.findall('.//testcase');failed=tree.findall('.//failure');errors=tree.findall('.//error');skipped=tree.findall('.//skipped')
    guard=read_json(out/'ACCESS_GUARD.json')
    result={'created_utc':utc_now(),'status':'PASS' if status==0 and not failed and not errors and not skipped and guard['status']=='PASS' else 'FAIL',
        'exit_code':int(status),'tests':len(cases),'failures':len(failed),'errors':len(errors),'skipped':len(skipped),
        'access_guard_sha256':digest(out/'ACCESS_GUARD.json'),'test_compatibility_sha256':digest(out/'TEST_COMPATIBILITY.json'),'xml_sha256':digest(out/'FULL_TESTS.xml'),
        'new_real_data_learner_fits':0,'unit_fixture_training_is_not_a_research_candidate_fit':True,'audit_started_utc':session['audit_started_utc']}
    atomic_json(out/'RESULTS.json',result,immutable=True)
    if result['status']!='PASS':raise RuntimeError('Full suite or data-access boundary requires reviewed recovery')
    progress('Complete final test suite passed',str(len(cases))+' tests; no skipped checks or canonical raw/old per-series reads',phase='FINAL_AUDIT',counts_as_research=False)
    print('NEXT5 COMPLETE TEST SUITE PASS',len(cases),flush=True)


if __name__=='__main__':main()
