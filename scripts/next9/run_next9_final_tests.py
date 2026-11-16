"""Run current NEXT9 tests with all outputs confined to this run."""
from pathlib import Path
import sys
import subprocess
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,digest,utc_now

def main():
    destination=(BASE/'tests/final_suite_temp').resolve()
    assert destination.is_relative_to((BASE/'tests').resolve()) and not destination.exists()
    xml=BASE/'tests/FINAL_NEXT9_TESTS.xml';log=BASE/'tests/FINAL_NEXT9_TESTS.log'
    assert not xml.exists() and not log.exists()
    command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B','-m','pytest','-c','scripts/next9/pytest.ini',
        'scripts/next9/tests','-q','--basetemp='+str(destination),'--junitxml='+str(xml)]
    with log.open('x',encoding='utf-8') as stream:
        result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0,'Failed test log and XML retained'
    suites=ET.parse(xml).getroot().findall('.//testsuite')
    total=sum(int(s.get('tests',0)) for s in suites);failed=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    skipped=sum(int(s.get('skipped',0)) for s in suites)
    assert total>=81 and failed==0 and skipped==0
    sources={}
    for directory in [ROOT/'src/next9',ROOT/'scripts/next9']:
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ['.py','.ini'] and '__pycache__' not in path.parts:
                if path.suffix=='.py':compile(path.read_text(encoding='utf-8'),str(path),'exec')
                sources[path.relative_to(ROOT).as_posix()]=digest(path)
    atomic(BASE/'tests/FINAL_TEST_SUITE_NEXT9.json',dict(status='PASS',timestamp=utc_now(),tests=total,failures=failed,skipped=skipped,
        command=command,xml_sha256=digest(xml),log_sha256=digest(log),source_sha256=sources,
        cacheprovider_disabled=True,all_pytest_temp_outputs_within_NEXT9=True,old_NEXT8_test_count_not_relabelled_as_new_execution=True,
        full_RP_and_actual_ZIP_runtime_evidence_separate=True,parent_wait_not_credited=True))
    print(total,'NEXT9 tests passed',flush=True)

if __name__=='__main__':main()
