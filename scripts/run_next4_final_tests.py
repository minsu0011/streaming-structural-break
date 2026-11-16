"""Run the complete suite once against a frozen execution-source snapshot."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress


def snapshot():
    files=set()
    for folder in ('src','scripts','tests','configs'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json','.toml','.ini','.cfg','.ps1') and '__pycache__' not in path.parts:files.add(path)
    files.update(ROOT.glob('*.py'));files.update([ROOT/'pyproject.toml',ROOT/'requirements-lock.txt'])
    return {p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9_]+',args.tag)
    session=read_json(BASE/'SESSION.json');assert session.get('audit_started_utc')
    out=BASE/'final_audit'/args.tag/'full_tests';out.mkdir(parents=True,exist_ok=False);sources=snapshot()
    atomic_json(out/'SOURCE_LOCK.json',{'created_utc':utc_now(),'files_sha256':sources},immutable=True)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'source_lock_sha256':digest(out/'SOURCE_LOCK.json')},immutable=True)
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    with (out/'COLLECTION.log').open('x',encoding='utf-8') as log:
        p=subprocess.run([sys.executable,'-m','pytest','--collect-only','-q','tests'],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
    p.check_returncode();content=(out/'COLLECTION.log').read_text(encoding='utf-8')
    counts=re.findall(r'(?m)^(\d+) tests? collected in ',content);assert len(counts)==1
    count=int(counts[0]);assert count>=1000;assert sources==snapshot()
    progress('Final audit complete pytest suite',str(count)+' collected tests against fixed sources',phase='FINAL_AUDIT',counts_as_research=False)
    with (out/'FULL_TESTS.log').open('x',encoding='utf-8') as log:
        p=subprocess.run([sys.executable,'-m','pytest','-q','tests','--junitxml='+str(out/'FULL_TESTS.xml')],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
    p.check_returncode();assert sources==snapshot();tree=ET.parse(out/'FULL_TESTS.xml')
    cases=tree.findall('.//testcase');assert len(cases)==count
    failures=tree.findall('.//failure');errors=tree.findall('.//error');skips=tree.findall('.//skipped')
    assert not failures and not errors and not skips
    classes={c.attrib.get('classname') for c in cases}
    required={'tests.test_ts_auc','tests.test_scorer_official_checks','tests.test_next_guard',
        'tests.test_next4_robust_finalist_runtime','tests.test_next4_finalization'}
    assert required.issubset(classes)
    seconds=sum(float(s.attrib.get('time',0)) for s in tree.findall('.//testsuite'))
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'collected_tests':count,'executed_tests':len(cases),'failures':0,'errors':0,'skips':0,
        'suite_seconds':seconds,'required_test_classes':sorted(required),'source_files':len(sources),
        'source_set_unchanged_before_after':True,'source_lock_sha256':digest(out/'SOURCE_LOCK.json'),
        'new_real_DEV_learner_fits':0,'prototype_or_mock_fits_are_test_fixtures':True,
        'source_sha256':digest(Path(__file__)),'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    progress('Final audit full test suite passed',str(count)+' cases, no failure/error/skip and unchanged execution sources',phase='FINAL_AUDIT',counts_as_research=False)
    print('NEXT4 FINAL FULL TESTS PASS',count,flush=True)


if __name__=='__main__':main()
