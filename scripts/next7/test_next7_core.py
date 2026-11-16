"""Meaningful causal/state tests plus the frozen NEXT6 extension regressions."""
from pathlib import Path
import os
import subprocess
import sys
import uuid
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE, save, digest, read_json, utc_now, activity, note


def main():
    token = utc_now().replace(':','').replace('.','_')
    env = os.environ.copy()
    env['NUMBA_CACHE_DIR'] = str(BASE/'numba_cache')
    env['NEXT7_TEST_RECEIPT'] = 'CORE_TEST_ACCESS_' + token + '.json'
    paths = ['src/next7/tests/test_engine.py'] + [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_next6_extension_*.py'))]
    with activity('NEXT7 core causal tests and NEXT6 regressions','TEST_GATE','F2_FRESH_CONFIRMATION'):
        logpath = BASE/('audit/CORE_TESTS_'+token+'.log')
        logpath.parent.mkdir(parents=True,exist_ok=True)
        with logpath.open('x',encoding='utf-8') as log:
            proc = subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',
                '-p','src.next7.test_guard',*paths,'--basetemp='+str(BASE/'runtime'/('pytest_'+token)),
                '--junitxml='+str(logpath.with_suffix('.xml'))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        receipt = {'status':'PASS' if proc.returncode==0 else 'FAIL','created_utc':utc_now(),
            'log':str(logpath.relative_to(BASE)),'source_sha256':{p.relative_to(ROOT).as_posix():digest(p)
                for p in (ROOT/'src/next7').rglob('*.py')},
            'tests':paths,'returncode':proc.returncode,'test_gate_before_science':True}
        save('audit/CORE_TESTS_'+token+'.json',receipt)
        save('runtime/CORE_TEST_GATE.json',receipt,False)
        if proc.returncode!=0:
            print(logpath.read_text(encoding='utf-8')[-9000:])
            raise RuntimeError('Core gate failed; no calibration or model experiments allowed')
    note('NEXT7 core test gate PASS','Begin source-locked calibration','F1_SIGNED_SCALE')


if __name__=='__main__':
    main()
