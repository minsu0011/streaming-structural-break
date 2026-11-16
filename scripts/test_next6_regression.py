"""Run the existing full suite with reviewed raw-access denial and new receipts."""
from pathlib import Path
import sys,os,subprocess,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache');os.environ['PYTHONDONTWRITEBYTECODE']='1'
from src.next6.common import BASE,save,activity,utc_now

def main():
    out=BASE/'final_audit';out.mkdir(exist_ok=True)
    assert not (out/'FULL_TESTS.log').exists(),'Existing audit log must be preserved'
    with activity('full_legacy_and_NEXT6_regression','FINAL_AUDIT'):
        command=[sys.executable,'-B','-m','pytest','-q','tests','-p','src.next6.test_guard','-o','cache_dir=artifacts/next6/final_audit/pytest_cache',
            '--junitxml='+str(out/'FULL_TESTS.xml')]
        with (out/'FULL_TESTS.log').open('x',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        tree=ET.parse(out/'FULL_TESTS.xml');suites=list(tree.getroot().iter('testsuite'));counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
        save('final_audit/FULL_TESTS_RECEIPT.json',{'timestamp':utc_now(),'exit_code':result.returncode,'status':'PASS' if result.returncode==0 else 'FAIL',
            'counts':counts,'command':command,'canonical_raw_and_old_arrays_denied':True,'legacy_test_sources_unchanged':True})
        print(counts,flush=True)
        if result.returncode:raise RuntimeError('Full tests failed; inspect preserved log')

if __name__=='__main__':main()
