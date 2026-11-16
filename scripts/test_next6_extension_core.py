from pathlib import Path
import sys,os,subprocess,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache');os.environ['PYTHONDONTWRITEBYTECODE']='1'
from src.next6_extension.common import *

def main():
    tests=sorted((ROOT/'tests').glob('test_next6_extension_*.py'));dest=BASE/'audit';dest.mkdir(exist_ok=True);xml=dest/'CORE_ENGINE_TESTS.xml'
    assert not (dest/'CORE_ENGINE_TEST_RECEIPT.json').exists(),'Preserve completed test receipt'
    with activity('all_extension_core_engine_tests','RUNTIME_DEBUG_AUDIT'):
        command=[sys.executable,'-B','-m','pytest',*[str(p) for p in tests],'-q','-p','no:cacheprovider','--junitxml',str(xml)];r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8');(dest/'CORE_ENGINE_TESTS.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');print(r.stdout,flush=True);assert r.returncode==0,r.stderr
        root=ET.parse(xml).getroot();counts={key:sum(int(s.attrib.get(key,0)) for s in root.iter('testsuite')) for key in ('tests','errors','failures','skipped')}
        source={p.relative_to(ROOT).as_posix():digest(p) for p in sorted((ROOT/'src/next6_extension').glob('*.py'))};save('audit/CORE_ENGINE_TEST_RECEIPT.json',{'status':'PASS','created_utc':utc_now(),'counts':counts,'command':command,'source_sha256':source,'tests_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in tests},'scope':'Extension core engines and adapters; initial NEXT6 1371-test receipt is separate and unchanged.'})
if __name__=='__main__':main()
