from pathlib import Path
import sys,os,shutil,subprocess,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6_extension.common import *

def main():
    stage=BASE/'runtime/demo_source_probe';assert not stage.exists(),'Keep previous source probe';stage.mkdir()
    selected=list(ROOT.glob('src/**/*.py'))+list(ROOT.glob('configs/*.json'))+[ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json',ROOT/'scripts/verify_next6_extension_portable.py',ROOT/'scripts/demo_next6_extension.py']+list((BASE/'observable_models').glob('fold_*/model.joblib'))
    for p in selected:
        dest=stage/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    python=BASE/'runtime/demo_venv/Scripts/python.exe';env=os.environ.copy();env['PYTHONPATH']=str(stage);env['PYTHONDONTWRITEBYTECODE']='1';env['NUMBA_CACHE_DIR']=str(stage/'artifacts/next6_extension/numba_cache')
    with activity('fresh_environment_copied_source_demo','REPRODUCTION_AUDIT'):
        r=subprocess.run([str(python),'-B',str(stage/'scripts/verify_next6_extension_portable.py')],cwd=stage,env=env,capture_output=True,text=True,encoding='utf-8');(BASE/'audit/FRESH_ENV_PORTABLE_PROBE.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
        if r.returncode:print(r.stderr[-4000:]);raise RuntimeError('Fresh environment portable probe failed')
        result=json.loads(r.stdout);result['created_utc']=utc_now();result['copied_files_sha256']={p.relative_to(ROOT).as_posix():digest(p) for p in selected};save('audit/FRESH_ENV_PORTABLE_PROBE.json',result)
        packages=result['new_environment_packages'];(BASE/'requirements-demo-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items()))+'\n',encoding='utf-8');print('PASS: fresh six-package environment,15 model/case replays,5 invalid-reference checks, no raw access',flush=True)
if __name__=='__main__':main()
