"""Create a separate minimal environment for the portable synthetic demo."""
from pathlib import Path
import sys,os,subprocess,importlib.metadata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6_extension.common import *

def main():
    dest=BASE/'runtime/demo_venv';assert not dest.exists(),'Keep an existing environment; do not overwrite it'
    packages=['numpy','numba','llvmlite','joblib','psutil'];pins={p:importlib.metadata.version(p) for p in packages};requirements=BASE/'requirements-demo-pinned.txt';requirements.write_text('\n'.join(f'{p}=={v}' for p,v in pins.items())+'\n',encoding='utf-8')
    with activity('fresh_minimal_demo_environment','REPRODUCTION_PREPARATION'):
        subprocess.run([sys.executable,'-m','venv',str(dest)],check=True,cwd=ROOT);python=dest/'Scripts/python.exe';command=[str(python),'-m','pip','install','--disable-pip-version-check','--only-binary=:all:','--index-url','https://pypi.org/simple','--cache-dir',str(BASE/'runtime/pip_cache'),'-r',str(requirements)]
        r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8');(BASE/'audit/DEMO_ENV_INSTALL.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
        save('audit/DEMO_ENVIRONMENT_PREPARATION.json',{'status':'PASS' if r.returncode==0 else 'DEPENDENCY_INSTALL_FAILED','created_utc':utc_now(),'environment':str(dest.relative_to(ROOT)),'packages':pins,'requirements_sha256':digest(requirements),'public_package_index':'https://pypi.org/simple','isolated_site_packages':True,'source_copy_demo_not_yet_run':True})
        print(r.stdout[-3000:],flush=True)
        if r.returncode:print(r.stderr[-3000:],flush=True);raise RuntimeError('Dependency install failed; log preserved')
if __name__=='__main__':main()
