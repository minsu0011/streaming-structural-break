"""Record installed versions without installing packages or exposing pip URLs."""
from pathlib import Path
import sys
import platform
import importlib.metadata as metadata
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,utc_now

names=['numpy','scipy','scikit-learn','joblib','threadpoolctl','numba','llvmlite','lightgbm','xgboost','pandas','pyarrow','psutil','pytest']
versions={name:metadata.version(name) for name in names}
atomic(BASE/'REPRODUCTION_ENVIRONMENT_NEXT9.json',dict(status='RECORDED',timestamp=utc_now(),python=platform.python_version(),
    implementation=platform.python_implementation(),platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
    versions=versions,numeric_threads=1,package_installation_performed=False,
    package_remote_availability_not_checked=True,actual_reproduction_uses_existing_verified_venv=True,
    fresh_compiler_cache_is_payload_relative_nc=True))
atomic(BASE/'REPRODUCTION_REQUIREMENTS_NEXT9.txt','\n'.join(f'{name}=={version}' for name,version in versions.items())+'\n')
