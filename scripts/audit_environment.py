import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, utc_now

def run(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}

def main():
    import psutil
    names = ['pip','numpy','pandas','pyarrow','scipy','scikit-learn','xgboost','lightgbm','joblib','numba','crunch-cli']
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"timestamp": utc_now(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "cpu_logical": os.cpu_count(), "cpu_physical": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total, "available_ram_bytes": psutil.virtual_memory().available,
        "packages": versions, "gpu": run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv']),
        "git": run(['git','--version']), "thread_policy": "training 4; inference 1; no nested process training"}
    write_json(ROOT / 'artifacts/audits/ENVIRONMENT.json', report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
