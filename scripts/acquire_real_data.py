"""Authenticated official acquisition. Never persist credentials or signed URLs."""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import utc_now, write_json, sha256

EXPECTED = (
    'X_train.parquet', 'y_train.parquet', 'y_train_index.parquet',
    'X_test.reduced.parquet', 'y_test.reduced.parquet', 'y_test_index.reduced.parquet',
)

def main():
    from crunch.api import Client
    import requests
    destination = ROOT / 'data/raw'
    destination.mkdir(parents=True, exist_ok=True)
    report = {'timestamp': utc_now(), 'canonical_directory': 'data/raw',
              'existing_setup_files': 'present but byte reads and ACL inspection denied by Windows',
              'original_files_modified': False, 'files': [], 'credentials_recorded': False}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            client, project = Client.from_project()
            release = project.competition.rounds.get('@current').phases.get_submission().get_data_release()
            available = release.data_files
        report['authentication'] = 'PROJECT_AUTHENTICATED'
        for name in EXPECTED:
            item = available[name]
            if item.name != name or item.compressed or not item.signed:
                raise RuntimeError('Unexpected official data contract')
            target = destination / name
            if not target.exists():
                temporary = destination / (name + '.part')
                with requests.get(item.url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            stream.write(chunk)
                if temporary.stat().st_size != item.size:
                    raise RuntimeError('Official byte-size mismatch')
                os.replace(temporary, target)
            if target.stat().st_size != item.size:
                raise RuntimeError('Existing canonical byte-size mismatch')
            record = {'name': name, 'bytes': target.stat().st_size, 'sha256': sha256(target)}
            report['files'].append(record)
            write_json(ROOT / 'artifacts/audits/REAL_DATA_ACQUISITION.json', report)
            print(json.dumps(record), flush=True)
        report['status'] = 'ACQUIRED'
    except Exception as exc:
        report['status'] = 'ACQUISITION_ERROR'
        report['exception_type'] = type(exc).__name__
        # HTTP exception strings and tracebacks may contain credential-bearing URLs.
    write_json(ROOT / 'artifacts/audits/REAL_DATA_ACQUISITION.json', report)
    print(report['status'], flush=True)

if __name__ == '__main__':
    main()
