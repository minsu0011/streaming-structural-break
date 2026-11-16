"""Check official data availability without logging any credential or signed URL."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json, utc_now, ledger

def main():
    from crunch.api import Client
    import requests
    report = {"timestamp": utc_now(), "credential_env_present": bool(os.environ.get("CRUNCH_API_KEY")),
        "local_project_present": (ROOT/'.crunchdao/project.json').exists(), "local_token_present": (ROOT/'.crunchdao/token').exists()}
    try:
        client = Client.from_project()[0] if report['local_project_present'] else Client.from_env(show_progress=False)
        competition = client.competitions.get('structural-break-real-time')
        release = competition.rounds.get('@current').phases.get_submission().get_data_release()
        report['files'] = [{"name": item.name,"bytes": item.size,"signed": item.signed} for item in release.data_files.values()]
        item = release.data_files['y_train_index.parquet']
        response = requests.head(item.url, timeout=30)
        report['sample_head_status'] = response.status_code
        report['status'] = 'ACCESS_AVAILABLE' if item.signed and response.ok else 'BLOCKED_AUTH'
    except Exception as exc:
        # Do not serialize exception text: HTTP exceptions can contain signed URLs.
        report['status'] = 'ACCESS_CHECK_ERROR'
        report['exception_type'] = type(exc).__name__
    write_json(ROOT/'artifacts/audits/DATA_ACCESS.json',report)
    ledger(experiment_id='official_data_access',model='none',status=report['status'],notes='Safe access metadata in artifacts/audits/DATA_ACCESS.json; no credential or URL persisted')
    print(report['status'])

if __name__ == '__main__':
    main()
