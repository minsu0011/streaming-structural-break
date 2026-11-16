"""Optional user-run secure setup; token is read without echo or command-line history.

Never invoked automatically when authentication is absent. Does not upload or submit.
"""
import contextlib
import getpass
import io
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import sha256

def main():
    from crunch.api import Client,SizeVariant
    from crunch.utils import ProjectInfo,write_project_info,write_token
    from crunch.command.download import download
    os.chdir(ROOT)
    if not (ROOT/'.crunchdao/project.json').exists():
        token=getpass.getpass('Crunch setup token (hidden; never paste into code): ')
        if not token:raise SystemExit('No token entered; no changes made.')
        try:
            project_token=Client.from_env(show_progress=False).project_tokens.upgrade(token)
            project=project_token.project
            if project.competition.name!='structural-break-real-time':raise ValueError('Wrong competition')
            (ROOT/'.crunchdao').mkdir(exist_ok=True)
            write_project_info(ProjectInfo(project.competition.name,project.name,project.user_id,SizeVariant.DEFAULT),str(ROOT))
            write_token(project_token.plain,str(ROOT))
        except Exception as exc:
            raise SystemExit(f'Setup failed ({type(exc).__name__}); no token or HTTP details are printed.') from None
        finally:
            token=None
    try:
        client,project=Client.from_project()
        if project.competition.name!='structural-break-real-time':raise ValueError('Wrong competition')
        # CLI progress can contain file URLs. Discard it, including on failure.
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            download(force=False)
    except BaseException as exc:
        raise SystemExit(f'Download failed ({type(exc).__name__}); credentials and HTTP details suppressed.') from None
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    copied=0
    for source in (ROOT/'data').glob('*.parquet'):
        destination=raw/source.name
        if destination.exists():
            if sha256(source)!=sha256(destination):raise RuntimeError('Refusing to overwrite immutable raw '+source.name)
        else:shutil.copy2(source,destination)
        copied+=1
    print(f'Official setup/download complete. {copied} parquet files available in data/raw. No submission performed.')
    print('Next: python scripts/audit_data.py; python scripts/verify_labels.py; python scripts/run_research.py')

if __name__=='__main__':main()
