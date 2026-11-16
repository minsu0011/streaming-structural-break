"""Screen explicitly shipped source/configuration literals without exposing values."""
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,digest,utc_now,activity,note

PATTERNS={
    'AWS_ACCESS_KEY':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'OPENAI_STYLE_SECRET':re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{25,}'),
    'GITHUB_STYLE_SECRET':re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}'),
    'PRIVATE_KEY_HEADER':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'CREDENTIAL_STRING_ASSIGNMENT':re.compile(r'''(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*["']?\s*[:=]\s*["']([^"'\r\n]{12,})["']'''),
}


def main():
    plan=read_json(BASE/'HANDOFF_CONTENT_PLAN_NEXT7.json')
    paths={p for p in (ROOT/'src').rglob('*.py') if '__pycache__' not in p.parts}
    paths.update((ROOT/'scripts/next7').glob('*.py'))
    paths.update((ROOT/'scripts/next7').glob('*.mjs'))
    paths.update((ROOT/'src/next7').glob('*.html'))
    paths.update(ROOT/'configs'/name for name in plan['required_config_files'])
    found=[]
    with activity('Explicit handoff source/configuration credential-literal screen','HANDOFF_ENGINEERING','REPRODUCTION_AND_RUNTIME'):
        hashes={}
        for path in sorted(paths):
            relative=path.relative_to(ROOT).as_posix()
            text=path.read_text(encoding='utf-8-sig')
            hashes[relative]=digest(path)
            for category,pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    found.append({'file':relative,'line':text.count('\n',0,match.start())+1,
                                  'category':category,'literal_value_redacted':True})
        save('audit/HANDOFF_SOURCE_LITERAL_SCREEN_001.json',{'status':'PASS' if not found else 'REVIEW_REQUIRED',
            'created_utc':utc_now(),'source_sha256':digest(__file__),'files_checked':len(paths),
            'files_sha256':hashes,'matches':found,'raw_secret_values_written':False,
            'scope':'Lexical screen of explicitly planned source and seven feature configurations. This complements the explicit exclusion of credential/data/cache directories; it is not a proof that arbitrary secrets can never occur in text.',
            'final_archive_created':False})
    note('Handoff source literal screen complete',str(len(paths))+' source/configuration files; '+str(len(found))+' redacted review locations','REPRODUCTION_AND_RUNTIME')


if __name__=='__main__':main()
