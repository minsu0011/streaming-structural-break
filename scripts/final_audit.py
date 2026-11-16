"""Evidence inventory and independent static audit; does not score data."""
import ast
import importlib.metadata
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,sha256,utc_now
from src.features.streaming import feature_version

def main():
    raw=ROOT/'artifacts/audits/RAW_DATA_MANIFEST.json'
    if raw.exists() and json.loads(raw.read_text(encoding='utf-8')).get('status')=='ACQUIRED':
        from scripts.final_audit_real import main as real_main
        if not (ROOT/'artifacts/validation/pytest_final_real.xml').exists():
            recent=max((ROOT/'artifacts/validation').glob('pytest_real_checkpoint*.xml'),key=lambda p:p.stat().st_mtime)
            sys.argv=[sys.argv[0],'--preflight','--tests',str(recent.relative_to(ROOT))]
        return real_main()
    checks={};limitations=[]
    tree=ast.parse((ROOT/'src/features/streaming.py').read_text(encoding='utf-8'))
    forbidden={'tau','target','dataset_id','online_length','has_break','y_test','y_train'}
    used={node.id for node in ast.walk(tree) if isinstance(node,ast.Name)}
    checks['feature_forbidden_identifiers_absent']=not bool(used&forbidden)
    source=json.loads((ROOT/'vendor/official/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    checks['official_source_hashes_match']=all(sha256(ROOT/item['path'])==item['sha256'] for item in source['sources'])
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    checks['source_data_module_tracked']='src/data/loader.py' in tracked
    checks['no_credential_paths_tracked']=not any('.crunchdao' in p or Path(p).name in ('token','.env') for p in tracked)
    checks['no_raw_or_binary_data_tracked']=not any(p.startswith('data/') or p.endswith(('.parquet','.joblib','.zip')) for p in tracked)
    prefix=json.loads((ROOT/'artifacts/validation/PREFIX_INVARIANCE.json').read_text(encoding='utf-8'))
    checks['prefix_pass_current_feature_hash']=prefix['status']=='PASS' and prefix['feature_hash']==feature_version()
    for name in ('SCORER_PARITY','DETERMINISM_REPORT','STREAMING_STRESS','FULL_OFFICIAL_RUNNER_SMOKE'):
        report=json.loads((ROOT/f'artifacts/validation/{name}.json').read_text(encoding='utf-8'))
        checks[name]=report['status']=='PASS'
    installed={}
    for line in (ROOT/'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
        if '==' in line:
            name,version=line.split('==',1)
            installed[name]=importlib.metadata.version(name)==version
    checks['locked_dependency_versions_match']=all(installed.values())
    tests_file=ROOT/'artifacts/validation/pytest.xml'
    if tests_file.exists():
        suites=ET.parse(tests_file).getroot()
        totals={k:sum(int(s.get(k,0)) for s in suites.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
        checks['pytest_all_pass']=totals['tests']>0 and totals['failures']==totals['errors']==0
    else:totals=None;checks['pytest_all_pass']=False
    limitations += ['Competition raw data unavailable: authentication blocker remains.',
        'No empirical official-label alignment, real-data CV, family ranking or champion.',
        'Seal IDs not yet allocated; policy frozen and no seal predictions generated.',
        'Official socket protocol PASS on synthetic data; authenticated crunch_tools.test and Linux cloud not run.',
        'Windows parallelism >1 unsupported by official runner; no cloud-worker scaling claims.',
        'Ablation selects learner columns; feature engine still computes all families.']
    result={'timestamp':utc_now(),'status':'ENGINEERING_AUDIT_PASS_DATA_BLOCKED' if all(checks.values()) else 'FAIL','checks':checks,
        'tests':totals,'limitations':limitations,'seal_opened':False,'reduced_scored_uses':0,'competition_cv_reported':False}
    write_json(ROOT/'artifacts/validation/FINAL_AUDIT.json',result);print(result)
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
