"""Restore frozen scientific evidence without reading prospective Shadow values."""
import csv
import importlib.metadata
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.guard import ResearchGuard,ShadowAccessError
from src.next.data import RAW_SHA


def main():
    out=BASE/'preflight'
    progress('NEXT5 frozen evidence and guard preflight','Restore all NEXT4 reports, protected model hashes and exact environment pins',phase='PREFLIGHT')
    guard=ResearchGuard(ROOT);blocked={}
    for purpose in ('training','validation','feature cache','OOF','candidate selection','bootstrap selection','label access'):
        for kind,ids in [('Shadow',guard.shadow_ids),('LOCAL_SEAL',guard.seal_ids)]:
            try:guard.admit([min(ids)],purpose=purpose)
            except ShadowAccessError:blocked[kind+':'+purpose]=True
            else:raise AssertionError('Forbidden membership admitted')
    raw={n:digest(ROOT/'data/raw'/n) for n in RAW_SHA};assert raw==RAW_SHA
    selection=read_json(ROOT/'DEV_CANDIDATES_NEXT4.json');audit=read_json(ROOT/'FINAL_AUDIT_NEXT4.json')
    assert audit['status']=='PASS' and all(v is True for v in audit['checks'].values())
    for name,sha in audit['evidence_sha256'].items():assert digest(ROOT/name)==sha,name
    for name,sha in selection['evidence_sha256'].items():assert digest(ROOT/name)==sha,name
    roles={}
    for old,new in [('FROZEN_PERFORMANCE','FROZEN_PRIMARY'),('NEXT4_ROBUSTNESS','NEXT4_ROBUSTNESS')]:
        role=selection['roles'][old]
        assert digest(ROOT/role['model_file'])==role['model_sha256']
        roles[new]=role
    bdir=ROOT/'artifacts/next4/refits/B2_LOG_ENERGY_SEMITAIL/diagnostic_all_dev_v1'
    bfit=read_json(bdir/'RESULTS.json');bscore=read_json(ROOT/'artifacts/next4/experiments/B2_LOG_ENERGY_SEMITAIL/full/RESULTS.json')
    assert digest(bdir/'model.joblib')==bfit['files_sha256']['model.joblib']
    roles['EXPLORATORY_NEAR_MISS']={'experiment_id':'B2_LOG_ENERGY_SEMITAIL','entry':'next4_b2_readonly_diagnostic_main.py',
        'model_file':(bdir/'model.joblib').relative_to(ROOT).as_posix(),'model_sha256':digest(bdir/'model.joblib'),
        'original_primary':{k:bscore[k] for k in ('mean','median','worst_fold','fold_scores','mean_delta')},
        'original_bootstrap':read_json(ROOT/'artifacts/next4/bootstrap/B2_LOG_ENERGY_SEMITAIL/RESULTS.json')['paired_all_time'],
        'promoted':False}
    d=read_json(ROOT/'DEV_CANDIDATES_NEXT2.json')['distilled'];dp=ROOT/d['model_directory']/'model.sbnext2'
    assert digest(dp)==d['canonical_model_sha256'];roles['FROZEN_DISTILLED']={**d,'model_file':dp.relative_to(ROOT).as_posix(),'model_sha256':digest(dp)}
    tables={}
    names=['MODEL_RESULTS_NEXT4.csv','CV_RESULTS_NEXT4.csv','PAIRED_BOOTSTRAP_NEXT4.csv','ALT_RESULTS_NEXT4.csv',
           'REGIME_RESULTS_NEXT4.csv','MECHANISM_RESULTS_NEXT4.csv','TIME_FRACTURE_NEXT4.csv','COUNTEREXAMPLE_RESULTS_NEXT4.csv',
           'LONG_NULL_RESULTS_NEXT4.csv','EXPERIMENT_LEDGER.csv']
    for name in names:
        with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        tables[name]={'rows':len(rows),'columns':list(rows[0]) if rows else [],'sha256':digest(ROOT/name)}
    reports={}
    for name in ('00_GPT_READ_FIRST_NEXT4.md','01_GPT_HANDOFF_NEXT4.md','LIMITATIONS_NEXT4.md','NEXT_SESSION_PLAN_NEXT4.md',
                 'LEAKAGE_AUDIT_NEXT4.json','PREFIX_INVARIANCE_NEXT4.json','DETERMINISM_NEXT4.json','REPRODUCTION_NEXT4.json','LATENCY_NEXT4.json'):
        text=(ROOT/name).read_text(encoding='utf-8');reports[name]={'characters':len(text),'sha256':digest(ROOT/name)}
    pins={}
    for line in (ROOT/'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):continue
        name,version=line.split('==');actual=importlib.metadata.version(name);assert actual==version;pins[name]=actual
    assert len(pins)==63
    snapshot=read_json(out/'PREEXISTING_SOURCE_AND_REPORT_SNAPSHOT.json')
    for name,r in snapshot['files'].items():assert digest(ROOT/name)==r['sha256'],name
    assert subprocess.check_output(['git','diff','--cached','--binary'],cwd=ROOT,stderr=subprocess.PIPE)==(out/'INITIAL_STAGED_PATCH.diff').read_bytes()
    assert read_json(ROOT/'artifacts/next4/SESSION.json')['status']=='COMPLETE'
    assert read_json(ROOT/'artifacts/next3/SESSION.json')['status']=='STOPPED_BY_USER'
    refs={'status':'FROZEN_NEXT5_REFERENCES','created_utc':utc_now(),'roles':roles,'raw_sha256':raw,
        'source_reports':reports,'prior_tables':tables,'exact_environment_pins':pins,'python':sys.version,
        'shadow_lock_sha256':guard.shadow_lock_sha,'research_split_sha256':guard.split.digest,
        'seal_lock_sha256':guard.lock_sha,'reduced_ledger_sha256':guard.ledger_sha,
        'preexisting_source_count':len(snapshot['files']),'staged_changes_preserved':True,
        'past_full_DEV_scores_are_NEXT5_selection_authority':False,'new_fits':0}
    atomic_json(out/'FROZEN_REFERENCES.json',refs,immutable=True)
    atomic_json(ROOT/'NEXT5_SHADOW_GUARD.json',{'status':'RUNTIME_ADMISSION_PASS_TESTS_PENDING','created_utc':utc_now(),
        'research_series':len(guard.research_ids),'shadow_series':len(guard.shadow_ids),'seal_series':len(guard.seal_ids),
        'blocked_admissions':blocked,'default_loader_scope':'NEXT5_RESEARCH_POOL_ONLY',
        'shadow_labels_loaded':False,'application_guard_not_OS_sandbox':True,
        'guard_source_sha256':digest(ROOT/'src/next5/guard.py'),'data_source_sha256':digest(ROOT/'src/next5/data.py')},immutable=True)
    atomic_json(out/'INITIAL_RESTORE.json',{'status':'HASH_GUARD_ENVIRONMENT_PASS_TESTS_PENDING','created_utc':utc_now(),
        'frozen_references_sha256':digest(out/'FROZEN_REFERENCES.json'),'new_fits':0,'shadow_rows':0,'seal_rows':0},immutable=True)
    print('NEXT5 FROZEN RESTORE PASS',len(snapshot['files']),'prior files;',len(pins),'pins; Shadow/Seal blocked',flush=True)


if __name__=='__main__':main()
