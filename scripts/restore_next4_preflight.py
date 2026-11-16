"""Fast reference restoration and byte/guard/environment checks, without fitting."""
from pathlib import Path
import sys
import importlib.metadata
import platform
import csv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,read_json,utc_now,progress
from src.next.guard import SealGuard,SealAccessError
from src.next.data import RAW_SHA


def main():
    out=BASE/'preflight'
    if (out/'INITIAL_RESTORE.json').exists(): raise RuntimeError('Keep earlier restoration evidence')
    progress('NEXT4 reference preflight','Raw byte hashes, guards, frozen scientific model receipts and exact environment pins',phase='PREFLIGHT')
    guard=SealGuard(ROOT);guard.check_state();guard.admit(purpose='NEXT4 DEV-only research')
    blocked={}
    for purpose in ('training','validation','OOF','feature generation','label access'):
        try:guard.admit([min(guard.seal_ids)],purpose=purpose)
        except SealAccessError:blocked[purpose]=True
        else:raise RuntimeError('Seal identity admission unexpectedly succeeded')
    raw={name:digest(ROOT/'data/raw'/name) for name in RAW_SHA}
    assert raw==RAW_SHA and len(guard.dev_ids)==8000 and len(guard.seal_ids)==2000
    assert not (guard.dev_ids&guard.seal_ids)
    selection=read_json(ROOT/'DEV_CANDIDATES_NEXT2.json')
    prior=read_json(ROOT/'artifacts/next3/preflight/FROZEN_REFERENCES.json')
    roles={}
    for role,key in [('FROZEN_DISTILLED','distilled'),('FROZEN_PERFORMANCE_BL80','performance')]:
        original=selection[key];model=ROOT/original['model_directory']/'model.sbnext2'
        assert digest(model)==original['canonical_model_sha256']
        expected_key='FROZEN_PERFORMANCE' if key=='performance' else 'FROZEN_DISTILLED'
        assert original==prior['roles'][expected_key]
        roles[role]={**original,'model_file':model.relative_to(ROOT).as_posix(),'actual_model_sha256':digest(model)}
    scientific=prior['roles']['FROZEN_J102_SCIENTIFIC']
    assert digest(ROOT/scientific['model'])==scientific['sha256']
    roles['FROZEN_J102']={**scientific,'role':'Frozen original scientific model, not a new fit'}
    assert raw==prior['raw_sha256'] and guard.split.digest==prior['split_sha256']
    assert guard.lock_sha==prior['seal_lock_sha256'] and guard.ledger_sha==prior['reduced_ledger_sha256']
    versions={}
    for line in (ROOT/'requirements-lock.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        name,expected=line.split('==');actual=importlib.metadata.version(name)
        assert actual==expected,(name,actual,expected);versions[name]=actual
    pointer=read_json(ROOT/'artifacts/next3/runtime/USER_STOP_POINTER.json')
    stop_path=Path(pointer['receipt']);assert digest(stop_path)==pointer['sha256']
    delivery=read_json(stop_path.parent/'DELIVERY_VERIFICATION.json');folder=Path(delivery['folder'])
    for name,record in delivery['files'].items():assert digest(folder/name)==record['sha256']
    assert len(list(folder.iterdir()))==20
    csv_summaries={}
    for path in sorted(folder.glob('*.csv')):
        with path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        csv_summaries[path.name]={'rows':len(rows),'columns':list(rows[0]) if rows else [],'sha256':digest(path)}
    assert read_json(ROOT/'artifacts/next3/SESSION.json')['status']=='STOPPED_BY_USER'
    source_names=['DEV_CANDIDATES_NEXT2.json','CURRENT_GENERATION_COMPONENT_GRAPH.md',
        'artifacts/next3/preflight/FROZEN_REFERENCES.json','artifacts/next3/SESSION.json',
        'artifacts/validation/GROUP_ASSIGNMENTS.parquet','artifacts/validation/SPLIT_LOCK_REAL.json',
        'ROBUSTNESS_SPLIT_LOCK.json','SEAL_PARTITION_500x4_LOCK.json','REDUCED_TEST_USAGE_LEDGER.json',
        'requirements-lock.txt','docs/RESEARCH_PROTOCOL_NEXT4.md']
    refs={'status':'FROZEN_NEXT4_SCIENTIFIC_REFERENCES','created_utc':utc_now(),'roles':roles,
        'raw_sha256':raw,'split_sha256':guard.split.digest,'seal_lock_sha256':guard.lock_sha,
        'reduced_ledger_sha256':guard.ledger_sha,'source_sha256':{n:digest(ROOT/n) for n in source_names},
        'NEXT3_stop_receipt_sha256':digest(stop_path),'NEXT3_20_file_delivery':delivery,
        'EXPLORATORY_REFERENCE_ONLY':['I2_HISTORICAL_RANK_PRODUCTS'],
        'NEXT3_E003_final_canonical':False,'new_fits':0}
    atomic_json(out/'FROZEN_REFERENCES.json',refs,immutable=True)
    atomic_json(out/'INITIAL_RESTORE.json',{'status':'HASH_GUARD_ENVIRONMENT_PASS_FRESH_REPLAY_AND_TESTS_PENDING',
        'created_utc':utc_now(),'blocked_seal_admissions':blocked,'DEV_series':8000,'seal_series':2000,
        'DEV_seal_overlap':0,'raw_sha256':raw,'frozen_references_sha256':digest(out/'FROZEN_REFERENCES.json'),
        'pins':versions,'python':sys.version,'platform':platform.platform(),'prior_result_tables':csv_summaries,
        'all_seals_unopened':True,'new_reduced_uses':0,'new_public_submissions':0,'new_cloud_submissions':0,
        'new_fits':0,'source_sha256':digest(Path(__file__)),
        'pending':['Fresh scorer/guard smoke tests','Fresh FROZEN_PERFORMANCE public prediction fingerprint'],
        'scope':'Full-file SHA checks do not analyze seal observations; all value access remains DEV-predicate filtered.'},immutable=True)
    progress('NEXT4 reference restoration passed','Canonical references, 20 stopped result files, raw/split/seal bytes and exact package pins verified',phase='PREFLIGHT')
    print('NEXT4 RESTORATION PASS',len(versions),'pins; 20 prior handoff files; zero fits',flush=True)


if __name__=='__main__':main()
