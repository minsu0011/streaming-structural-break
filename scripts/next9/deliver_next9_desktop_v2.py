"""Create a new exact20-file desktop handoff and verified matching ZIP."""
from pathlib import Path
import sys
import argparse
import shutil
import zipfile
import datetime as dt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

def main(archive_tag):
    from src.next9.work_time import snapshot
    from src.next9.queue import identity_alive
    clock=snapshot();assert clock['productive_seconds']>=36000 and clock['audit_seconds']>=3600
    audit=read_json(BASE/'FINAL_AUDIT_NEXT9.json');assert audit['status']=='PASS'
    reproduction=read_json(BASE/'delivery'/archive_tag/'REPRODUCTION_DONE.json')
    assert reproduction['status']=='PASS' and reproduction['final_payload']
    archive=ROOT/reproduction['archive_path'];assert digest(archive)==reproduction['archive_sha256']
    owner=read_json(BASE/'runtime/SCHEDULER_OWNER.json')
    assert not identity_alive(owner['pid'],owner['process_created']),'Stop owned scheduler before final delivery'
    for path in (BASE/'runtime/queue').glob('*/attempt_*/START.json'):
        record=read_json(path);assert not identity_alive(record['pid'],record['process_created']),str(path)
    # Resolve the user's actual Windows Desktop known folder (including a
    # redirected Desktop) without enumerating or altering existing outputs.
    import subprocess
    result=subprocess.run(['powershell','-NoProfile','-Command','[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes([Environment]::GetFolderPath("Desktop")))'],capture_output=True,text=True,check=True)
    import base64
    desktop=Path(base64.b64decode(result.stdout.strip()).decode('utf-8')).resolve();assert desktop.is_dir()
    stamp=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    folder=desktop/f'StructuralBreak_NEXT9_핵심결과20_{stamp}'
    matching=desktop/(folder.name+'.zip')
    full=desktop/f'STRUCTURAL_BREAK_NEXT9_MOVEMENT_STATE_{stamp}.zip'
    assert not folder.exists() and not matching.exists() and not full.exists()
    stage=BASE/'delivery'/f'd20_{stamp}';stage.mkdir(parents=True,exist_ok=False)
    direct={
        '00_READ_FIRST_NEXT9_KO.md':'00_READ_FIRST_NEXT9_KO.md',
        '01_GPT_HANDOFF_NEXT9.md':'01_GPT_HANDOFF_NEXT9.md',
        '03_MOVEMENT_ELIGIBILITY_NEXT9.csv':'MOVEMENT_ELIGIBILITY.csv',
        '05_MOVEMENT_STATE_CONTRACT_NEXT9.md':'MOVEMENT_STATE_CONTRACT.md',
        '06_COMMON_RANGE_RESULTS_NEXT9.csv':'COMMON_RANGE_RESULTS.csv',
        '07_CONDITIONAL_ORIENTATION_RESULTS_NEXT9.csv':'CONDITIONAL_ORIENTATION_RESULTS.csv',
        '08_VARIABLE_CERTIFICATION_NEXT9.json':'VARIABLE_CERTIFICATION_NEXT9.json',
        '11_MODEL_COMPARISON_NEXT9.csv':'MODEL_COMPARISON_NEXT9.csv',
        '12_PAIRED_UNCERTAINTY_NEXT9.csv':'PAIRED_UNCERTAINTY_NEXT9.csv',
        '14_COUNTEREXAMPLES_NEXT9.csv':'COUNTEREXAMPLES_NEXT9.csv',
        '15_FEATURE_CARDS_NEXT9_KO.md':'FEATURE_CARDS_NEXT9_KO.md',
        '16_NEXT9_TEN_FIGURES.pdf':'FIGURES/NEXT9_TEN_FIGURES.pdf',
        '18_EXPERIMENT_LEDGER_NEXT9.csv':'EXPERIMENT_LEDGER_NEXT9.csv',
        '19_NEXT_SESSION_PLAN_NEXT9.md':'NEXT_SESSION_PLAN_NEXT9.md'}
    bundles={
        '02_PROTOCOL_BUDGET_AND_PROCESS_NEXT9.json':['NEXT9_PROTOCOL_LOCK.json','NEXT9_CANDIDATE_BUDGET.json','FINAL_TIME_ACCOUNTING_NEXT9.json','EXECUTION_AUDIT_NEXT9.json','CURRENT_CONTROLLERS_NEXT9.json'],
        '04_MOVEMENT_AND_CENTER_NEXT9.json':['TARGET_FREE_MOVEMENT_SCREEN.csv','H_CENTER_COMPARISON.csv','EARLY_LATE_MOVEMENT_RESULTS.csv','CENTER_DECOMPOSITION_SUMMARY_NEXT9.csv'],
        '09_SELF_TRANSITIONS_NEXT9.json':['SELF_TRANSITION_RESULTS.csv','SELF_TRANSITION_MODELS.json','STATE_OBSERVABILITY_NEXT9.json','SELF_CADENCE_DONE_NEXT9.json','SELF_CADENCE_FIVE_FOLD_NEXT9.csv','SELF_CADENCE_FINDING_NEXT9_KO.md'],
        '10_PREDICTIVE_FACTOR_GRAPH_NEXT9.json':['PREDICTIVE_FACTOR_GRAPH_NEXT9.json','CROSS_FACTOR_RESULTS.csv'],
        '13_TIME_AND_H_GENERALIZATION_NEXT9.json':['TIME_GENERALIZATION_NEXT9.csv','H_GENERALIZATION_NEXT9.csv','CV_RESULTS_NEXT9.csv','STATE_TRANSPORT_SUMMARY_NEXT9.csv','STATE_TRANSPORT_FINDING_NEXT9_KO.md','SAMPLING_PHASE_SUMMARY_NEXT9.csv','SAMPLING_PHASE_FINDING_NEXT9_KO.md'],
        '17_FINAL_LEAKAGE_RUNTIME_REPRODUCTION_NEXT9.json':['FINAL_AUDIT_NEXT9.json','LEAKAGE_AUDIT_NEXT9.json','LATENCY_NEXT9.json','DETERMINISM_NEXT9.json','REPRODUCTION_NEXT9.json','INDEPENDENT_TS_AUC_AUDIT_NEXT9.json','LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json','FITTED_MODEL_PARAMETER_AUDIT_NEXT9.json','FAILED_ATTEMPT_RECONCILIATION_NEXT9.json','EAGER_SCHEMA_CACHE_QUALIFICATION_NEXT9.json']}
    assert len(direct)+len(bundles)==20
    import csv
    with activity('Publish verified exact20 desktop handoff, matching ZIP and full raw-free archive','FINAL_AUDIT','HANDOFF'):
        for name,source in direct.items():shutil.copyfile(BASE/source,stage/name)
        for name,sources in bundles.items():
            content={}
            for source in sources:
                path=BASE/source
                if path.suffix=='.json':value=read_json(path)
                elif path.suffix=='.csv':
                    with path.open(encoding='utf-8-sig',newline='') as f:value=list(csv.DictReader(f))
                else:
                    assert path.suffix in ('.md','.txt')
                    value=path.read_text(encoding='utf-8-sig')
                content[source]=dict(sha256=digest(path),content=value)
            atomic(stage/name,dict(status='COMPLETE',files=content,lossless_consolidation_for_exact20_handoff=True))
        hashes={p.name:digest(p) for p in stage.iterdir() if p.is_file()};assert len(hashes)==20
        folder.mkdir(exist_ok=False)
        for name,sha in hashes.items():shutil.copyfile(stage/name,folder/name);assert digest(folder/name)==sha
        assert {p.name for p in folder.iterdir()}==set(hashes)
        with zipfile.ZipFile(matching,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in sorted(hashes):z.write(folder/name,name)
        import hashlib
        with zipfile.ZipFile(matching) as z:
            assert set(z.namelist())==set(hashes) and len(z.namelist())==20 and z.testzip() is None
            for name,sha in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
        shutil.copyfile(archive,full);assert digest(full)==reproduction['archive_sha256']
        atomic(BASE/'DESKTOP_DELIVERY_NEXT9.json',dict(status='PASS',timestamp=utc_now(),folder=str(folder),exact_files=20,
            files_sha256=hashes,matching_zip=str(matching),matching_zip_sha256=digest(matching),matching_zip_bytes=matching.stat().st_size,
            full_zip=str(full),full_zip_sha256=digest(full),full_zip_bytes=full.stat().st_size,
            full_reproduction_receipt_sha256=digest(BASE/'delivery'/archive_tag/'REPRODUCTION_DONE.json'),
            originals_preserved=True,no_owned_scheduler_or_workers_alive=True,receipt_external_to_archives=True))
        print(str(folder),str(matching),str(full),sep='\n',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive-tag',required=True);main(p.parse_args().archive_tag)
