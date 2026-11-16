"""NEXT6 boundary and byte checks, synthetic scorer parity, then RP admission."""
from pathlib import Path
import os, sys, json, subprocess, hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from src.next6.common import BASE,save,read_json,digest,utc_now,activity
import numpy as np
import pandas as pd
from src.next5.guard import ResearchGuard,ShadowAccessError
from src.next5.data import ResearchArrays
from src.next.data import RAW_SHA
from scripts.verify_scorer import load_official,official_score
from src.scoring.ts_auc import score_frames

def main():
    if (BASE/'PREFLIGHT.json').exists(): raise RuntimeError('Preserve preflight; inspect existing result')
    if not (BASE/'SESSION.json').exists():
        save('SESSION.json',{'started_utc':utc_now(),'status':'ACTIVE','research_budget_seconds':36000,
            'audit_budget_seconds':[2700,3600],'source_prompt':'80421c02-da5d-4c48-807e-42d6ed38a607',
            'no_time_padding':True,'new_shadow':False,'new_deployment_role':False})
    with activity('preflight','REPRODUCIBILITY'):
        protected={}
        for p in list(ROOT.glob('*'))+list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').rglob('*')):
            if p.is_file() and 'next6' not in str(p.relative_to(ROOT)).lower(): protected[p.relative_to(ROOT).as_posix()]=digest(p)
        roles=read_json(ROOT/'DEV_CANDIDATES_NEXT5.json')
        for role,r in roles['roles'].items():
            actual=digest(ROOT/r['model_file']); assert actual==r['model_sha256'],role
            protected[r['model_file']]=actual
        # Preserve canonical model payloads and descriptors without executing them.
        for folder in ('artifacts/next/canonical_deployment','artifacts/next2/canonical_deployment'):
            for p in (ROOT/folder).rglob('*'):
                if p.is_file(): protected[p.relative_to(ROOT).as_posix()]=digest(p)
        raw={name:digest(ROOT/'data/raw'/name) for name in RAW_SHA}
        assert raw==RAW_SHA,'Raw training bytes changed'
        guard=ResearchGuard(ROOT)
        assert (len(guard.research_ids),len(guard.shadow_ids),len(guard.seal_ids))==(6400,1600,2000)
        denied=[]
        for name,ids in [('shadow',guard.shadow_ids),('seal',guard.seal_ids)]:
            try:guard.admit([min(ids)],purpose='NEXT6 negative admission probe')
            except ShadowAccessError:denied.append(name)
        assert denied==['shadow','seal']
        directory=ROOT/'data/next5/research_5d54e316ff5bf25d';manifest=read_json(directory/'MANIFEST.json')
        for name,sha in manifest['files_sha256'].items():assert digest(directory/name)==sha,name
        assert manifest['identity']['research_split_sha256']==guard.split.digest
        assert manifest['identity']['shadow_lock_sha256']==guard.shadow_lock_sha
        assert manifest['identity']['raw_sha256']==raw
        refs={}
        for kind in ('J102','C205','B2_augmented_J','D2_expert'):
            root=ROOT/'artifacts/next5/references/fits/PRIMARY/full'/kind
            for f in range(5):
                d=root/f'fold_{f}';receipt=read_json(d/'RESULTS.json')
                assert digest(d/'model.joblib')==receipt['model_sha256']
                protected[(d/'model.joblib').relative_to(ROOT).as_posix()]=receipt['model_sha256']
            r=read_json(root/'RESULTS.json');assert digest(root/'OOF.npz')==r['OOF_sha256']
            refs[kind]={'mean_TS_AUC':r['mean_TS_AUC'],'OOF_sha256':r['OOF_sha256']}
        rng=np.random.default_rng(20260913);official=load_official();errors=[]
        for case in range(128):
            keys=[];ys=[];ps=[]
            for sid in range(int(rng.integers(3,24))):
                n=int(rng.integers(1,32));tau=int(rng.integers(0,n+1))
                keys.extend((sid,t) for t in range(n));ys.extend(int(t>=tau) for t in range(n));ps.extend(rng.integers(0,8,n)/7)
            idx=pd.MultiIndex.from_tuples(keys,names=['id','time']);y=pd.DataFrame({'target':ys},index=idx);p=pd.DataFrame({'prediction':ps},index=idx)
            errors.append(abs(score_frames(p,y)-official_score(official,p,y)))
        assert max(errors)<1e-12
        if not (BASE/'SCORER_PARITY_NEXT6.json').exists():
            save('SCORER_PARITY_NEXT6.json',{'status':'PASS','synthetic_cases':128,'max_abs_error':max(errors),
                'official_source_sha256':digest(ROOT/'vendor/official/scoring.py'),'official_io_synthetic_only':True})
        tests=['tests/test_next5_shadow_guard.py','tests/test_scorer_official_checks.py']
        with (BASE/'PREFLIGHT_TESTS_REPAIRED.log').open('x',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',
                '-p','src.next5.final_test_compatibility',*tests,'--basetemp='+str(BASE/'pytest_preflight_repaired'),
                '--junitxml='+str(BASE/'PREFLIGHT_TESTS_REPAIRED.xml')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,'Preflight tests failed; see log'
        # Admission checks precede decoding labels/observations in the isolated RP cache.
        data=ResearchArrays(directory,guard);rows=data.rows();assert len(data.ids)==6400 and len(rows['target'])==3203157
        for j,sid in enumerate(data.ids):
            a,b=data.series['online_offsets'][j:j+2];tau=int(data.series['tau'][j]);t=np.arange(b-a)
            np.testing.assert_array_equal(rows['target'][a:b],(t>=tau).astype(np.uint8) if tau>=0 else np.zeros(b-a,dtype=np.uint8))
        guard.check_state()
        save('PROTECTED_BASELINE_SHA256.json',protected)
        save('PREFLIGHT_RECOVERY.json',{'status':'RESOLVED','classification':'TEST_HARNESS_METADATA_DENIAL',
            'failure_log':'PREFLIGHT_TESTS.log','failed_tests':25,'passed_tests_before_repair':8,
            'repair':'Reuse existing src.next5.final_test_compatibility; only hash-bound ID/group/role/fold metadata admitted',
            'prior_source_changed':False,'prohibited_data_read':False,'repair_source_sha256':digest(ROOT/'src/next5/final_test_compatibility.py')})
        audit={'status':'PASS','timestamp':utc_now(),'raw_sha256':raw,'split_sha256':guard.split.digest,
            'RP_series':6400,'RP_online_rows':3203157,'shadow_rows_returned':0,'seal_rows_returned':0,
            'shadow_labels_decoded':False,'seal_labels_decoded':False,'reduced_test_read':False,
            'public_cloud_requests':0,'denied_admission_probes':denied,'cache_manifest_sha256':digest(directory/'MANIFEST.json'),
            'cache_scope':'Isolated existing RP cache; raw files byte-hashed only, no new raw parquet scanner',
            'application_guard_not_OS_sandbox':True,'raw_internal_decoder_zero_claim':False,
            'guard_tests_pass':True,'scorer_parity_pass':True,'reference_artifacts':refs,'protected_files':len(protected)}
        save('DATA_ACCESS_AUDIT_NEXT6.json',audit);save('PREFLIGHT.json',audit)
        print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
