"""Prove complete historical references run from a fresh source/model payload."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,digest,utc_now,activity,note
from src.next7.worker_environment import environment
from src.next7.handoff_archive import checked_source


def main():
    relative='reproduction_frozen_references/research_001'
    directory=BASE/relative
    directory.mkdir(parents=True,exist_ok=False)
    payload=directory/'payload'
    lock=read_json(BASE/'FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json')
    files=set(lock['files_sha256'])
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py') if '__pycache__' not in p.parts)
    files.update('configs/'+name for name in read_json(BASE/'HANDOFF_CONTENT_PLAN_NEXT7.json')['required_config_files'])
    files.update(('COUNTEREXAMPLE_SUITE_LOCK.json','scripts/next7/reproduce_frozen_references.py',
                  'artifacts/next7/FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json',
                  'artifacts/next7/frozen_reference_fixture/FIXTURE.json',
                  'artifacts/next7/frozen_reference_fixture/PREDICTIONS.npz'))
    hashes={name:digest(checked_source(ROOT,name)) for name in files}
    save(relative+'/START.json',{'status':'STARTED','created_utc':utc_now(),'source_sha256':digest(__file__),
        'files_sha256':hashes,'new_fits':0,'new_scores':0,'required_descriptors_included':True})
    with activity('Isolated public-entry replay of historical BL80 D2 J102 references','HANDOFF_REPRODUCTION_ENGINEERING','REPRODUCTION_AND_RUNTIME'):
        for name in sorted(files):
            target=payload/name;target.parent.mkdir(parents=True,exist_ok=True)
            with (ROOT/name).open('rb') as source,target.open('xb') as destination:shutil.copyfileobj(source,destination)
            assert digest(target)==hashes[name]
        env=environment();env.pop('PYTHONPATH',None)
        env['PYTHONDONTWRITEBYTECODE']='1';env['NUMBA_CACHE_DIR']=str(payload/'artifacts/next7/numba_cache')
        command=[sys.executable,'-B','scripts/next7/reproduce_frozen_references.py',
                 '--output-directory',str(payload/'demo_reference_output'),'--deny-original-root',str(ROOT)]
        with (directory/'REPRODUCTION.log').open('x',encoding='utf-8') as output:
            process=subprocess.run(command,cwd=payload,env=env,stdout=output,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        assert process.returncode==0,'Preserve failed payload and inspect its reproduction log'
        result=read_json(payload/'demo_reference_output/FROZEN_REFERENCE_REPRODUCTION_DONE.json')
        assert result['status']=='PASS' and result['public_entry_paths']==9 and result['scalar_predictions']==9216
        assert result['original_workspace_access_blocked'] and result['denied_access_attempts']==0
        assert all(item['bitwise_equal'] and item['max_absolute_error']==0 for item in result['checks'])
        save('audit/FROZEN_REFERENCE_ISOLATED_REPLAY_001.json',{'status':'PASS','created_utc':utc_now(),
            'source_sha256':digest(__file__),'start_sha256':digest(directory/'START.json'),
            'reference_lock_sha256':digest(BASE/'FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json'),
            'fixture_sha256':digest(BASE/'frozen_reference_fixture/FIXTURE.json'),
            'payload_replay_receipt_sha256':digest(payload/'demo_reference_output/FROZEN_REFERENCE_REPRODUCTION_DONE.json'),
            'public_entry_paths':9,'scalar_predictions':9216,'all_outputs_bitwise_equal':True,
            'original_workspace_access_blocked':True,'denied_access_attempts':0,
            'original_model_hashes_unchanged':True,'new_models_or_scores':False,
            'fresh_output_process_using_existing_library_environment':True,
            'packaging_dependency_repair':'Include original BL80/J102 model.json and D2 RESULTS.json with models and entry points. The issue was found before any final ZIP was built.'})
    note('Historical reference isolated replay passed','Nine public-entry paths and9216 scalar values; complete descriptors and no original project reads','REPRODUCTION_AND_RUNTIME')


if __name__=='__main__':main()
