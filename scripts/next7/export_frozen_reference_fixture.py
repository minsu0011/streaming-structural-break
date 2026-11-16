"""Bind complete historical model artifacts and a raw-free inference fixture."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next7/numba_cache')
from src.next7.worker_environment import environment
os.environ.update(environment())
import numpy as np
from src.next7.common import BASE,read_json,save,digest,utc_now,activity,note,require_guard


def main():
    require_guard()
    references=[
        {'id':'BL80_PRIMARY','role':'FROZEN_PRIMARY','entry':'next2_shared_performance_main.py',
         'model_directory':'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1',
         'model_files':['model.sbnext2','model.json']},
        {'id':'D2_REFERENCE','role':'PRESERVED_PREEXISTING_ROBUSTNESS_REFERENCE','entry':'next4_robustness_main.py',
         'model_directory':'artifacts/next4/refits/D2_DISCRETE_SCALE_MIXTURE/nominated_all_dev_v1',
         'model_files':['model.joblib','RESULTS.json']},
        {'id':'J102_REFERENCE','role':'PRESERVED_PREEXISTING_DISTILLED_REFERENCE','entry':'next2_canonical_distilled_main.py',
         'model_directory':'artifacts/next2/canonical_deployment/DISTILLED_tree_used_v1',
         'model_files':['model.sbnext2','model.json']}]
    protected=read_json(BASE/'PROTECTED_BEFORE_NEXT7_SHA256.json')
    files,coverage={},[]
    for reference in references:
        for name in [reference['entry'],*[reference['model_directory']+'/'+n for n in reference['model_files']]]:
            sha=digest(ROOT/name);files[name]=sha
            if name in protected:assert sha==protected[name]
            coverage.append({'file':name,'in_session_start_protected_manifest':name in protected,
                             'session_start_hash_matches':True if name in protected else None})
    assert files[references[0]['model_directory']+'/model.sbnext2']=='1cdfcde2c715db69ab2bf91b8e9803ac1dd24a5e146bea8f94cbc7e16e7b1d90'
    assert files[references[1]['model_directory']+'/model.joblib']=='ce04a393a127506291c1d7ee7bff75a11781dac435fc9dee89684a69c754568c'
    assert files[references[2]['model_directory']+'/model.sbnext2']=='27307b578f78bf4bcf11e8fb3011d34f82b14156eae7508b7514177cbc22b15d'
    d2=read_json(ROOT/references[1]['model_directory']/'RESULTS.json')
    assert d2['status']=='PASS' and d2['candidate']=='D2_DISCRETE_SCALE_MIXTURE'
    assert d2['files_sha256']['model.joblib']==files[references[1]['model_directory']+'/model.joblib']
    save('FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json',{'status':'FROZEN','created_utc':utc_now(),
        'source_sha256':digest(__file__),'references':references,'files_sha256':files,'authority_coverage':coverage,
        'source_dependency_note':'Public BL80 and J102 loaders require model.json. D2 requires RESULTS.json and the original BL80 model directory. All descriptors and entries are included explicitly.',
        'D2_descriptor_scope':'RESULTS.json was outside the initial protected inventory. Its current bytes are bound here and its model hash matches the independently protected original D2 model; no stronger original descriptor hash is claimed.',
        'historical_training_scope':'Preexisting fitted references may have historical full-DEV training before the NEXT5 RP/Shadow boundary. NEXT7 does not retrain them or decode hidden observations. Their replay is not RP OOF validation.',
        'selected_new_NEXT7_roles':[],'retained_operating_primary':'BL80_PRIMARY',
        'new_reference_fits':0,'new_candidates':0,'new_score_evaluations':0})
    def deny_data(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to((ROOT/'data').resolve()):raise PermissionError('Canonical fixture export cannot read competition data')
    sys.addaudithook(deny_data)
    with activity('Bind historical reference inference fixture and required descriptors','HANDOFF_REPRODUCTION_ENGINEERING','REPRODUCTION_AND_RUNTIME'):
        from scripts.next7.reproduce_frozen_references import calculate
        predictions,inputs=calculate()
        directory=BASE/'frozen_reference_fixture';directory.mkdir(parents=True,exist_ok=False)
        with (directory/'PREDICTIONS.npz').open('xb') as handle:np.savez_compressed(handle,**predictions)
        save('frozen_reference_fixture/FIXTURE.json',{'status':'FROZEN','created_utc':utc_now(),
            'reproducer_sha256':digest(ROOT/'scripts/next7/reproduce_frozen_references.py'),
            'exporter_sha256':digest(__file__),
            'reference_lock_sha256':digest(BASE/'FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json'),
            'predictions_sha256':digest(directory/'PREDICTIONS.npz'),'synthetic_input_sha256':inputs,
            'paths':9,'scalar_predictions':sum(len(v) for v in predictions.values()),
            'new_random_draws':0,'new_fits':0,'new_score_evaluations':0,'original_fixed_examples_reused':True,
            'scope':'Public entry scalar outputs of historical BL80/D2/J102 references, independent of the fivefold NEXT7 research-model fixture.'})
    note('Historical reference fixture exported','Complete BL80/D2/J102 model descriptors and nine raw-free public-entry paths','REPRODUCTION_AND_RUNTIME')


if __name__=='__main__':main()
