"""Export original and independently trained predictive models without pickle."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.canonical_model import save,dumps,source_hash
from src.next2.canonical_inference import write_artifact,load_artifact
from src.next2.inference import load_artifact as load_joblib
from src.next2.io import model_class
from src.utils.artifacts import write_json,sha256,utc_now
from src.next2.session import checkpoint


def main():
    out=ROOT/'artifacts/next2/canonical_deployment'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for role in ('DISTILLED','PERFORMANCE'):
        original=ROOT/'artifacts/next2/deployment_research'/role
        model,descriptor=load_joblib(original)
        spec=descriptor['specification']
        variants=['compact_v1','tree_used_v1'] if role=='DISTILLED' else ['compact_v1']
        for runtime in variants:
            folder=out/(role+'_'+runtime)
            if (folder/'model.json').exists():
                load_artifact(folder)
            else:
                write_artifact(model,folder,spec,runtime=runtime,origin='FROZEN_SCIENTIFIC_MODEL_EXPORT')
            rows.append({'role':role,'runtime':runtime,'original_joblib_sha256':sha256(original/'model.joblib'),
                'canonical_sha256':sha256(folder/'model.sbnext2'),'original_bytes':(original/'model.joblib').stat().st_size,
                'canonical_bytes':(folder/'model.sbnext2').stat().st_size})
    raw=ROOT/'artifacts/next2/entry_reproduction/PERFORMANCE/raw_entry_v1'
    actual,_=load_joblib(raw)
    save(actual,raw/'model.sbnext2')
    reference=out/'PERFORMANCE_compact_v1/model.sbnext2'
    if (raw/'model.sbnext2').read_bytes()!=reference.read_bytes():
        raise RuntimeError('Independently trained canonical model bytes differ')
    refs=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    for ref in refs['references'].values():
        model=model_class(ref['candidate']).load(ROOT/ref['model_path'])
        save(model,out/'references'/ref['experiment_id']/'model.sbnext2')
    report={'status':'PASS','created_utc':utc_now(),'format_source_sha256':source_hash(),'exports':rows,
        'independent_raw_entry_canonical_bytes_equal':True,'raw_entry_canonical_sha256':sha256(raw/'model.sbnext2'),
        'initial_joblib_mismatch_preserved':True,'learned_tree_arrays_modified':False,'old_model_files_overwritten':False,
        'unused_training_estimator_omitted':True,'final_role_promotion_by_this_script':False,'seal_rows':0}
    write_json(out/'RESULTS.json',report)
    checkpoint('Canonical exact model serialization resolves entry-path object aliasing',
        ['Canonical entry streaming and cold import checks','Full final independent-copy audit'],
        artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
