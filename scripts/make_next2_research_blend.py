"""Assemble the eligible fixed blend from independently identified DEV models."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.io import model_class
from src.next2.blend import FixedBlendBundle
from src.utils.artifacts import write_json,sha256,utc_now
from src.next2.session import checkpoint


def main():
    eid='BL2_C205_DEPENDENCE_80'
    research=json.loads((ROOT/'artifacts/next2/experiments'/eid/'RESULTS.json').read_text())
    if research['status']!='FINALIST_DIAGNOSTICS_PENDING':
        raise RuntimeError('Fixed blend numeric admission required')
    refs=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    reference=next(r for r in refs['references'].values() if r['experiment_id']=='J102_SERIAL_VARIANCE_512')
    primary_path=ROOT/reference['model_path']
    if sha256(primary_path)!=reference['model_sha256']:
        raise RuntimeError('Frozen primary bytes changed')
    folder=ROOT/'artifacts/next2/refits/C205_DEPENDENCE/cache_v1'
    trained=json.loads((folder/'TRAINING.json').read_text())
    result=json.loads((folder/'RESULTS.json').read_text())
    if result['status']!='PASS' or sha256(folder/'model.joblib')!=result['model_sha256']:
        raise RuntimeError('Complete-DEV complement receipt invalid')
    out=ROOT/'artifacts/next2/refits'/eid/'components_v1'
    out.mkdir(parents=True,exist_ok=True)
    identity={'candidate':research['candidate'],'primary_model_sha256':sha256(primary_path),
        'complement_model_sha256':sha256(folder/'model.joblib'),'script_sha256':sha256(Path(__file__)),
        'primary_path':reference['model_path'],'complement_path':(folder/'model.joblib').relative_to(ROOT).as_posix()}
    if (out/'RESULTS.json').exists():
        old=json.loads((out/'RESULTS.json').read_text())
        if old['identity']!=identity or sha256(out/'model.joblib')!=old['model_sha256']:
            raise RuntimeError('Research blend identity changed')
        print('NEXT2 ALL-DEV BLEND RETAINED',flush=True)
        return
    if (out/'model.joblib').exists():
        raise RuntimeError('Partial blend must be reviewed')
    primary=model_class(reference['candidate']).load(primary_path)
    complement=model_class(trained['candidate']).load(folder/'model.joblib')
    blend=FixedBlendBundle(primary,complement,.8)
    blend.save(out/'model.joblib')
    FixedBlendBundle.load(out/'model.joblib')
    write_json(out/'RESULTS.json',{'status':'PASS','identity':identity,'model_sha256':sha256(out/'model.joblib'),
        'created_utc':utc_now(),'new_fits':0,'component_training_series':8000,'seal_rows':0,
        'weight':.8,'quality_source':'Honest component-paired five-fold OOF; no in-sample quality claim',
        'engineering_status':'Further full-DEV parity and deployment checks pending'})
    checkpoint('All-DEV fixed blend assembled',['Complete runtime parity','Counterexample extension'],
        artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    print('NEXT2 ALL-DEV BLEND CREATED',sha256(out/'model.joblib'),flush=True)


if __name__=='__main__':
    main()
