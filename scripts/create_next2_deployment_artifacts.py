"""Create reviewable research deployment artifacts, without final promotion."""
from pathlib import Path
import sys
import json
import shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.guard import SealGuard
from src.next2.inference import write_descriptor,load_artifact
from src.utils.artifacts import write_json,sha256


def main():
    guard=SealGuard(ROOT)
    refs=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    reference=next(r for r in refs['references'].values() if r['experiment_id']=='J102_SERIAL_VARIANCE_512')
    comp=json.loads((ROOT/'artifacts/next2/experiments/C205_DEPENDENCE/RESULTS.json').read_text())['candidate']
    base={'status':'FROZEN_NEXT2_DEPLOYMENT','development_ids':sorted(guard.dev_ids),
        'known_training_ids':sorted(guard.split.mapping),'split_sha256':guard.split.digest,
        'candidate_definition_frozen':True,'final_role_promotion_by_this_script':False}
    for role in ('DISTILLED','PERFORMANCE'):
        spec={**base,'role':role,'bundle_kind':'single' if role=='DISTILLED' else 'fixed_blend'}
        if role=='DISTILLED':
            spec['candidate']=reference['candidate']
            model=ROOT/reference['model_path']
            expected=reference['model_sha256']
        else:
            spec.update(primary=reference['candidate'],complement=comp,weight=.8)
            folder=ROOT/'artifacts/next2/refits/BL2_C205_DEPENDENCE_80/components_v1'
            model=folder/'model.joblib'
            expected=json.loads((folder/'RESULTS.json').read_text())['model_sha256']
        if sha256(model)!=expected:
            raise RuntimeError('Reviewable model identity changed')
        config=ROOT/'configs/next2'/('DEPLOYMENT_RESEARCH_'+role+'.json')
        if config.exists():
            if json.loads(config.read_text())!=spec:
                raise RuntimeError('Frozen research deployment recipe changed')
        else:
            write_json(config,spec)
        out=ROOT/'artifacts/next2/deployment_research'/role
        out.mkdir(parents=True,exist_ok=True)
        if (out/'model.json').exists():
            load_artifact(out)
        else:
            if (out/'model.joblib').exists():
                raise RuntimeError('Partial deployment artifact requires review')
            shutil.copyfile(model,out/'model.joblib')
            write_descriptor(out,spec,origin='IDENTIFIED_EXISTING_ALL_DEV_MODEL')
        print('NEXT2 REVIEWABLE DEPLOYMENT',role,expected,flush=True)


if __name__=='__main__':
    main()
