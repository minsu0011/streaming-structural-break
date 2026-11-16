"""Freeze exact reference feature and execution hashes before any reference score."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5 import reference_features,model
from src.next5.scheduler import register_job


def main():
    assert read_json(BASE/'preflight/NEXT5_PREFLIGHT.json')['status']=='PASS'
    policy=read_json(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json')
    files=('scripts/train_next5_reference.py','src/next5/cache.py','src/next5/data.py','src/next5/guard.py',
        'src/next5/model.py','src/scoring/ts_auc.py')
    for kind in reference_features.KINDS:
        settings={'version':'next5_reference_features_v1','kind':kind};cid='REF_'+kind
        atomic_json(BASE/'references/registrations'/(cid+'.json'),{
            'experiment_id':cid,'created_utc':utc_now(),'created_before_score':True,'reference_not_new_candidate':True,
            'feature_module':'src.next5.reference_features','feature_settings':settings,
            'feature_sha256':reference_features.implementation_hash(settings),'model_source_sha256':model.source_hash(),
            'parameters':{'learner':policy['learner'],'learner_params':policy['learner_params'],'weighting':policy['weighting']},
            'execution_source_sha256':{n:digest(ROOT/n) for n in files},
            'reference_policy_sha256':digest(BASE/'references/POLICY.json'),'shadow_lock_sha256':digest(ROOT/'NEXT5_SHADOW_LOCK.json'),
            'training_scope':'Only locked ResearchPool train groups, no Shadow, no LOCAL SEAL',
            'use':'Unchanged historical recipe retrained for matched ResearchPool baseline comparisons'},immutable=True)
    register_job({'id':'REF000_FEATURE_CONTRACTS','kind':'validation','priority':20,'depends_on_success':['PF016_FINALIZE_RECOVERY'],
        'argv':['-m','pytest','-q','tests/test_next5_reference_features.py','--junitxml=artifacts/next5/references/FEATURE_CONTRACTS.xml'],
        'description':'Four exact reference adapters, 100 suffix tests each and original feature parity','retry_safe':False})
    for kind in reference_features.KINDS:
        for scope in (('full','screen') if kind in ('J102','C205') else ('full',)):
            register_job({'id':'REF_'+kind+'_'+scope,'kind':'validation','priority':25,
                'depends_on_success':['REF000_FEATURE_CONTRACTS'],
                'argv':['scripts/train_next5_reference.py','--kind',kind,'--scope',scope],
                'description':'Fixed matched ResearchPool '+kind+' '+scope+' reference; no Shadow rows','retry_safe':False})
    progress('Reference recipe and source preregistration complete','24 prospective reference learner fits, held-out fold groups, training-only weights and columns')


if __name__=='__main__':main()
