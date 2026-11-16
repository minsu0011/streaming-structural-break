"""One immutable registration schema for remaining already-declared family slots."""
from pathlib import Path
import importlib
import xml.etree.ElementTree as ET
from src.next5.artifacts import ROOT,BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.model import source_hash
from src.next5.scheduler import register_job


def register_family(*,family,module_name,version,entries,test_job,test_xml,caller,priority=40):
    state=read_json(BASE/'jobs'/test_job/'STATE.json');assert state['status']=='COMPLETE' and state['exit_code']==0
    tree=ET.parse(ROOT/test_xml)
    if not tree.findall('.//testcase') or tree.findall('.//failure') or tree.findall('.//error') or tree.findall('.//skipped'):
        raise RuntimeError('Completed non-skipped family feature contracts required')
    policy=read_json(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json');contract=read_json(ROOT/'HYPOTHESIS_FAMILY_LOCK_NEXT5.json')['families'][family]
    module=importlib.import_module(module_name)
    files=('scripts/evaluate_next5_candidate.py','src/next5/cache.py','src/next5/data.py','src/next5/guard.py',
        'src/next5/model.py','src/next5/references.py','src/scoring/ts_auc.py')
    for entry in entries:
        cid=entry['experiment_id'];assert cid in contract['slots'];settings={'version':version,'kind':entry['kind']}
        combination=entry.get('combination',{'ResearchPool_BL80':.9,'new_model':.1})
        record={'experiment_id':cid,'family':family,'parent':entry['parent'],'created_utc':utc_now(),'created_before_score':True,
            'scientific_question':entry['scientific_question'],'mechanism':entry['mechanism'],
            'exact_feature_change':entry['exact_feature_change'],'exact_model_change':'One fixed W3 LightGBM expert; fixed combination '+str(combination),
            'parameter_menu':{**contract['menu'],**entry.get('implementation_constants',{}),'actual_kind':entry['kind'],'additional_search':False},
            'screen_gate':policy['screen'],'full_gate':'Only a passed frozen screen; five locked PRIMARY ResearchPool folds',
            'promotion_gate':policy['performance_gate'],'stop_rule':'Failed screen closes slot; no micro-search; global saturation and hard budgets apply.',
            'expected_risk':entry['expected_risk'],'feature_module':module_name,'feature_settings':settings,
            'feature_names':module.feature_names(settings),'feature_sha256':module.implementation_hash(settings),'model_source_sha256':source_hash(),
            'execution_source_sha256':{n:digest(ROOT/n) for n in files},
            'parameters':{'learner':policy['learner'],'learner_params':policy['learner_params'],'weighting':policy['weighting']},
            'combination':combination,'policy_sha256':digest(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json'),
            'family_lock_sha256':digest(ROOT/'HYPOTHESIS_FAMILY_LOCK_NEXT5.json'),'shadow_lock_sha256':digest(ROOT/'NEXT5_SHADOW_LOCK.json'),
            'feature_contract_test_sha256':digest(ROOT/test_xml),'registration_source_sha256':{caller:digest(ROOT/caller),'src/next5/registration.py':digest(Path(__file__))},
            'shadow_data_used':False}
        atomic_json(BASE/'registrations'/(cid+'.json'),record,immutable=True)
        for scope in ('screen','full'):
            dependencies=[test_job,'REF_J102_'+scope,'REF_C205_'+scope]
            if scope=='full':dependencies.append(cid+'_screen')
            register_job({'id':cid+'_'+scope,'kind':'research','priority':priority,'depends_on_success':dependencies,
                'candidate_id':cid,'family':family,'argv':['scripts/evaluate_next5_candidate.py','--candidate',cid,'--scope',scope],
                'description':entry['scientific_question']+' '+scope,'retry_safe':False})
        register_job({'id':'BOOT_'+cid,'kind':'validation','priority':priority+5,'depends_on_success':[cid+'_full','DIAG002_BOOTSTRAP_CONTRACT'],
            'argv':['scripts/bootstrap_next5_candidate.py','--candidate',cid],'description':'Frozen paired ResearchPool whole-series uncertainty for '+cid,'retry_safe':False})
    progress('Family '+family+' exact preregistrations frozen',str(len(entries))+' distinct locked hypotheses, fixed source/learner/blend; feature tests passed')
