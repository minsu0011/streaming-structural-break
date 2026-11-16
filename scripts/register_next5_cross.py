"""Register the final three finite cross-family slots after parent/source freeze."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5 import cross,model
from src.next5.scheduler import register_job


def main():
    state=read_json(BASE/'jobs/X000_FEATURE_CONTRACTS/STATE.json');assert state['status']=='COMPLETE' and state['exit_code']==0
    xml=BASE/'validation/X_FEATURE_CONTRACTS.xml';tree=ET.parse(xml);assert not tree.findall('.//failure') and not tree.findall('.//error')
    policy=read_json(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json');family=read_json(ROOT/'HYPOTHESIS_FAMILY_LOCK_NEXT5.json')['families']['X']
    files=('scripts/evaluate_next5_candidate.py','src/next5/cache.py','src/next5/data.py','src/next5/guard.py','src/next5/model.py','src/next5/references.py','src/scoring/ts_auc.py')
    for cid,kind,question in [
      ('X1_SCALE_PERSISTENCE','scale_persistence','Can fixed scale-by-persistence interactions add beyond the two parent evidence banks?'),
      ('X2_SCALE_TAIL','scale_tail','Can scale likelihood interact with H-relative energy PIT to improve real-data ranking?'),
      ('X3_DEPENDENCE_SCALE','dependence_scale','Can a fixed dependence-by-scale interaction add beyond standalone complements?')]:
        assert cid in family['slots'];settings=cross.locked_settings()[kind]
        record={'experiment_id':cid,'family':'X','parent':[p['experiment_id'] for p in settings['parents']],
            'created_utc':utc_now(),'created_before_score':True,'scientific_question':question,
            'mechanism':'Fixed causal parent feature banks plus eight explicitly named bounded products; no learned parent predictions or meta weights.',
            'exact_feature_change':{'unchanged_parent_banks':settings['parents'],'feature_count':len(cross.feature_names(settings)),
                'bounded_products':settings['interactions'],'formula':'tanh(left / left_scale) * tanh(right / right_scale)'},
            'exact_model_change':'One fixed W3 LightGBM expert, 0.9 matched ResearchPool BL80 + 0.1 expert.',
            'parameter_menu':family['menu'],'screen_gate':policy['screen'],'full_gate':'Passed frozen screen, then five locked PRIMARY ResearchPool folds',
            'promotion_gate':policy['performance_gate'],'stop_rule':'No product, weight, parent or seed search after this registration; hard budgets and saturation apply.',
            'expected_risk':['Parent selection is ResearchPool-adaptive','Correlated parent banks can add no new information','Interaction features may duplicate tree splits','Runtime and held-out generalization still required'],
            'feature_module':'src.next5.cross','feature_settings':settings,'feature_names':cross.feature_names(settings),
            'feature_sha256':cross.implementation_hash(settings),'model_source_sha256':model.source_hash(),
            'execution_source_sha256':{n:digest(ROOT/n) for n in files},
            'parameters':{'learner':policy['learner'],'learner_params':policy['learner_params'],'weighting':policy['weighting']},
            'combination':{'ResearchPool_BL80':.9,'new_model':.1},'policy_sha256':digest(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json'),
            'family_lock_sha256':digest(ROOT/'HYPOTHESIS_FAMILY_LOCK_NEXT5.json'),'parent_selection_sha256':digest(BASE/'locks/X_PARENT_SELECTION.json'),
            'settings_lock_sha256':digest(BASE/'locks/X_SETTINGS_LOCK.json'),'feature_contract_test_sha256':digest(xml),'shadow_data_used':False}
        atomic_json(BASE/'registrations'/(cid+'.json'),record,immutable=True)
        for scope in ('screen','full'):
            dependencies=['X000_FEATURE_CONTRACTS','REF_J102_'+scope,'REF_C205_'+scope]
            if scope=='full':dependencies.append(cid+'_screen')
            register_job({'id':cid+'_'+scope,'kind':'research','priority':70,'depends_on_success':dependencies,'candidate_id':cid,'family':'X',
                'argv':['scripts/evaluate_next5_candidate.py','--candidate',cid,'--scope',scope],'description':question+' '+scope,'retry_safe':False})
        register_job({'id':'BOOT_'+cid,'kind':'validation','priority':75,'depends_on_success':[cid+'_full','DIAG002_BOOTSTRAP_CONTRACT'],
            'argv':['scripts/bootstrap_next5_candidate.py','--candidate',cid],'description':'Frozen paired whole-series uncertainty for '+cid,'retry_safe':False})
        register_job({'id':'FORENSIC_'+cid,'kind':'validation','priority':80,'depends_on_success':['BOOT_'+cid,'DIAG003_FRACTURE_CONTRACT'],
            'argv':['scripts/forensic_next5_candidate.py','--candidate',cid],'description':'Fixed paired time/age/H/mechanism fractures for '+cid,'retry_safe':False})
    progress('Final X slots registered','Three distinct cross-family questions; all seven family menus now exactly registered without extra micro-search')


if __name__=='__main__':main()
