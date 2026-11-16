"""Freeze four distinct scale questions before any S candidate score."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5 import scale,model
from src.next5.scheduler import register_job


def main():
    state=read_json(BASE/'jobs/S000_FEATURE_CONTRACTS/STATE.json');assert state['status']=='COMPLETE' and state['exit_code']==0
    tree=ET.parse(BASE/'validation/S_FEATURE_CONTRACTS.xml');assert not tree.findall('.//failure') and not tree.findall('.//error')
    policy=read_json(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json');family=read_json(ROOT/'HYPOTHESIS_FAMILY_LOCK_NEXT5.json')['families']['S']
    menu=[
      ('S1_DIRECTIONAL_SCALE','directional','Can separate scale-decrease/increase Gaussian likelihood across raw and conditional channels add to BL80?',
       '32 features: two H-only innovation channels, windows 8/32/128 plus current, lower/upper/difference/fixed mixture.'),
      ('S2_STUDENT_SCALE','student','Does a variance-standardized t5 working likelihood retain scale evidence under tail misspecification?',
       '32 features replacing Gaussian density ratios with t5 variance ratios; same channels, windows and directional summaries.'),
      ('S4A_AUGMENTED_J_SCALE','augmented_j','Does direct supervised fusion of J102 features and t5 scale evidence improve beyond adding an expert?',
       '109 features: exact original J102 77 plus S2 32. The augmented J component replaces J at fixed 0.8, C205 remains 0.2.'),
      ('S4B_UNIFIED_SCALE_SPECIALIST','unified','Can one specialist learn complementary Gaussian/t and B2-inspired directional semitail evidence?',
       '80 features: S1 32 plus S2 32 plus 16 H-relative log-energy semitail descriptors. These use the same new innovation channels; they are not claimed byte-identical to old B2 descriptors.')]
    files=('scripts/evaluate_next5_candidate.py','src/next5/cache.py','src/next5/data.py','src/next5/guard.py',
        'src/next5/model.py','src/next5/references.py','src/scoring/ts_auc.py')
    for cid,kind,question,change in menu:
        assert cid in family['slots'];settings={'version':'next5_scale_unification_v1','kind':kind}
        combination={'C205':.2,'new_model':.8} if kind=='augmented_j' else {'ResearchPool_BL80':.9,'new_model':.1}
        registration={'experiment_id':cid,'family':'S','parent':['ResearchPool_BL80','NEXT4_B2','NEXT4_D2'],
            'created_utc':utc_now(),'created_before_score':True,'scientific_question':question,
            'mechanism':'Separate lower/upper innovation-scale evidence under fixed Gaussian/t working alternatives.',
            'exact_feature_change':change,'exact_model_change':'One fixed W3 LightGBM per outer training fold; fixed combination '+str(combination),
            'parameter_menu':{**family['menu'],'H_only_channel_cap':12.,'actual_kind':kind,'additional_search':False},
            'screen_gate':policy['screen'],'full_gate':'Only a passed frozen screen; five locked PRIMARY ResearchPool folds',
            'promotion_gate':policy['performance_gate'],'stop_rule':'No micro-search; failed screen closes slot; global 12-hypothesis saturation and hard budgets apply.',
            'expected_risk':['Estimated H null may be misspecified','Likelihood accumulation can retain shocks','Repeated ResearchPool selection remains adaptive','Added runtime must be measured'],
            'feature_module':'src.next5.scale','feature_settings':settings,'feature_names':scale.feature_names(settings),
            'feature_sha256':scale.implementation_hash(settings),'model_source_sha256':model.source_hash(),
            'execution_source_sha256':{n:digest(ROOT/n) for n in files},
            'parameters':{'learner':policy['learner'],'learner_params':policy['learner_params'],'weighting':policy['weighting']},
            'combination':combination,'policy_sha256':digest(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json'),
            'family_lock_sha256':digest(ROOT/'HYPOTHESIS_FAMILY_LOCK_NEXT5.json'),'shadow_lock_sha256':digest(ROOT/'NEXT5_SHADOW_LOCK.json'),
            'feature_contract_test_sha256':digest(BASE/'validation/S_FEATURE_CONTRACTS.xml'),'shadow_data_used':False}
        atomic_json(BASE/'registrations'/(cid+'.json'),registration,immutable=True)
        for scope in ('screen','full'):
            dependencies=['S000_FEATURE_CONTRACTS','REF_J102_'+scope,'REF_C205_'+scope]
            if scope=='full':dependencies.append(cid+'_screen')
            register_job({'id':cid+'_'+scope,'kind':'research','priority':40,'depends_on_success':dependencies,
                'candidate_id':cid,'family':'S','argv':['scripts/evaluate_next5_candidate.py','--candidate',cid,'--scope',scope],
                'description':question+' '+scope,'retry_safe':False})
    progress('S family exact source preregistrations frozen','Four distinct hypotheses, fixed learners and blends, independent density/window/prefix tests passed')


if __name__=='__main__':main()
