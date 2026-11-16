"""Freeze candidate budget and complete selection algorithm before RP scoring."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE, save, digest, utc_now, note, read_json
from src.next8.protocol import contract, model_menu
from src.next8.features import DETAILED, NAMES, source_hash


def main():
    assert read_json(BASE/'PREFLIGHT_NEXT8.json')['status']=='PASS'
    assert read_json(BASE/'BASELINE_PROVENANCE_NEXT8.json')['status']=='PASS'
    assert not (BASE/'OUTER_SCORE_RELEASE_NEXT8.json').exists()
    value={'status':'FROZEN','timestamp':utc_now(),**contract(),
           'inner_mapping_source_sha256':digest(ROOT/'artifacts/next7/NEXT7_PROTOCOL_LOCK.json'),
           'feature_engine_source_sha256_at_lock':source_hash(),
           'implementation_source_sha256':{p:digest(ROOT/p) for p in ('src/next8/features.py','src/next8/data.py','src/next8/protocol.py', 'scripts/next8/lock_next8_protocol.py')},
           'first_NEXT8_RP_association_or_model_score_observed':False}
    save('NEXT8_PROTOCOL_LOCK.json',value)
    save('NEXT8_FEATURE_BUDGET.json',{'status':'FROZEN','timestamp':utc_now(),
         'catalog_coordinates':len(NAMES),'detailed_max':12,'eligible_ids':list(DETAILED),
         'eligible_names':[NAMES[j] for j in DETAILED], 'final_stable_max':6,'final_variable_max':6,
         'parents_per_feature_max':2,'cross_edges_max':12,'horizons':[1,4,16],
         'major_state_methods_max':6,'no_candidate_replacement_after_outer_results':True,
         'screening_all153_counted_separately':True,'classification_and_selection_each_training_scope':True})
    save('NEXT8_MODEL_MENU_LOCK.json',{'status':'FROZEN','timestamp':utc_now(),
         'models':model_menu(),'M0_through_M6_only':True,
         'tree':{'max_depth':3,'max_iter':100,'learning_rate':.05,'min_samples_leaf':128,'l2_regularization':10.,'early_stopping':False,'random_state':8101604},
         'ridge':{'alpha':10.,'standardization':'weighted training means/scales only','solver':'deterministic direct linear algebra'},
         'logistic':{'C':.1,'max_iter':250,'tol':1e-6,'solver':'lbfgs'},
         'no_outcome_adaptive_hyperparameter_grid':True})
    note('NEXT8 scientific protocol frozen','Three questions,12 eligible detailed features, strict nested procedure and M0-M6 gates fixed before new scores','B9_NESTED_SELECTION')


if __name__=='__main__':main()
