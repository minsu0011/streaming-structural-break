"""Export complete frozen lineage and explicit fitted direction-rule roles."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,atomic,read_json,digest,csv_write,utc_now,activity,note
from src.next8.structural_inventory import fitted_inventory,complete_lineage,STATE_CONTRACT_KO


def main():
    lock=read_json(BASE/'STRUCTURAL_INVENTORY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    directory=BASE/'structural_inventory';directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise RuntimeError('Existing structural inventory preserved')
    with activity('Inventory actual fitted coordinates and advance linear rules','STRUCTURAL_INVENTORY','B7_NORMALIZATION_AND_LINEAGE'):
        coordinates,rules,scopes=fitted_inventory();lineage=complete_lineage(coordinates)
        csv_write('structural_inventory/FEATURE_LINEAGE_NEXT8.csv',lineage)
        atomic(directory/'DIRECTION_RULES.json',{'status':'COMPLETE','timestamp':utc_now(),'rules':rules,'training_scope_inventory':scopes,
            'empty_rule_table_is_valid_when_no_variable_selected':True,'B_or_joint_certification_not_inferred_from_coefficients':True,
            'complete_nonlinear_derivative_not_necessarily_known_before_current_F':True})
        atomic(directory/'STATE_TRANSITION_CONTRACT.md',STATE_CONTRACT_KO)
        atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'bank_features':153,'lineage_rows':len(lineage),
            'distinct_fitted_coordinate_names':len(coordinates),'variable_rule_instances':len(rules),'scopes':len(scopes),
            'initial_lineage_preserved_sha256':digest(BASE/'FEATURE_LINEAGE_NEXT8.csv'),
            'files_sha256':{n:digest(directory/n) for n in ('FEATURE_LINEAGE_NEXT8.csv','DIRECTION_RULES.json','STATE_TRANSITION_CONTRACT.md')}})
    note('Complete actual fitted lineage and direction roles','No coefficient or state accuracy treated as direction certification','B7_NORMALIZATION_AND_LINEAGE')


if __name__=='__main__':main()
