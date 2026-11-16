"""Freeze direct direction contrast domains before any new outer score."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,atomic,digest,utc_now,activity,note
from src.next8.cache import FeatureCache
from src.next8.data import outer_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.features import DETAILED
from src.next8.screening import sampled_rows
from src.next8.joint_models import orientation_states
from src.next8.direction_domains import training_cases
from src.next8.diagnostics import original_groups
from scripts.next8.fit_next8_models import prepare_inputs


def main():
    if (BASE/'PREDICTION_FREEZE_NEXT8.json').exists():raise RuntimeError('Direction domains must precede outer score release')
    lock=read_json(BASE/'DIRECTION_DOMAIN_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();scopes={};sources={}
    for early in (False,True):
        for fold in range(5):
            scope=early_scope(outer_scope(fold,cache.data),early)
            with activity('Freeze TRAIN direction support '+scope.name,'DIRECTION_DOMAIN_FREEZE','B1_CONDITIONAL_DIRECTION'):
                sp=BASE/'h_context_selection'/scope.name/'SELECTION.json';selection=read_json(sp)
                gp=BASE/'parent_selection'/scope.name/'GRAPH.json';graph=read_json(gp)
                sources[sp.relative_to(BASE).as_posix()]=digest(sp);sources[gp.relative_to(BASE).as_posix()]=digest(gp)
                records=[]
                if selection['variable_features']:
                    train_cross,_,provenance=prepare_inputs(cache,fold,None,early,scope,selection,graph)
                    rows=sampled_rows(cache,scope.training_ids,199 if early else None);scope.admit_training(cache.meta['dataset_id'][rows])
                    groups=original_groups(cache,rows);y=cache.meta['target'][rows]
                    for j in selection['variable_features']:
                        own=orientation_states(cache,rows,j,selection);f=cache.current(rows,[j])[:,0];cross_state=train_cross[:,DETAILED.index(j)]
                        for model,cross in (('O2_OWN',False),('O3_GRAPH',True)):
                            p=BASE/'orientation_refits'/scope.name/f'F{j:03d}'/model/'DONE.json';refit=read_json(p)
                            sources[p.relative_to(BASE).as_posix()]=digest(p)
                            cases=training_cases(f,own,cross_state,y,groups,refit,cross=cross)
                            records.append({'feature':j,'orientation_model':model,'selection_state_family':selection['orientation_state_specs'][str(j)]['kind'],
                                'orientation_estimator_family':'FACTORISED_T3_PRE_CURRENT_V1' if cross else selection['orientation_state_specs'][str(j)]['kind'],
                                'training_cases':cases,'training_state_provenance':provenance,
                                'inner_entry_rules':selection['classifications'][j]['repeated_conditional_rules']})
                scopes[scope.name]={'scope':scope.provenance(),'records':records,'training_max_time':199 if early else None}
    atomic(BASE/'DIRECTION_DOMAIN_LOCK_NEXT8.json',{'status':'FROZEN','timestamp':utc_now(),'scopes':scopes,
        'source_receipts_sha256':sources,'no_new_model_fit':True,'no_outer_labels_used':True,'no_variable_list_or_state_estimator_change':True,
        'graph_contrast':'Hold own/H condition fixed while testing cross-state directions; avoid changing own-state mixture masquerading as graph direction',
        'fixed_nonlinear_kink':'Both one-sided TRAIN derivatives must share each claimed sign; ambiguous cases remain unresolved',
        'implementation_lock_sha256':digest(BASE/'DIRECTION_DOMAIN_IMPLEMENTATION_LOCK_NEXT8.json')})
    atomic(BASE/'DIRECTION_DOMAIN_FREEZE_DONE_NEXT8.json',{'status':'COMPLETE','timestamp':utc_now(),
        'direction_domain_sha256':digest(BASE/'DIRECTION_DOMAIN_LOCK_NEXT8.json'),'new_outer_scores_seen':False})
    note('Freeze direct orientation contrast support','All ten scopes; graph contrasts hold own/H condition fixed','B1_CONDITIONAL_DIRECTION')


if __name__=='__main__':main()
