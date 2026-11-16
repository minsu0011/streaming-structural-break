"""Single outcome-release barrier after all registered fits/predictions finish."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity,note
from src.next8.cache import FeatureCache
from src.next8.data import outer_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.h_context import HStateSpec,CONTEXT_MENU

PREDICTION_ROOTS=('selection','parent_selection','h_context_selection','forecasts','transitions',
    'state_forecasts','graph_states','meta_state_inputs','early_state_inputs','models','integration',
    'orientation_refits','early_baseline','baseline_conditioned','early_integration')


def main():
    if (BASE/'PREDICTION_FREEZE_NEXT8.json').exists():raise RuntimeError('Outer release is once only')
    admission=read_json(BASE/'PREDICTION_FREEZE_TASKS_NEXT8.json')
    for task in admission['required_task_ids']:
        p=BASE/'runtime/queue'/task/'TERMINAL.json'
        if not p.exists() or read_json(p)['status']!='COMPLETE':raise RuntimeError('Required fitting task incomplete '+task)
    sources={}
    for p in sorted(BASE.glob('*IMPLEMENTATION_LOCK_NEXT8.json')):
        r=read_json(p)
        for name,sha in r['source_sha256'].items():
            assert digest(ROOT/name)==sha,name
            if name in sources:assert sources[name]==sha,name
            sources[name]=sha
    cache=FeatureCache();maps={}
    with activity('Freeze all outer predictions and evaluation-only H maps','PREDICTION_RELEASE','B9_NESTED_SELECTION'):
        for early in (False,True):
            for fold in range(5):
                scope=early_scope(outer_scope(fold,cache.data),early)
                maps[scope.name]={kind:vars(HStateSpec.fit(cache,scope,kind)) for kind in CONTEXT_MENU}
        path=BASE/'EVALUATION_DOMAIN_MAPS_NEXT8.json'
        contract={'status':'FROZEN','scopes':maps,'fit':'Each original outer TRAIN H only, equal per-series quantiles; no Online values or labels','same_maps_reused_during_scoring':True}
        if path.exists():
            old=read_json(path)
            for k,v in contract.items():assert old[k]==v,k
        else:atomic(path,{'timestamp':utc_now(),**contract})
        files={path.name:digest(path)}
        for name in PREDICTION_ROOTS:
            for p in sorted((BASE/name).rglob('*')):
                if p.is_file() and p.suffix in ('.json','.npz','.joblib'):
                    files[p.relative_to(BASE).as_posix()]=digest(p)
        for p in sorted(BASE.glob('*.json')):
            if any(word in p.name for word in ('LOCK','POLICY','SUPPLEMENT','GATE','BUDGET','MENU','PROVENANCE','PREFLIGHT')):
                files[p.name]=digest(p)
        atomic(BASE/'PREDICTION_FREEZE_NEXT8.json',{'status':'FROZEN','timestamp':utc_now(),
            'required_completed_task_ids':admission['required_task_ids'],'required_task_contract_sha256':digest(BASE/'PREDICTION_FREEZE_TASKS_NEXT8.json'),
            'files_sha256':files,'source_sha256':sources,'first_new_outer_score_released_before_this_barrier':False,
            'all_five_normal_and_early_procedures_frozen':True,'all_required_early_inner_fits_complete_or_explicitly_unused':True,
            'after_barrier':'Scoring, fixed counterexamples, execution audits, figures and reports only. No candidate settings or feature/parent/rule selection changes from outer outcomes.',
            'protected_assets_used':False,'historical_adaptive_DEV_usage_not_erased_by_this_barrier':True})
    note('Release frozen outer predictions for once-only scoring','All registered fits complete; no outer-driven model revisions','B9_NESTED_SELECTION')


if __name__=='__main__':main()
