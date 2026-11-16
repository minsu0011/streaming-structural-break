"""One complete feature-selection scope; queue dispatch never invents later tasks."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,atomic,digest,utc_now,note
from src.next8.cache import FeatureCache
from src.next8.screening import procedure_scope,selection_splits,screen_split,summarize_screen
from src.next8.data import FitScope


def main(fold,meta_inner,early=False):
    assert read_json(BASE/'NESTED_SCREEN_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock=read_json(BASE/'SCREEN_IMPLEMENTATION_LOCK_NEXT8.json')
    for p,sha in lock['source_sha256'].items():assert digest(ROOT/p)==sha,p
    cache=FeatureCache();parent=procedure_scope(cache,fold,meta_inner)
    if early:parent=FitScope(parent.training_ids,parent.validation_ids,parent.outer_excluded_ids,parent.name+'_EARLY_T199')
    directory=BASE/'selection'/parent.name
    directory.mkdir(parents=True,exist_ok=True)
    sources={p:digest(ROOT/p) for p in ('scripts/next8/screen_next8_scope.py','src/next8/screening.py',
        'src/next8/learning.py','src/next8/statistics.py','src/next8/cache.py','src/next8/data.py')}
    if (directory/'DONE.json').exists():raise RuntimeError('Completed selection is immutable and must not be rerun')
    if not (directory/'START.json').exists():
        atomic(directory/'START.json',{'status':'FROZEN','timestamp':utc_now(),'scope':parent.provenance(),
            'source_sha256':sources,'protocol_sha256':digest(BASE/'NEXT8_PROTOCOL_LOCK.json'),
            'outer_validation_excluded':True,'meta_validation_excluded':meta_inner is not None})
    else:assert read_json(directory/'START.json')['source_sha256']==sources
    panels=[];corrections=[]
    for k,scope in enumerate(selection_splits(cache,fold,meta_inner)):
        if early:scope=FitScope(scope.training_ids,scope.validation_ids,scope.outer_excluded_ids,scope.name+'_EARLY_T199')
        p=directory/f'split_{k}'
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json')
            for name,sha in done['records_sha256'].items():assert digest(p/name)==sha
            records=[read_json(p/f'F{j:03d}.json') for j in range(153)]
            adjustment=read_json(p/'MULTIPLICITY.json')['tests']
        else:records,adjustment=screen_split(cache,scope,p,cutoff=199 if early else None)
        panels.append(records);corrections.append(adjustment)
    result=summarize_screen(panels,corrections,parent)
    atomic(directory/'SELECTION.json',{'status':'FROZEN','timestamp':utc_now(),**result})
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':parent.provenance(),
        'selection_sha256':digest(directory/'SELECTION.json'),'source_sha256':sources,
        'all_feature_state_classification_inside_training_only':True,'new_outer_score':False})
    note('Complete training-scoped feature selection',parent.name+'; result stored, no outer score released','B9_NESTED_SELECTION')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--meta-inner',type=int,choices=range(3));p.add_argument('--early',action='store_true')
    a=p.parse_args();main(a.fold,a.meta_inner,a.early)
