"""Fit broad BL-conditioned screening with admitted and audited OOF inputs."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
import numpy as np
from src.next8.common import BASE,read_json,atomic,digest,utc_now,note
from src.next8.cache import FeatureCache
from src.next8.data import outer_scope,baseline_oof,rows_for
from src.next8.screening import sampled_rows
from src.next8.baseline_conditioned import fit_baseline_conditioned
from scripts.next8.fit_next8_models import prepare_inputs


def main(fold):
    assert read_json(BASE/'BASELINE_CONDITIONAL_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock_path=BASE/'BASELINE_CONDITIONAL_IMPLEMENTATION_LOCK_NEXT8.json';lock=read_json(lock_path)
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();scope=outer_scope(fold,cache.data)
    sp=BASE/'h_context_selection'/scope.name;gp=BASE/'parent_selection'/scope.name
    assert digest(sp/'SELECTION.json')==read_json(sp/'DONE.json')['selection_sha256']
    assert digest(gp/'GRAPH.json')==read_json(gp/'DONE.json')['graph_sha256']
    selection=read_json(sp/'SELECTION.json');graph=read_json(gp/'GRAPH.json')
    train_cross,full_cross,state_provenance=prepare_inputs(cache,fold,None,False,scope,selection,graph)
    tr=sampled_rows(cache,scope.training_ids);va=sampled_rows(cache,scope.validation_ids)
    full_rows=rows_for(cache.meta,scope.validation_ids);valid_cross=full_cross[cache.meta['time_online'][full_rows]%8==0]
    train_source=baseline_oof(fold,True,cache.data);valid_source=baseline_oof(fold,False,cache.data)
    aligned=[]
    for rows,source in ((tr,train_source),(va,valid_source)):
        assert np.all(np.diff(source['global_row'])>0)
        at=np.searchsorted(source['global_row'],rows)
        np.testing.assert_array_equal(source['global_row'][at],rows)
        np.testing.assert_array_equal(source['dataset_id'][at],cache.meta['dataset_id'][rows])
        aligned.append(source['prediction'][at])
    p=BASE/'baseline_conditioned'/scope.name;p.mkdir(parents=True,exist_ok=True)
    if (p/'DONE.json').exists():raise RuntimeError('Completed BL-conditioned screening cannot be silently rerun')
    if not (p/'START.json').exists():atomic(p/'START.json',{'status':'FROZEN','timestamp':utc_now(),
        'scope':scope.provenance(),'implementation_lock_sha256':digest(lock_path),
        'baseline_provenance_audit_sha256':digest(BASE/'BASELINE_PROVENANCE_NEXT8.json'),
        'feature_selection_sha256':digest(sp/'SELECTION.json'),'state_input_provenance':state_provenance,
        'training_baseline_prediction_sha256':digest(ROOT/f'artifacts/next7/nested_base/fold_{fold}/BL80_INNER_OOF.npz'),
        'outer_baseline_sources':'Audited corresponding ResearchPool outer J102/C205 predictions; no full-DEV model'})
    fit_baseline_conditioned(cache,scope,selection,tr,va,*aligned,train_cross,valid_cross,p)
    note('BL80 conditional feature information audit frozen',scope.name+'; no outer labels or feedback to selected M models','B6_BL80_REDUNDANCY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    main(p.parse_args().fold)
