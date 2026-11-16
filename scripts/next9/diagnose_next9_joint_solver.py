"""TRAIN-only stationarity diagnosis of the unscored first joint procedure."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary

violations=install_workspace_boundary()
import numpy as np
import joblib
from src.next9.data import FeatureBank,outer_scope
from src.next9.labels import ScopedLabels
from src.next9.joint_design import assemble
from src.next9.learning_converged_v2 import logistic_stationarity
bank=FeatureBank();scope=outer_scope(bank,0);labels=ScopedLabels(bank);records=[]
for mode in ['NORMAL','EARLY']:
    with activity(f'TRAIN-only capped250optimizer stationarity outer0 {mode}','SOLVER_DIAGNOSIS','B9_NORMALIZATION_STATE_AVAILABILITY'):
        directory=BASE/'joint_heads/outer_0'/mode;spec=read_json(directory/'SPEC_FROZEN.json')
        rows=bank.sampled_rows(scope.training_ids,cutoff=199 if mode=='EARLY' else None)
        y=labels.get(rows,scope,'TRAIN',early=mode=='EARLY');ids=bank.row_ids[rows]
        assert not spec['rule_specs'] and not spec['law_features']
        x,names=assemble(bank,rows,spec['candidates'],{}, {int(j):np.asarray(v) for j,v in spec['run_maps'].items()}, {},'M5')
        assert list(names)==spec['input_names']
        for method in sorted(set(spec['aliases'].values())):
            path=directory/(method+'.joblib');assert digest(path)==spec['files_sha256'][path.name]
            model=joblib.load(path)
            r=logistic_stationarity(model,x[:,spec['selected_columns'][method]],y,ids)
            records.append(dict(mode=mode,method=method,converged=model.training['converged'],iterations=model.training['iterations'],
                model_sha256=digest(path),**r))
assert not violations
atomic(BASE/'repairs/JOINT_SOLVER_TRAIN_DIAGNOSIS_NEXT9.json',dict(status='COMPLETE',timestamp=utc_now(),records=records,
    only_TRAIN_labels=True,no_outer_or_internal_quality_scores_inspected=True,no_fits=True,label_access=labels.accesses))
print(records,flush=True)
