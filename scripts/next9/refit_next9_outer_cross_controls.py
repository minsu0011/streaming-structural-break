"""Freeze exact remove-one-parent predictions before outer evaluation."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(fold):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'OUTER_CROSS_CONTROLS_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    with target_free_npz() as accesses:
        import numpy as np
        import joblib
        from src.next9.data import FeatureBank,outer_scope
        from src.next9.cross import fit_law
        from src.next9.detailed_movement import Forecast,design
        from src.next9.io import save_npz
        bank=FeatureBank();scope=outer_scope(bank,fold);root=BASE/'outer_cross_controls'/scope.name
        if (root/'DONE.json').exists():return
        modes=[]
        for mode in ['NORMAL','EARLY']:
            source=BASE/'cross_and_tabular'/scope.name/mode;done=read_json(source/'DONE.json')
            assert done['scope']==scope.provenance()
            for name,sha in done['files_sha256'].items():assert digest(source/name)==sha
            spec=read_json(source/'SPEC_FROZEN.json');edges=spec['provisional_edges'];records=[]
            for edge in edges:
                j,p=edge['destination'],edge['parent'];out=root/mode/f'EDGE_{p}_TO_{j}'
                if (out/'DONE.json').exists():records.append(dict(edge=edge,sha256=digest(out/'DONE.json')));continue
                with activity(f'{scope.name} {mode} exact remove-one-parent frozen control {p} to{j}',
                              'OUTER_CROSS_REFIT','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                    directory=source/f'FINAL_DESTINATION_{j}'
                    for name,sha in spec['final_forecast_files_sha256'][str(j)].items():assert digest(directory/name)==sha
                    full=joblib.load(directory/'CROSS_LAW.joblib')
                    parents=[e['parent'] for e in edges if e['destination']==j]
                    tr=bank.forecast_rows(bank.sampled_rows(scope.training_ids),1,cutoff=199 if mode=='EARLY' else None)
                    ids=bank.row_ids[tr];groups=np.asarray([bank.guard.split.groups[int(s)] for s in ids],dtype=np.int32)
                    target=bank.pre_state(tr+1,j)['S1_DIRECTION']
                    remove=fit_law(bank,tr,j,[v for v in parents if v!=p],target,ids,groups,scope)
                    va=bank.forecast_rows(bank.sampled_rows(scope.validation_ids,start=400 if mode=='EARLY' else 0),16)
                    states=bank.pre_state(va,j);base,names=design(bank,va,j,states,'D3');pred={}
                    for method,name in [('D3','D3_FORECAST.json'),('D5','D5_FORECAST.json'),('PARENT',f'PARENT_{p}_FORECAST.json')]:
                        r=read_json(directory/name)
                        model=Forecast(r['width'],np.asarray(r['columns'],dtype=int),np.asarray(r['center']),np.asarray(r['scale']),
                            np.asarray(r['coefficient']),np.asarray(r['intercept']),r['training'])
                        x=base if method=='D3' else design(bank,va,j,states,'D5')[0] if method=='D5' else np.column_stack([base,bank.history(va,p)[:,[-1,-4,-16]]])
                        pred[method]=model.predict(x).astype(np.float32)
                    pred.update(FULL_LAW=full.probabilities(bank,va),REMOVE_PARENT_LAW=remove.probabilities(bank,va))
                    out.mkdir(parents=True,exist_ok=True);joblib.dump(remove,out/'REMOVE_PARENT_LAW.joblib')
                    save_npz(out/'PREDICTIONS.npz',global_row=va,dataset_id=bank.row_ids[va],pre_direction=states['S1_DIRECTION'],**pred)
                    atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),mode=mode,edge=edge,
                        full_parents=parents,remove_parents=[v for v in parents if v!=p],horizons=[1,4,16],outer_auxiliary_targets_used=False,
                        exact_remove_one_parent=True,prediction_sha256=digest(out/'PREDICTIONS.npz'),
                        remove_model_sha256=digest(out/'REMOVE_PARENT_LAW.joblib'),source_spec_sha256=digest(source/'SPEC_FROZEN.json')))
                    records.append(dict(edge=edge,sha256=digest(out/'DONE.json')))
            atomic(root/mode/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),mode=mode,edges=records,
                source_spec_sha256=digest(source/'SPEC_FROZEN.json'),zero_edge_branch=len(edges)==0,no_invented_edges=True))
            modes.append(dict(mode=mode,sha256=digest(root/mode/'DONE.json')))
        assert not violations
        atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),modes=modes,
            break_labels_or_tau_used=False,NPZ_members=sorted(set(accesses))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));main(p.parse_args().fold)
