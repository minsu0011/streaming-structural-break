"""Freeze all single-feature O comparators before any outer target scoring."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary


def main(fold):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'OUTER_ORIENTATION_REFIT_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    import numpy as np
    import joblib
    from src.next9.data import FeatureBank,outer_scope
    from src.next9.labels import ScopedLabels
    from src.next9.orientation_contract import PAIR_MENU,fit_common_range,fit_run_fallback,orientation_design,one_hot
    from src.next9.learning import fit_classifier
    from src.next9.domains import fit_domains
    from src.next9.io import save_npz
    bank=FeatureBank();scope=outer_scope(bank,fold);labels=ScopedLabels(bank)
    root=BASE/'outer_orientation_refits'/scope.name
    if (root/'DONE.json').exists():return
    domain_spec=fit_domains(bank,scope);records=[]
    for mode in ['NORMAL','EARLY']:
        selection=read_json(BASE/'orientation_selection'/scope.name/mode/'SELECTION_FROZEN.json')
        assert selection['scope']==scope.provenance()
        tr=bank.sampled_rows(scope.training_ids,cutoff=199 if mode=='EARLY' else None)
        va=bank.sampled_rows(scope.validation_ids,start=400 if mode=='EARLY' else 0)
        y=labels.get(tr,scope,'TRAIN',early=mode=='EARLY')
        ids=bank.row_ids[tr];groups=np.asarray([bank.guard.split.groups[int(s)] for s in ids],dtype=np.int32)
        feature_records=[]
        for j in selection['candidates']:
            out=root/mode/f'F{j:03d}'
            if (out/'DONE.json').exists():feature_records.append(dict(feature=j,sha256=digest(out/'DONE.json')));continue
            if out.exists() and any(out.iterdir()):raise RuntimeError('Partial orientation refit preserved for explicit recovery')
            with activity(f'{scope.name} {mode} F{j:03d} frozen outer O comparators without heldout labels','OUTER_ORIENTATION_REFIT','B1_COMMON_CURRENT_RANGE'):
                ts=bank.pre_state(tr,j);vs=bank.pre_state(va,j)
                tf=bank.current(tr,[j])[:,0];vf=bank.current(va,[j])[:,0]
                mapping=fit_run_fallback(ts['S2_DIRECTION_RUN'],ids,scope)
                ranges=[fit_common_range(tf,ts[p[0]],ids,y,scope,p) for p in PAIR_MENU]
                def arguments(rows,values,states):
                    s=bank.series_index[rows];past=bank.history(rows,j)[:,-1]
                    level=np.where(~bank.series['valid'][s,j],3,np.where(past<bank.series['low'][s,j],0,np.where(past>bank.series['high'][s,j],2,1)))
                    return values,states['S1_DIRECTION'],mapping[states['S2_DIRECTION_RUN']],states['S4_REVERSAL'],level
                ta=arguments(tr,tf,ts);vargs=arguments(va,vf,vs)
                predictions={};models={};files={};out.mkdir(parents=True,exist_ok=True)
                for method in ['O0','O1','O2_LEVEL','O3_MOVE','O4_MOVE_RUN','O5_MOVE_REV','O6_TABULAR','STATE_ONLY']:
                    tx=orientation_design(*ta,method);vx=orientation_design(*vargs,method)
                    model=fit_classifier(tx,y,ids,groups,scope,method='DEPTH3' if method=='O6_TABULAR' else 'LOGISTIC')
                    models[method]=model;predictions[method]=model.predict_proba(vx)[:,1].astype(np.float32)
                    joblib.dump(model,out/(method+'.joblib'));files[method+'.joblib']=digest(out/(method+'.joblib'))
                for method,family,count in [('STATE_ONLY_MOVE','S1_DIRECTION',4),('STATE_ONLY_RUN','S2_DIRECTION_RUN',13)]:
                    a,b=ts[family],vs[family]
                    if family=='S2_DIRECTION_RUN':a=mapping[a];b=mapping[b]
                    model=fit_classifier(one_hot(a,count),y,ids,groups,scope)
                    models[method]=model;predictions[method]=model.predict_proba(one_hot(b,count))[:,1].astype(np.float32)
                    joblib.dump(model,out/(method+'.joblib'));files[method+'.joblib']=digest(out/(method+'.joblib'))
                predictions['F_ONLY']=predictions['O1']
                save_npz(out/'PREDICTIONS.npz',global_row=va,dataset_id=bank.row_ids[va],current_F=vf,
                    **{name:vs[name].astype(np.int8) for name in ['S1_DIRECTION','S2_DIRECTION_RUN','S3_DIRECTION_SLOPE','S4_REVERSAL']},**predictions)
                files['PREDICTIONS.npz']=digest(out/'PREDICTIONS.npz')
                coefficients={}
                for method,model in models.items():
                    if method!='O6_TABULAR':
                        coef,intercept=model.linear_parameters();coefficients[method]=dict(raw_design_coefficients=coef.tolist(),intercept=intercept)
                frozen=dict(status='FROZEN',timestamp=utc_now(),scope=scope.provenance(),mode=mode,feature=j,
                    ranges=[v.record() for v in ranges],run_mapping=mapping.tolist(),domain_spec=domain_spec,
                    coefficients=coefficients,files_sha256=files,training_rows=len(tr),prediction_rows=len(va),
                    orientation_prediction_stride=8,pre_current_history_full_contiguous16=True,heldout_labels_used=False,
                    provisional_rule=next((r for r in selection['provisional_rules'] if r['feature']==j),None),
                    selection_sha256=digest(BASE/'orientation_selection'/scope.name/mode/'SELECTION_FROZEN.json'),
                    estimator_fit_calls=sum(m.training['estimator_fit_calls'] for m in models.values()))
                atomic(out/'FROZEN.json',frozen)
                atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),feature=j,scope=scope.provenance(),
                    frozen_sha256=digest(out/'FROZEN.json'),files_sha256=files))
                feature_records.append(dict(feature=j,sha256=digest(out/'DONE.json')))
        atomic(root/mode/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),mode=mode,features=feature_records))
        records.append(dict(mode=mode,sha256=digest(root/mode/'DONE.json')))
    assert not violations
    atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),records=records,
        label_access=labels.accesses,outer_labels_used=False,implementation_sha256=digest(BASE/'OUTER_ORIENTATION_REFIT_LOCK_NEXT9.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));main(p.parse_args().fold)
