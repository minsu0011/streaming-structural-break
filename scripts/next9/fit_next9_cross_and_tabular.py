"""Only-if-needed cross tests, M6, and exactly matched depth3 comparator."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary


def main(fold):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'CROSS_AND_TABULAR_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    import numpy as np
    import joblib
    from src.next9.data import FeatureBank,outer_scope,selection_splits
    from src.next9.labels import ScopedLabels
    from src.next9.cross import proposals,fit_law,path_features,qualified_horizon
    from src.next9.detailed_movement import Forecast,design as forecast_design,fit as forecast_fit
    from src.next9.movement_screen import auxiliary_targets,paired_group_summary
    from src.next9.joint_design import assemble
    from src.next9.learning import fit_classifier
    from src.next9.io import save_npz
    bank=FeatureBank();outer=outer_scope(bank,fold);labels=ScopedLabels(bank)
    root=BASE/'cross_and_tabular'/outer.name
    if (root/'DONE.json').exists():return
    with np.load(BASE/'state_cache/INDEX.npz') as z:allrows=z['global_row']
    rows=bank.forecast_rows(allrows,16)
    ids=bank.row_ids[rows];times=bank.times[rows]
    groups=np.asarray([bank.guard.split.groups[int(s)] for s in ids],dtype=np.int32)
    receipts=[]
    def restore_forecast(p):
        r=read_json(p)
        return Forecast(r['width'],np.asarray(r['columns'],dtype=int),np.asarray(r['center']),np.asarray(r['scale']),
            np.asarray(r['coefficient']),np.asarray(r['intercept']),r['training'])
    for mode in ['NORMAL','EARLY']:
        out=root/mode
        if (out/'DONE.json').exists():receipts.append(dict(mode=mode,sha256=digest(out/'DONE.json')));continue
        head_dir=BASE/'joint_heads'/outer.name/mode;head=read_json(head_dir/'DONE.json');spec=read_json(head_dir/'SPEC_FROZEN.json')
        assert head['scope']==outer.provenance()==spec['scope']
        for name,sha in head['files_sha256'].items():assert digest(head_dir/name)==sha
        candidates=spec['candidates'];rules={int(k):v for k,v in spec['rule_specs'].items()}
        proposed=proposals(candidates,rules,spec['law_features']);edges=[];edge_records=[]
        for destination,parent in proposed:
            directory=out/f'EDGE_{parent}_TO_{destination}'
            if (directory/'DONE.json').exists():
                record=read_json(directory/'DONE.json');edge_records.append(record)
                if record['qualified_horizon'] is not None:edges.append(dict(destination=destination,parent=parent,horizon=record['qualified_horizon']))
                continue
            with activity(f'{outer.name} {mode} exact remove-parent/mechanical cross {parent} to{destination}','CROSS_CONTROL','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                states=bank.pre_state(rows,destination)
                target=auxiliary_targets(bank,rows,destination,states['S1_DIRECTION'])
                next_state=bank.pre_state(rows+1,destination)['S1_DIRECTION']
                own,names=forecast_design(bank,rows,destination,states,'D3')
                ph=bank.history(rows,parent)[:,[-1,-4,-16]]
                x=np.column_stack([own,ph]);xnames=names+[f'parent{parent}_lag{k}' for k in [1,4,16]]
                panels=[]
                for split,scope in enumerate(selection_splits(bank,fold)):
                    cut=(times+16<=199) if mode=='EARLY' else np.ones(len(rows),dtype=bool)
                    tr=np.flatnonzero(cut & np.isin(ids,list(scope.training_ids)))
                    va=np.flatnonzero(cut & np.isin(ids,list(scope.validation_ids)))
                    panel=directory/f'panel_{split}'
                    if (panel/'DONE.json').exists():panels.append(read_json(panel/'DONE.json'));continue
                    droot=BASE/'detailed_movement'/outer.name/mode/f'F{destination:03d}'
                    d3=restore_forecast(droot/'D3'/f'panel_{split}'/'MODEL.json')
                    d5=restore_forecast(droot/'D5'/f'panel_{split}'/'MODEL.json')
                    assert d3.training['scope']==scope.name==d5.training['scope']
                    fitted=forecast_fit(x[tr],target[tr],ids[tr],groups[tr],scope,xnames,'D3_PLUS_PARENT')
                    vstate={k:v[va] for k,v in states.items()}
                    mechanical,_=forecast_design(bank,rows[va],destination,vstate,'D5')
                    predictions={'D3':d3.predict(own[va]),'D5':d5.predict(mechanical),'CROSS':fitted.predict(x[va])}
                    contrasts=[]
                    for k,h in enumerate([1,4,16]):
                        loss={m:(target[va,5*k]-np.clip(p[:,5*k],-12,12))**2 for m,p in predictions.items()}
                        contrasts.append(dict(horizon=h,**{m:paired_group_summary(loss[m],loss['CROSS'],groups[va]) for m in ['D3','D5']}))
                    baseline=fit_law(bank,rows[tr],destination,[],next_state[tr],ids[tr],groups[tr],scope)
                    cross=fit_law(bank,rows[tr],destination,[parent],next_state[tr],ids[tr],groups[tr],scope)
                    bp=baseline.probabilities(bank,rows[va]);cp=cross.probabilities(bank,rows[va]);ix=np.arange(len(va))
                    b_loss=-np.log(np.maximum(bp[ix,next_state[va]],1e-8));c_loss=-np.log(np.maximum(cp[ix,next_state[va]],1e-8))
                    atomic(panel/'FORECAST.json',fitted.record());joblib.dump(cross,panel/'CROSS_LAW.joblib');joblib.dump(baseline,panel/'REMOVE_PARENT_LAW.joblib')
                    record=dict(status='COMPLETE',split=split,scope=scope.provenance(),forecast_contrasts=contrasts,
                        next_state_contrast=paired_group_summary(b_loss,c_loss,groups[va]),
                        forecast_source_sha256={m:digest(droot/m/f'panel_{split}'/'MODEL.json') for m in ['D3','D5']},
                        files_sha256={n:digest(panel/n) for n in ['FORECAST.json','CROSS_LAW.joblib','REMOVE_PARENT_LAW.joblib']})
                    atomic(panel/'DONE.json',record);panels.append(record)
                horizon=qualified_horizon(panels)
                record=dict(status='COMPLETE',destination=destination,parent=parent,mode=mode,scope=outer.provenance(),
                    qualified_horizon=horizon,panels=panels,outer_certified=False,causal_parent_claim=False)
                atomic(directory/'DONE.json',record);edge_records.append(record)
                if horizon is not None:edges.append(dict(destination=destination,parent=parent,horizon=horizon))
        with activity(f'{outer.name} {mode} frozen M6 and exactly same-input depth3 comparator','CROSS_MODEL_TABULAR','B6_BL80_REDUNDANCY'):
            tr=bank.sampled_rows(outer.training_ids,cutoff=199 if mode=='EARLY' else None)
            va=bank.sampled_rows(outer.validation_ids,start=400 if mode=='EARLY' else 0,stride=1)
            train_ids=bank.row_ids[tr];train_groups=np.asarray([bank.guard.split.groups[int(s)] for s in train_ids],dtype=np.int32)
            y=labels.get(tr,outer,'TRAIN',early=mode=='EARLY')
            cross_laws={};forecast_receipts={}
            for destination in sorted(set(e['destination'] for e in edges)):
                parents=[e['parent'] for e in edges if e['destination']==destination]
                cr=bank.forecast_rows(tr,1,cutoff=199 if mode=='EARLY' else None)
                cg=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[cr]],dtype=np.int32)
                target=bank.pre_state(cr+1,destination)['S1_DIRECTION']
                law=fit_law(bank,cr,destination,parents,target,bank.row_ids[cr],cg,outer)
                directory=out/f'FINAL_DESTINATION_{destination}';directory.mkdir(parents=True,exist_ok=True)
                joblib.dump(law,directory/'CROSS_LAW.joblib');cross_laws[destination]=law
                fr=bank.forecast_rows(tr,16,cutoff=199 if mode=='EARLY' else None)
                fs=bank.pre_state(fr,destination);fy=auxiliary_targets(bank,fr,destination,fs['S1_DIRECTION'])
                fg=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[fr]],dtype=np.int32)
                base_x,names=forecast_design(bank,fr,destination,fs,'D3')
                for method in ['D3','D5']:
                    x,n=forecast_design(bank,fr,destination,fs,method)
                    fitted=forecast_fit(x,fy,bank.row_ids[fr],fg,outer,n,method)
                    atomic(directory/(method+'_FORECAST.json'),fitted.record())
                for parent in parents:
                    x=np.column_stack([base_x,bank.history(fr,parent)[:,[-1,-4,-16]]])
                    n=names+[f'parent{parent}_lag{k}' for k in [1,4,16]]
                    fitted=forecast_fit(x,fy,bank.row_ids[fr],fg,outer,n,'D3_PLUS_PARENT')
                    atomic(directory/(f'PARENT_{parent}_FORECAST.json'),fitted.record())
                forecast_receipts[str(destination)]={p.name:digest(p) for p in directory.iterdir() if p.is_file()}
            run_maps={int(k):np.asarray(v,dtype=int) for k,v in spec['run_maps'].items()};self_laws={}
            for j in spec['law_features']:
                directory=BASE/'self_transitions'/outer.name/mode/f'F{j:03d}'
                done=read_json(directory/'DONE.json');assert digest(directory/'LAW.joblib')==done['law_sha256']
                self_laws[j]=joblib.load(directory/'LAW.joblib')
            def inputs(rr):
                x,names=assemble(bank,rr,candidates,rules,run_maps,self_laws,'M5')
                pieces=[x];names=list(names)
                for j,law in cross_laws.items():
                    pieces.append(path_features(bank,rr,law));names.extend(f'cross_F{j}_{v}' for v in ['surprise','reversal','continuation','entropy'])
                return np.column_stack(pieces).astype(np.float32),tuple(names)
            x,names=inputs(tr)
            out.mkdir(parents=True,exist_ok=True)
            m6=fit_classifier(x,y,train_ids,train_groups,outer) if edges else None
            tabular=fit_classifier(x,y,train_ids,train_groups,outer,method='DEPTH3')
            if m6 is not None:joblib.dump(m6,out/'M6.joblib')
            joblib.dump(tabular,out/'TABULAR_SAME_INPUT.joblib')
            atomic(out/'SPEC_FROZEN.json',dict(status='FROZEN',timestamp=utc_now(),scope=outer.provenance(),mode=mode,proposed_edges=[list(e) for e in proposed],
                provisional_edges=edges,edge_records=edge_records,final_forecast_files_sha256=forecast_receipts,
                input_names=list(names),M6_alias='M5' if not edges else None,matched_tabular_exact_same_numeric_input=True,
                outer_labels_used=False,source_M5_spec_sha256=digest(head_dir/'SPEC_FROZEN.json')))
            del x
            with np.load(head_dir/'PREDICTIONS.npz') as z:hr=z['global_row'];hp=z['M5']
            index=np.searchsorted(hr,va);np.testing.assert_array_equal(hr[index],va)
            predictions={'M6':hp[index].copy(),'TABULAR_SAME_INPUT':np.empty(len(va),dtype=np.float32)}
            for start in range(0,len(va),32768):
                sl=slice(start,start+32768);x,nn=inputs(va[sl]);assert nn==names
                if m6 is not None:predictions['M6'][sl]=m6.predict_proba(x)[:,1]
                predictions['TABULAR_SAME_INPUT'][sl]=tabular.predict_proba(x)[:,1]
            save_npz(out/'PREDICTIONS.npz',global_row=va,dataset_id=bank.row_ids[va],**predictions)
            files={n:digest(out/n) for n in ['SPEC_FROZEN.json','PREDICTIONS.npz','TABULAR_SAME_INPUT.joblib']}
            if m6 is not None:files['M6.joblib']=digest(out/'M6.joblib')
            atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=outer.provenance(),mode=mode,
                proposed_edge_count=len(proposed),provisional_edge_count=len(edges),outer_certified_edges=None,
                M6_alias='M5' if not edges else None,files_sha256=files,outer_scores_released=False,
                estimator_fit_calls_M6=0 if m6 is None else m6.training['estimator_fit_calls'],
                estimator_fit_calls_TABULAR=tabular.training['estimator_fit_calls']))
            receipts.append(dict(mode=mode,sha256=digest(out/'DONE.json')))
    assert not violations
    atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=outer.provenance(),records=receipts,
        label_access=labels.accesses,outer_scores_released=False,implementation_sha256=digest(BASE/'CROSS_AND_TABULAR_IMPLEMENTATION_LOCK_NEXT9.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));main(p.parse_args().fold)
