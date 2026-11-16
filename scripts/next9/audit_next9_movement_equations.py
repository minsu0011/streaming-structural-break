"""Validate every frozen screening ridge by its TRAIN normal equations.

No solve, refit, model update, new candidate, or break label is used. Rebuild
the admitted causal design and auxiliary responses, then verify fitted
centers/scales/intercepts, stationarity equations, and held-out group MSE.
"""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now,csv_write
from src.next9.boundary import install_workspace_boundary,target_free_npz

def main(part):
    assert read_json(BASE/'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json')['status']=='PASS'
    violations=install_workspace_boundary()
    with target_free_npz() as accesses:
        import numpy as np
        from src.next9.data import FeatureBank,selection_splits
        from src.next9.movement_screen import feature_inputs,auxiliary_targets,group_weights,grouped_means
        bank=FeatureBank()
        with np.load(BASE/'state_cache/INDEX.npz') as z:rows=z['global_row']
        ids=bank.row_ids[rows];groups=np.asarray([bank.guard.split.groups[int(s)] for s in ids],np.int32);times=bank.times[rows]
        common=np.isin(rows,bank.forecast_rows(rows,16));plans=[]
        for fold in range(5):
            for inner in [None,0,1,2]:
                for split,scope in enumerate(selection_splits(bank,fold,inner)):
                    for early in [False,True]:
                        tr=np.flatnonzero(common & np.isin(ids,list(scope.training_ids)) & ((times+16<=199) if early else True))
                        va=np.flatnonzero(common & np.isin(ids,list(scope.validation_ids)) & ((times>=400) if early else True))
                        scope.admit_training(ids[tr]);scope.admit_validation(ids[va])
                        name=f'outer_{fold}_'+('main' if inner is None else f'inner_{inner}')+f'_selection_{split}_'+('EARLY' if early else 'NORMAL')
                        plans.append((name,scope,early,tr,va,group_weights(groups[tr])))
        root=BASE/'audit/movement_equations'/f'part_{part}';summary=[]
        if (root/'DONE.json').exists():raise FileExistsError('Existing complete audit is not repeated')
        for j in list(range(153))[part::4]:
            out=root/f'F{j:03d}.json'
            if out.exists():summary.append(read_json(out));continue
            with activity(f'All120frozen TRAIN ridge normal equations and heldout MSE, F{j:03d}','FINAL_AUDIT','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                statepath=BASE/'state_cache'/f'F{j:03d}'/'STATES.npz'
                assert digest(statepath)==read_json(statepath.parent/'DONE.json')['states_sha256']
                with np.load(statepath) as z:state={k:z[k] for k in z.files}
                x=feature_inputs(bank,rows,j,state);y=np.zeros((len(rows),15),np.float32)
                y[common]=auxiliary_targets(bank,rows[common],j,state['S1_DIRECTION'][common]);records=[]
                for name,scope,early,tr,va,w in plans:
                    directory=BASE/'movement_screen'/f'F{j:03d}'/name;done=read_json(directory/'DONE.json')
                    for file,sha in done['files_sha256'].items():assert digest(directory/file)==sha
                    fitted=read_json(directory/'MODEL.json');training=fitted['training']
                    for key,value in scope.provenance().items():assert training[key]==value
                    assert training['rows']==len(tr) and training['alpha']==10 and training['break_label_or_tau_used'] is False
                    xx=x[tr].astype(float);yy=y[tr].astype(float)
                    center=np.asarray(fitted['center']);scale=np.asarray(fitted['scale']);intercept=np.asarray(fitted['intercept']);coef=np.asarray(fitted['coefficients'])
                    np.testing.assert_allclose(center,np.average(xx,axis=0,weights=w),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(scale,np.maximum(np.sqrt(np.average((xx-center)**2,axis=0,weights=w)),1e-6),rtol=0,atol=1e-12)
                    np.testing.assert_allclose(intercept,np.average(yy,axis=0,weights=w),rtol=0,atol=1e-12)
                    z=(xx-center)/scale;gram=z.T@(z*w[:,None]);gram.flat[::10]+=10.
                    rhs=z.T@((yy-intercept)*w[:,None]);residual=gram@coef-rhs
                    relative=float(np.max(np.abs(residual))/max(1.,float(np.max(np.abs(rhs)))))
                    assert relative<=1e-10,(j,name,relative)
                    prediction=((x[va].astype(float)-center)/scale)@coef+intercept
                    mse=[]
                    current=bank.values[rows[va],j]
                    for k in range(3):
                        target=y[va,5*k]
                        gids,loss=grouped_means(np.column_stack([(target-current)**2,target**2,(target-np.clip(prediction[:,5*k],-12,12))**2]),groups[va])
                        mse.append(loss)
                    rebuilt=np.stack(mse,axis=1)
                    with np.load(directory/'GROUP_LOSSES.npz') as z:
                        np.testing.assert_array_equal(gids,z['group_id']);error=float(np.max(np.abs(rebuilt-z['loss'])))
                    assert error<=1e-10,(j,name,error)
                    records.append(dict(procedure=name,normal_equation_relative_residual=relative,heldout_group_MSE_max_error=error,
                        training_scope_sha256=training['actual_training_ids_sha256'],strict_early=early,new_fit_calls=0))
                r=dict(status='PASS',timestamp=utc_now(),feature=j,checked_models=len(records),records=records,
                    no_linear_solve_or_new_fit=True,no_break_labels_or_tau=True)
                atomic(out,r);summary.append(r)
        assert not violations
        atomic(root/'DONE.json',dict(status='PASS',timestamp=utc_now(),part=part,features=[r['feature'] for r in summary],
            checked_models=sum(r['checked_models'] for r in summary),new_fit_calls=0,NPZ_members=sorted(set(accesses)),
            maximum_normal_equation_relative_residual=max(v['normal_equation_relative_residual'] for r in summary for v in r['records']),
            maximum_heldout_group_MSE_error=max(v['heldout_group_MSE_max_error'] for r in summary for v in r['records'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,choices=range(4),required=True);main(p.parse_args().part)
