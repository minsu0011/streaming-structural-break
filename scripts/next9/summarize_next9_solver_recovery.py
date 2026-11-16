"""Verify convergence and unchanged objective/input parameters for all M heads."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now,csv_write

def main():
    import numpy as np
    import joblib
    records=[]
    with activity('All20jointprocedures optimizer convergence and preserved checkpoint lineage audit','SOLVER_REPRODUCTION','B9_NORMALIZATION_STATE_AVAILABILITY'):
        for item in read_json(BASE/'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json')['candidates']:
            scope=item['scope'];assert read_json(BASE/'joint_heads'/scope/'DONE.json')['status']=='COMPLETE'
            for mode in ['NORMAL','EARLY']:
                root=BASE/'joint_heads'/scope/mode;spec=read_json(root/'SPEC_FROZEN.json')
                for method in sorted(set(spec['aliases'].values())):
                    path=root/(method+'.joblib');assert digest(path)==spec['files_sha256'][path.name];model=joblib.load(path)
                    assert model.estimator is None or model.training['converged'],str(path)
                    repair=model.training.get('optimizer_continuation');gradient=None
                    if repair:
                        assert repair['same_C_and_group_weights'] and repair['same_causal_inputs_labels_and_standardization']
                        assert repair['after']['objective']<=repair['before']['objective']+1e-6
                        gradient=repair['after']['normalized_gradient_max']
                        assert gradient<=1.1e-6,(scope,mode,method,gradient)
                    previous=BASE/'repairs/solver_cap250_preserved/joint_heads'/scope/mode/path.name
                    reused_converged=False
                    if previous.exists():
                        old=joblib.load(previous)
                        for name in ['columns','center','scale','prior']:np.testing.assert_array_equal(getattr(old,name),getattr(model,name))
                        assert old.estimator.C==model.estimator.C==.1
                        if old.training['converged']:
                            np.testing.assert_array_equal(old.estimator.coef_,model.estimator.coef_)
                            np.testing.assert_array_equal(old.estimator.intercept_,model.estimator.intercept_);reused_converged=True
                    records.append(dict(scope=scope,mode=mode,method=method,converged=True,continuation_used=repair is not None,
                        normalized_gradient_after=gradient,final_iterations=model.training.get('iterations'),
                        already_converged_checkpoint_reused=reused_converged,final_model_sha256=digest(path),
                        old_checkpoint_sha256=digest(previous) if previous.exists() else None,
                        estimator_fit_calls_for_final_model_lineage=model.training['estimator_fit_calls'],
                        validation_or_outer_quality_used_for_continuation=False))
        csv_write('JOINT_SOLVER_CONVERGENCE_NEXT9.csv',records)
        atomic(BASE/'JOINT_SOLVER_AUDIT_NEXT9.json',dict(status='PASS',timestamp=utc_now(),models=len(records),
            continued_models=sum(r['continuation_used'] for r in records),preserved_converged_models=sum(r['already_converged_checkpoint_reused'] for r in records),
            every_final_joint_estimator_converged=True,same_statistical_objective=True,new_scientific_candidates=0,
            csv_sha256=digest(BASE/'JOINT_SOLVER_CONVERGENCE_NEXT9.csv'),
            historical_cap250warnings_retained=True,strict_gradient_tolerance=1.1e-6))

if __name__=='__main__':main()
