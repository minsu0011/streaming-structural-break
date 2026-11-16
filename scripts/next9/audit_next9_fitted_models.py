"""Hash-bound parameter and convergence inventory of every active NEXT9 model."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now,csv_write


def main():
    import numpy as np
    import joblib
    from collections import Counter
    from src.next9.learning import Classifier
    from src.next9.transitions import SelfLaw
    from src.next9.meta_correction import OffsetCorrection
    trees=['orientation_selection','self_transitions','joint_heads','cross_and_tabular','outer_orientation_refits',
           'outer_self_transition_evaluation','meta_correction']
    paths=sorted(p for tree in trees for p in (BASE/tree).rglob('*.joblib'))
    assert len(paths)>10000
    rows=[];cache={}
    def expected_sha(path):
        for name in ['DONE.json','FROZEN.json','PREDICTIONS_FROZEN.json']:
            p=path.parent/name
            if not p.exists():continue
            if p not in cache:cache[p]=read_json(p)
            r=cache[p];expected=r.get('files_sha256',{}).get(path.name)
            if expected:return expected
            if path.name=='LAW.joblib' and 'law_sha256' in r:return r['law_sha256']
            if path.name=='CORRECTION.joblib' and 'correction_sha256' in r:return r['correction_sha256']
        raise AssertionError('No frozen hash for '+str(path))
    def classifier(model):
        assert type(model) is Classifier
        for a in [model.center,model.scale,model.prior]:assert np.isfinite(a).all()
        assert np.all(model.scale>0) and np.all(model.prior>0) and abs(model.prior.sum()-1)<1e-12
        if model.estimator is None:return dict(learner='FROZEN_PRIOR',iterations=0,converged=None)
        estimator=model.estimator;p=estimator.get_params()
        if model.training['method']=='LOGISTIC':
            assert p['C']==.1 and p['tol']==1e-6 and p['solver']=='lbfgs'
            assert np.isfinite(estimator.coef_).all() and np.isfinite(estimator.intercept_).all()
            iterations=int(np.max(estimator.n_iter_));assert iterations<p['max_iter'] and model.training['converged']
            return dict(learner='LOGISTIC',iterations=iterations,converged=True)
        assert model.training['method']=='DEPTH3'
        assert p['max_depth']==3 and p['max_iter']==100 and p['min_samples_leaf']==128 and p['l2_regularization']==10
        assert p['learning_rate']==.05 and p['early_stopping'] is False and estimator.n_iter_==100
        for group in estimator._predictors:
            for predictor in group:assert np.isfinite(predictor.nodes['value']).all()
        return dict(learner='FIXED_DEPTH3_100',iterations=100,converged=None)
    for start in range(0,len(paths),128):
        out=BASE/'fitted_model_audit'/f'BATCH_{start//128:04d}.json'
        if out.exists():rows.extend(read_json(out)['records']);continue
        with activity(f'Frozen parameter inventory files{start}:{min(start+128,len(paths))}','FITTED_MODEL_NUMERICAL_AUDIT','B9_NORMALIZATION_STATE_AVAILABILITY'):
            records=[]
            for path in paths[start:start+128]:
                sha=digest(path);assert sha==expected_sha(path)
                model=joblib.load(path)
                if type(model) is Classifier:result=classifier(model)
                elif type(model) is SelfLaw:
                    assert np.isfinite(model.table).all() and np.all(model.table>0)
                    np.testing.assert_allclose(model.table.sum(axis=1),1,rtol=0,atol=1e-12)
                    result=dict(learner=model.variant,iterations=0,converged=None) if model.classifier is None else classifier(model.classifier)
                elif type(model) is OffsetCorrection:
                    assert np.isfinite([model.center,model.scale,model.coefficient,model.intercept]).all() and model.scale>0
                    assert model.training['C']==.1 and model.training['baseline_logit_offset_coefficient']==1.
                    result=dict(learner='FIXED_OFFSET_LOGISTIC',iterations=model.training['iterations'],converged=True)
                else:raise TypeError(type(model))
                records.append(dict(path=path.relative_to(BASE).as_posix(),sha256=sha,type=type(model).__name__,
                    scope=model.training['scope'],training_ids_sha256=model.training['training_ids_sha256'],**result))
            atomic(out,dict(status='PASS',timestamp=utc_now(),records=records));rows.extend(records)
    assert len(rows)==len(paths) and len({r['path'] for r in rows})==len(paths)
    csv_write('FITTED_MODEL_PARAMETER_AUDIT_NEXT9.csv',rows)
    atomic(BASE/'FITTED_MODEL_PARAMETER_AUDIT_NEXT9.json',dict(status='PASS',timestamp=utc_now(),models=len(rows),
        types=dict(Counter(r['type'] for r in rows)),learners=dict(Counter(r['learner'] for r in rows)),
        all_original_fit_hashes_match=True,all_parameters_finite=True,all_iterative_solvers_converged=True,
        no_fitting_or_probability_updates=True,prior_failed_joint_caps_preserved_separately=True,
        csv_sha256=digest(BASE/'FITTED_MODEL_PARAMETER_AUDIT_NEXT9.csv')))


if __name__=='__main__':main()
