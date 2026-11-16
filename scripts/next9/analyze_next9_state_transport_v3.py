"""Post hoc covariate/score transport diagnosis; no labels, refits or new scores."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now,csv_write
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(fold):
    violations=install_workspace_boundary()
    import numpy as np
    import joblib
    from src.next9.data import FeatureBank,outer_scope
    from src.next9.joint_design import assemble
    bank=FeatureBank();scope=outer_scope(bank,fold);root=BASE/'state_transport'/f'outer_{fold}'
    if (root/'DONE.json').exists():raise FileExistsError('Transport already diagnosed')
    families=['S1_DIRECTION','S2_DIRECTION_RUN','S3_DIRECTION_SLOPE','S4_REVERSAL']
    results=[];frequencies=[];source_hashes={}
    with target_free_npz() as members:
        for mode in ['NORMAL','EARLY']:
            directory=BASE/'joint_heads'/scope.name/mode
            spec=read_json(directory/'SPEC_FROZEN.json');done=read_json(directory/'DONE.json')
            assert spec['scope']==scope.provenance() and not spec['rule_specs']
            for name in ['SPEC_FROZEN.json','M2.joblib','M3.joblib']:
                assert digest(directory/name)==done['files_sha256'][name]
                source_hashes[(directory/name).relative_to(BASE).as_posix()]=digest(directory/name)
            m2=joblib.load(directory/'M2.joblib');m3=joblib.load(directory/'M3.joblib')
            assert m2.estimator is not None and m3.estimator is not None
            assert m2.training['method']==m3.training['method']=='LOGISTIC'
            names=tuple(spec['input_names']);cols2=np.asarray(spec['selected_columns']['M2']);cols3=np.asarray(spec['selected_columns']['M3'])
            retained3=np.asarray([names[k] for k in cols3[m3.columns]])
            state_masks=np.asarray([[('_'+family+'_') in name for name in retained3] for family in families])
            shared_mask=~np.any(state_masks,axis=0)
            maps={int(j):np.asarray(v) for j,v in spec['run_maps'].items()}
            sets={'TRAIN':bank.sampled_rows(scope.training_ids,cutoff=199 if mode=='EARLY' else None),
                  'HELDOUT':bank.sampled_rows(scope.validation_ids,start=400 if mode=='EARLY' else 0,stride=1)}
            frozen_directory=BASE/'frozen_predictions'/scope.name/mode
            frozen=read_json(frozen_directory/'FROZEN.json')
            assert frozen['scope']==scope.provenance()
            assert digest(frozen_directory/'PREDICTIONS.npz')==frozen['prediction_sha256']
            with np.load(frozen_directory/'PREDICTIONS.npz') as z:expected_rows=z['global_row']
            np.testing.assert_array_equal(sets['HELDOUT'],expected_rows)
            source_hashes[(frozen_directory/'FROZEN.json').relative_to(BASE).as_posix()]=digest(frozen_directory/'FROZEN.json')
            by_set={}
            for label,rows in sets.items():
                assert len(rows)==(done['training_rows'] if label=='TRAIN' else frozen['rows'])
                with activity(f'outer{fold} {mode} {label} fixed M3/M2 centered score decomposition and state occupancy','STATE_TRANSPORT_DIAGNOSTIC','B4_EARLY_LATE_REPLICATION'):
                    ids=bank.row_ids[rows];unique,inverse,counts=np.unique(ids,return_inverse=True,return_counts=True)
                    weights=1./counts[inverse];total=float(weights.sum())
                    sum_x=np.zeros(len(names));sum_parts=np.zeros(5);square_parts=np.zeros(5)
                    sum_gap=0.;max_error=0.
                    for start in range(0,len(rows),8192):
                        rr=rows[start:start+8192];w=weights[start:start+8192]
                        x,actual=assemble(bank,rr,spec['candidates'],{},maps,{},'M5');assert actual==names
                        z2=m2.transform(x[:,cols2]);z3=m3.transform(x[:,cols3])
                        coef2=m2.estimator.coef_[0];coef3=m3.estimator.coef_[0]
                        score2=z2@coef2+m2.estimator.intercept_[0];score3=z3@coef3+m3.estimator.intercept_[0]
                        parts=[z3[:,shared_mask]@coef3[shared_mask]-z2@coef2]
                        parts.extend(z3[:,mask]@coef3[mask] for mask in state_masks)
                        parts=np.column_stack(parts)
                        delta_intercept=float(m3.estimator.intercept_[0]-m2.estimator.intercept_[0])
                        error=float(np.max(np.abs(parts.sum(axis=1)+delta_intercept-(score3-score2))))
                        assert error<=1e-10,error
                        max_error=max(max_error,error);sum_x+=w@x
                        sum_parts+=w@parts;square_parts+=w@(parts*parts);sum_gap+=float(w@(score3-score2))
                    mean=sum_parts/total;sd=np.sqrt(np.maximum(square_parts/total-mean*mean,0))
                    state_frequency=sum_x/total;by_set[label]=state_frequency
                    results.append(dict(fold=fold,mode=mode,partition=label,rows=len(rows),original_groups=len(unique),
                        centered_component_names=['shared_coefficients_changed',*families],
                        mean_centered_logit_components=mean.tolist(),SD_centered_logit_components=sd.tolist(),
                        intercept_difference=delta_intercept,mean_logit_M3_minus_M2=sum_gap/total,
                        maximum_decomposition_error=max_error,equal_original_group_weight=True))
            for j in spec['candidates']:
                for family in families:
                    loc=[k for k,name in enumerate(names) if name.startswith(f'F{j}_{family}_')]
                    a=by_set['TRAIN'][loc];b=by_set['HELDOUT'][loc]
                    assert abs(a.sum()-1)<1e-8 and abs(b.sum()-1)<1e-8
                    frequencies.append(dict(fold=fold,mode=mode,feature=j,family=family,
                        total_variation=float(np.abs(a-b).sum()/2),
                        state_codes='|'.join(name.rsplit('_',1)[1] for k,name in enumerate(names) if k in loc),
                        TRAIN_frequencies='|'.join(map(str,a)),HELDOUT_frequencies='|'.join(map(str,b)),
                        support='All frozen onehot states, including UNKNOWN and TRAIN sparse-run parent fallback',
                        classification_labels_used=False))
    assert not violations
    csv_write(str((root/'STATE_FREQUENCIES.csv').relative_to(BASE)),frequencies)
    atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),fold=fold,records=results,
        frequencies_sha256=digest(root/'STATE_FREQUENCIES.csv'),source_sha256=source_hashes,NPZ_members=sorted(set(members)|set(bank.decoded_NPZ_members)),
        exploratory_after_outer_model_results=True,no_labels_or_refits=True,no_candidate_or_model_changes=True,
        component_attribution_not_causal_or_unique_under_recoding=True,
        interpretation='Fixed standardized linear-score identity and group-weighted covariate drift. No ablated prediction model or new TS-AUC candidate is evaluated.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5));main(p.parse_args().fold)
