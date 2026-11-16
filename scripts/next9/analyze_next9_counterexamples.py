"""Paired observation-shock memory and fixed-forecast synthetic diagnostics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,csv_write,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main():
    violations=install_workspace_boundary()
    lock=read_json(BASE/'COUNTEREXAMPLE_ANALYSIS_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    with target_free_npz() as accesses:
        import numpy as np
        from src.next9.negative_controls_v2 import BACKGROUNDS,INTERVENTIONS,SEEDS,H_LENGTHS,generate
        from src.next8.features import FeatureState
        from src.next9.replay_checks import ArrayBank
        from src.next9.movement_screen import RidgeBundle,feature_inputs
        from src.next9.detailed_movement import Forecast,design
        candidates=read_json(BASE/'candidates/outer_0/FROZEN.json')['normal_candidates'];forecasts={}
        for j in candidates:
            directory=BASE/'outer_movement_refits/outer_0/NORMAL'/f'F{j:03d}';frozen=read_json(directory/'FROZEN.json')
            for name,sha in frozen['files_sha256'].items():
                if name.endswith('_MODEL.json'):assert digest(directory/name)==sha
            r=read_json(directory/'D1_MODEL.json')
            models={'D1':RidgeBundle(np.asarray(r['center']),np.asarray(r['scale']),np.asarray(r['coefficients']),np.asarray(r['intercept']),r['training'])}
            for method in ['D3','D5']:
                r=read_json(directory/(method+'_MODEL.json'))
                models[method]=Forecast(r['width'],np.asarray(r['columns'],dtype=int),np.asarray(r['center']),np.asarray(r['scale']),np.asarray(r['coefficient']),np.asarray(r['intercept']),r['training'])
            forecasts[j]=models
        memory=[];results=[];cases=0;raw_equal_checks=0
        for background in BACKGROUNDS:
            for h_length in H_LENGTHS:
                for seed in SEEDS:
                    neutral=BASE/'negative_controls'/f'{background}_{h_length}_{seed}_NONE';r=read_json(neutral/'DONE.json')
                    assert digest(neutral/'SYNTHETIC_TRAJECTORY.npz')==r['synthetic_trajectory_sha256']
                    with np.load(neutral/'SYNTHETIC_TRAJECTORY.npz') as z:base_values=z['values'];base_state=z['S1_DIRECTION']
                    h,base_online,metadata=generate(background,'NONE',seed,h_length)
                    engine=FeatureState(h);seeds=engine.history.copy();epsilon=engine.epsilon.copy();context=engine.context.copy()
                    for intervention in INTERVENTIONS:
                        case=f'{background}_{h_length}_{seed}_{intervention}';directory=BASE/'negative_controls'/case;receipt=read_json(directory/'DONE.json')
                        assert digest(directory/'SYNTHETIC_TRAJECTORY.npz')==receipt['synthetic_trajectory_sha256']
                        with activity('Paired synthetic control and frozen forecast analysis '+case,'COUNTEREXAMPLE_FORENSIC','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                            with np.load(directory/'SYNTHETIC_TRAJECTORY.npz') as z:values=z['values'];state=z['S1_DIRECTION']
                            onset=receipt['case']['known_synthetic_onset'];np.testing.assert_array_equal(values[:onset],base_values[:onset]);np.testing.assert_array_equal(state[:onset+1],base_state[:onset+1])
                            if intervention=='SINGLE_SHOCK':
                                hh,oo,_=generate(background,intervention,seed,h_length);np.testing.assert_array_equal(hh,h);np.testing.assert_array_equal(oo[onset+1:],base_online[onset+1:]);raw_equal_checks+=1
                            delta=np.abs(values-base_values);t=np.arange(512)
                            for name,keep in [('ONSET',t==onset),('NEXT_1_15',(t>onset)&(t<=onset+15)),('NEXT_16_63',(t>=onset+16)&(t<=onset+63)),('NEXT_64_PLUS',t>=onset+64)]:
                                for j in range(153):
                                    memory.append(dict(background=background,intervention=intervention,seed=seed,H_length=h_length,feature=j,window=name,
                                        points=int(keep.sum()),mean_absolute_feature_difference=float(delta[keep,j].mean()) if keep.any() else None,
                                        max_absolute_feature_difference=float(delta[keep,j].max()) if keep.any() else None,
                                        feature_difference_above_1e_minus6_fraction=float((delta[keep,j]>1e-6).mean()) if keep.any() else None,
                                        pre_state_disagreement_fraction=float((state[keep,j]!=base_state[keep,j]).mean()) if keep.any() else None,
                                        raw_after_single_shock_exactly_neutral=intervention=='SINGLE_SHOCK',new_fits=0))
                            bank=ArrayBank(seeds,values,epsilon,context,h_length);rows=np.arange(0,496,8)
                            for j in candidates:
                                states=bank.pre_state(rows,j);models=forecasts[j]
                                predictions={'D1':models['D1'].predict(feature_inputs(bank,rows,j,states))}
                                for method in ['D3','D5']:predictions[method]=models[method].predict(design(bank,rows,j,states,method)[0])
                                for k,horizon in enumerate([1,4,16]):
                                    target=values[rows+horizon,j];current=values[rows,j];constant=float(models['D1'].intercept[5*k])
                                    loss={method:float(np.mean((target-np.clip(p[:,5*k],-12,12))**2)) for method,p in predictions.items()}
                                    loss.update(D0A=float(np.mean((target-current)**2)),D0B=float(np.mean(target**2)),TRAIN_MEAN=float(np.mean((target-constant)**2)))
                                    results.append(dict(control_family='PAIRED_SYNTHETIC_FORECAST',background=background,intervention=intervention,seed=seed,H_length=h_length,
                                        feature=j,horizon=horizon,**{m+'_MSE':v for m,v in loss.items()},
                                        D1_gain_vs_persistence=loss['D0A']-loss['D1'],D1_gain_vs_Hmedian=loss['D0B']-loss['D1'],D1_gain_vs_TRAINmean=loss['TRAIN_MEAN']-loss['D1'],
                                        same_RP_outer0_NORMAL_models_for_all_cases=True,new_fits=0,confidence_interval='Not inferred from these few paired synthetic cases',
                                        no_selection_or_real_target_quality_claim=True))
                            cases+=1
        analytic=read_json(BASE/'ANALYTIC_FIXED_NONLINEARITY_CONTROL_NEXT9.json')
        results.extend(dict(control_family='ANALYTIC_FIXED_U_SHAPE',**r) for r in analytic['records'])
        csv_write('MECHANICAL_SHOCK_MEMORY_NEXT9.csv',memory);csv_write('COUNTEREXAMPLES_NEXT9.csv',results)
        assert cases==168 and raw_equal_checks==24 and not violations
        atomic(BASE/'COUNTEREXAMPLE_ANALYSIS_NEXT9.json',dict(status='COMPLETE',timestamp=utc_now(),paired_synthetic_cases=cases,
            unique_background_seed_H_noise_paths=24,seven_paired_interventions_per_path=True,raw_single_shock_future_equal_to_neutral_checks=raw_equal_checks,
            forecast_model_scope='Original RP outer0 NORMAL fixed before any outer quality, no synthetic refit',all_three_horizons_retained=True,new_fits=0,
            synthetic_cases_not_independent_real_confirmation=True,all153_memory_diagnostics=True,analytic_U_shape=analytic,
            no_candidate_threshold_or_graph_changes=True,no_real_labels_or_tau=True,NPZ_members=sorted(set(accesses))))


if __name__=='__main__':main()
