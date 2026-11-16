"""Deterministic, frozen-source diagnostic data for scientific figures3..10."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary


def main():
    violations=install_workspace_boundary()
    lock=read_json(BASE/'FIGURE_DIAGNOSTIC_DETAILS_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    scientific=read_json(BASE/'SCIENTIFIC_RESULTS_NEXT9.json');assert scientific['status']=='COMPLETE'
    with activity('Prepare deterministic heldout trajectories, matched-current bins, and frozen coefficients for figures','FIGURE_DIAGNOSTICS','B2_NO_REPLICATED_OPPOSITE_ORIENTATION'):
        import numpy as np
        import joblib
        from src.next9.data import FeatureBank,outer_scope
        from src.next9.labels import ScopedLabels
        from src.next9.transition_inputs import path_features,probabilities
        from src.next9.movement_screen import grouped_means
        from src.next9.io import save_npz
        bank=FeatureBank();scope=outer_scope(bank,0);labels=ScopedLabels(bank)
        candidates=read_json(BASE/'candidates'/scope.name/'FROZEN.json')['normal_candidates']
        sid=min(scope.validation_ids);s=int(np.flatnonzero(bank.ids==sid)[0]);lo,hi=map(int,bank.offsets[s:s+2])
        rows=np.arange(lo,min(hi,lo+257),dtype=np.int64);arrays={'time':bank.times[rows]};bins=[]
        for j in candidates:
            state=bank.pre_state(rows,j);arrays[f'F{j}_current']=bank.current(rows,[j])[:,0]
            for name in ['S1_DIRECTION','S2_DIRECTION_RUN','S3_DIRECTION_SLOPE','S4_REVERSAL','direction_run_capped15']:
                arrays[f'F{j}_{name}']=state[name]
            lawdir=BASE/'self_transitions'/scope.name/'NORMAL'/f'F{j:03d}';done=read_json(lawdir/'DONE.json')
            assert digest(lawdir/'LAW.joblib')==done['law_sha256'];law=joblib.load(lawdir/'LAW.joblib')
            arrays[f'F{j}_path']=path_features(bank,rows,j,law)
            arrays[f'F{j}_next_prob']=probabilities(law,state,bank.series['context'][bank.series_index[rows]])
            directory=BASE/'outer_orientation_refits'/scope.name/'NORMAL'/f'F{j:03d}'
            frozen=read_json(directory/'FROZEN.json');assert digest(directory/'PREDICTIONS.npz')==frozen['files_sha256']['PREDICTIONS.npz']
            domain=frozen['ranges'][0];assert domain['family']=='S1_DIRECTION'
            with np.load(directory/'PREDICTIONS.npz') as z:
                rr=z['global_row'];f=z['current_F'];state=z['S1_DIRECTION']
            y=labels.get(rr,scope,'FROZEN_OUTER_EVALUATION',prediction_receipt=frozen)
            ids=bank.row_ids[rr];edges=np.linspace(domain['low'],domain['high'],11)
            if domain['status']!='SUPPORTED':
                bins.append(dict(feature=j,status=domain['status'],common_low=domain['low'],common_high=domain['high']));continue
            for k in range(10):
                for code in [0,2]:
                    keep=(state==code)&(f>=edges[k])&((f<edges[k+1]) if k<9 else (f<=edges[k+1]))
                    groups=np.unique(ids[keep]);mean=None
                    if len(groups)>=64:mean=float(grouped_means(y[keep],ids[keep])[1].mean())
                    bins.append(dict(feature=j,state=code,bin=k,current_F_low=float(edges[k]),current_F_high=float(edges[k+1]),
                        current_F_midpoint=float((edges[k]+edges[k+1])/2),group_weighted_break_fraction=mean,groups=len(groups),rows=int(keep.sum()),
                        status='DESCRIPTIVE_SUPPORTED' if mean is not None else 'INSUFFICIENT_BIN_GROUP_SUPPORT',
                        common_range_training_only=True,not_same_time_adjusted_inference=True))
        coefficient_rows=[]
        for fold in range(5):
            directory=BASE/'joint_heads'/f'outer_{fold}'/'NORMAL';spec=read_json(directory/'SPEC_FROZEN.json')
            alias=spec['aliases']['M4'];assert digest(directory/(alias+'.joblib'))==spec['files_sha256'][alias+'.joblib']
            model=joblib.load(directory/(alias+'.joblib'))
            coef,intercept=model.linear_parameters();names=model.training['input_names']
            for j in candidates:
                rule=spec['rule_specs'].get(str(j))
                for index in [0,1]:
                    name=f'F{j}_conditional_{index}';active=name in names
                    coefficient_rows.append(dict(fold=fold,feature=j,pair=None if rule is None else rule['pair'],pair_state_index=index,
                        conditional_increment_coefficient=float(coef[names.index(name)]) if active else 0.,active_conditional_branch=active,
                        M4_actual_alias=alias,not_total_current_F_derivative=True))
        out=BASE/'figure_data';save_npz(out/'TRAJECTORIES.npz',**arrays)
        atomic(out/'COMMON_F_BINS.json',dict(status='COMPLETE',timestamp=utc_now(),fold=0,mode='NORMAL',fixed_bins=10,records=bins,
            interpretation='Equal-group break proportions inside identical TRAIN-frozen current-F bins. Descriptive only, not same-time-adjusted evidence; main B inference uses exact TS-AUC and paired controls.'))
        atomic(out/'CONDITIONAL_COEFFICIENTS.json',dict(status='COMPLETE',records=coefficient_rows,
            zero_absent_branch_not_estimated_null=True,coefficients_are_joint_M4_conditional_increments_not_total_F_effect=True))
        assert not violations
        atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),trajectory_dataset_id=sid,fold=0,candidates=candidates,
            deterministic_series_choice='Smallest original heldout fold0 dataset ID, never selected by target, score or trajectory shape',
            trajectory_points=len(rows),label_access=labels.accesses,source_scientific_results_sha256=digest(BASE/'SCIENTIFIC_RESULTS_NEXT9.json'),
            files_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},raw_RP_observations_not_copied=True))


if __name__=='__main__':main()
