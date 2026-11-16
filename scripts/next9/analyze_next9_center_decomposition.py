"""Diagnostic: separate H-median centering from actual own-history variation.

The TRAIN constant is each already fitted ridge intercept, not a new fitted
candidate. This report cannot alter target-free eligibility or any model menu.
"""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now,csv_write
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(part):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'CENTER_DECOMPOSITION_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items(): assert digest(ROOT/name)==sha,name
    with target_free_npz() as accesses:
        import numpy as np
        from src.next9.data import FeatureBank
        from src.next9.movement_screen import grouped_means,paired_group_summary
        bank=FeatureBank()
        with np.load(BASE/'state_cache/INDEX.npz') as z: rows=z['global_row']
        rows=bank.forecast_rows(rows,16)
        groups=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[rows]],dtype=np.int32)
        times=bank.times[rows];output=[];maximum_error=0.
        for j in list(range(153))[part::4]:
            with activity(f'F{j:03d} Hmedian versus frozen TRAINmean versus ownhistory decomposition','CENTER_FORENSICS','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                moments={}
                for mode,keep in [('NORMAL',np.ones(len(rows),dtype=bool)),('EARLY_LATE',times>=400),('EARLY_SELECT',times+16<=199)]:
                    target=np.column_stack([bank.values[rows[keep]+h,j] for h in [1,4,16]])
                    gids,means=grouped_means(np.column_stack([target,target.astype(float)**2,target**2]),groups[keep])
                    moments[mode]=(gids,means)
                for fold in range(5):
                    for inner in [None,0,1,2]:
                        for split in range(3):
                            for mode in ['NORMAL','EARLY_LATE','EARLY_SELECT']:
                                name=f'outer_{fold}_'+('main' if inner is None else f'inner_{inner}')+f'_selection_{split}_'+('NORMAL' if mode=='NORMAL' else 'EARLY')
                                model_path=BASE/'movement_screen'/f'F{j:03d}'/name/'MODEL.json'
                                intercept=np.asarray(read_json(model_path)['intercept'])[[0,5,10]]
                                loss_path=BASE/('early_selection' if mode=='EARLY_SELECT' else 'movement_screen')/f'F{j:03d}'/name/'GROUP_LOSSES.npz'
                                with np.load(loss_path) as z: ids=z['group_id'];loss=z['loss']
                                gids,m=moments[mode];index=np.searchsorted(gids,ids)
                                np.testing.assert_array_equal(gids[index],ids);m=m[index]
                                constant=m[:,3:6]-2*m[:,:3]*intercept+intercept**2
                                error=float(np.max(np.abs(m[:,6:9]-loss[:,:,1])))
                                maximum_error=max(maximum_error,error)
                                assert error<1e-5,'Forecast row or arithmetic lineage mismatch'
                                for k,h in enumerate([1,4,16]):
                                    a=paired_group_summary(loss[:,k,1],constant[:,k],ids)
                                    b=paired_group_summary(constant[:,k],loss[:,k,2],ids)
                                    output.append(dict(feature=j,fold=fold,inner=inner,selection_split=split,mode=mode,horizon=h,
                                        frozen_TRAIN_target_mean=float(intercept[k]),Hmedian_to_TRAINconstant_gain=a['mean_gain'],
                                        TRAINconstant_to_D1_gain=b['mean_gain'],TRAINconstant_to_D1_ci_low=b['ci_low'],TRAINconstant_to_D1_ci_high=b['ci_high'],
                                        group_count=len(ids),same_Hmedian_loss_max_error=error,screening_eligibility_changed=False,
                                        interpretation='Diagnostic centering versus own-history variation, not a new selected baseline or underlying-mechanism proof'))
        csv_write(f'center_decomposition/PART_{part}.csv',output)
        assert not violations
        atomic(BASE/'center_decomposition'/f'PART_{part}_DONE.json',dict(status='COMPLETE',timestamp=utc_now(),rows=len(output),
            source_sha256=digest(ROOT/'scripts/next9/analyze_next9_center_decomposition.py'),
            csv_sha256=digest(BASE/'center_decomposition'/f'PART_{part}.csv'),maximum_Hmedian_loss_parity_error=maximum_error,
            new_fits=0,no_break_labels_tau=True,eligibility_and_model_spec_unchanged=True,NPZ_members=sorted(set(accesses))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,required=True,choices=range(4));main(p.parse_args().part)
