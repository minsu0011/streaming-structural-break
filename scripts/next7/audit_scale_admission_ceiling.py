"""Measure where a shared G2 cutoff exceeds the bounded scale branch ceiling."""
from pathlib import Path
import os
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.worker_environment import environment
os.environ.update(environment())
import numpy as np
from src.next7.common import BASE,read_json,save,csv_write,digest,utc_now,activity,note
from src.next7.calibration import load_calibration,cutoff
from src.next7.null_growth import GRID


def main():
    test=BASE/'audit/SCALE_CEILING_TESTS_001.xml'
    xml=ET.parse(test)
    assert len(xml.findall('.//testcase'))==4 and not xml.findall('.//failure') and not xml.findall('.//error')
    save('SCALE_ADMISSION_CEILING_LOCK_NEXT7.json',{'status':'FROZEN','created_utc':utc_now(),
        'source_sha256':{name:digest(ROOT/name) for name in ('scripts/next7/audit_scale_admission_ceiling.py',
            'src/next7/scale.py','src/next7/event.py','src/next7/null_growth.py')},'tests_sha256':digest(test),
        'origin':'Posthoc algebraic hypothesis motivated by bounded S1 strength and the shared core/scale admission cut; no candidate or threshold change.',
        'theorem':'S1 signed block evidence is in[-4,4]; each registered four-block confirmation strength is at most4. Its raw ratio is at most4/scale_cut. With strict raw_ratio>admission_cut, an admission cutoff at least that ceiling precludes scale-only qualification at that routed horizon. Any qualification there must be won by core. Earlier qualified memory can still exist.',
        'scope':'Original6144 H calibrations, four null backgrounds, H1000/2048/5000 and allnine original grid cuts. No online paths, tau, labels, draws, models or inference outputs are generated.',
        'uncertainty':'Reuse original2000 multiplicity rows over512 H seed units for marginal descriptive fractions; no new draws or simultaneous validity claim.',
        'no_inference_rule_or_role_changed':True})
    calibration=load_calibration()
    count_path=BASE/'synthetic_stats/COUNTS_2000x512.npy'
    assert digest(count_path)==read_json(BASE/'synthetic_stats/COUNTS_LOCK.json')['sha256']
    counts=np.load(count_path,allow_pickle=False)
    details,summary,inputs=[],[],{}
    with activity('Bounded scale evidence versus shared H admission cuts','H_CALIBRATION_FORENSIC','F4_NULL_GROWTH_CORRECTION'):
        for case in ('N0','N1','N2','N3'):
            parent=BASE/f'conditional_null/diagonal/{case}'
            p=parent/'H_CALIBRATIONS.json'
            assert digest(p)==read_json(parent/'DONE.json')['H_CALIBRATIONS_sha256']
            inputs[p.relative_to(BASE).as_posix()]=digest(p)
            original={(r['H_length'],r['reference_index']):r['calibration'] for r in read_json(p)}
            for length in (1000,2048,5000):
                scale_cut=cutoff(calibration,length)
                cap=4/scale_cut
                cuts=np.array([original[length,seed]['admission_cuts'] for seed in range(512)])
                assert cuts.shape==(512,9) and np.all(np.diff(cuts,axis=1)>=0)
                disabled=cuts>=cap
                for j,grid in enumerate(GRID):
                    first_fresh_point=63 if j==0 else 16*(int(GRID[j-1])+1)-1
                    values=disabled[:,j].astype(np.float64)
                    draws=counts@values/512
                    low,high=np.quantile(draws,[.025,.975])
                    ratio=cuts[:,j]/cap
                    summary.append({'case':case,'H_length':length,'grid_blocks':int(grid),
                        'grid_horizon_points':int(grid*16),'routed_cut_first_possible_fresh_point':first_fresh_point,
                        'scale_confirmation_cut':scale_cut,'scale_raw_ratio_ceiling':cap,
                        'histories_scale_only_qualification_impossible_at_this_cut':int(values.sum()),
                        'fraction_scale_only_qualification_impossible':float(values.mean()),
                        'fraction_CI_low_pointwise':float(low),'fraction_CI_high_pointwise':float(high),
                        'cut_divided_by_scale_ceiling_q10':float(np.quantile(ratio,.1)),
                        'cut_divided_by_scale_ceiling_median':float(np.median(ratio)),
                        'cut_divided_by_scale_ceiling_q90':float(np.quantile(ratio,.9)),
                        'histories':512,'bootstrap_draws':2000,'earlier_memory_can_exist':True,
                        'not_a_scale_detection_rate':True,'new_threshold_selected':False})
                for seed in range(512):
                    blocked=np.flatnonzero(disabled[seed])
                    first=int(blocked[0]) if len(blocked) else None
                    details.append({'case':case,'H_length':length,'reference_index':seed,
                        'scale_confirmation_cut':scale_cut,'scale_raw_ratio_ceiling':cap,
                        'core_raw_ratio_global_ceiling':12/calibration['core_cut'],
                        'first_blocked_grid_blocks':int(GRID[first]) if first is not None else None,
                        'first_blocked_fresh_confirmation_point':(63 if first==0 else 16*(int(GRID[first-1])+1)-1) if first is not None else None,
                        'all_supported_scale_only_confirmations_blocked':bool(disabled[seed,0]),
                        **{f'cut_at_{int(grid)}_blocks':float(cuts[seed,j]) for j,grid in enumerate(GRID)},
                        **{f'scale_only_blocked_at_{int(grid)}_blocks':bool(disabled[seed,j]) for j,grid in enumerate(GRID)}})
        csv_write('SCALE_ADMISSION_CEILING_RESULTS_NEXT7.csv',summary)
        csv_write('SCALE_ADMISSION_CEILING_PER_H_NEXT7.csv',details)
        save('SCALE_ADMISSION_CEILING_DONE_NEXT7.json',{'status':'COMPLETE','created_utc':utc_now(),
            'lock_sha256':digest(BASE/'SCALE_ADMISSION_CEILING_LOCK_NEXT7.json'),'histories':6144,
            'H_grid_cut_comparisons':len(details)*9,'summary_rows':len(summary),'original_count_sha256':digest(count_path),
            'inputs_sha256':inputs,'outputs_sha256':{name:digest(BASE/name) for name in
                ('SCALE_ADMISSION_CEILING_RESULTS_NEXT7.csv','SCALE_ADMISSION_CEILING_PER_H_NEXT7.csv')},
            'new_candidates':0,'new_draws':0,'future_observations_used':0})
    note('Scale admission ceiling audit complete','Original6144 H calibrations, allnine grids; bounded scale-only feasibility without inference changes','F4_NULL_GROWTH_CORRECTION')


if __name__=='__main__':main()
