"""DEV H-only held-out whitening/normalization diagnostics, no online targets."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import time
import json
import numpy as np
import pandas as pd
from numba import njit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.whitening import fit_historical, innovation_update, AR_ORDERS
from src.next.normalization import fit_normalization, replay_normalization, MODES
from src.next.regimes import acf
from src.next.progress import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


@njit(cache=True)
def residual_replay(points, reference, coefficients, ring, counter):
    residual = np.empty(len(points),dtype=np.float64)
    for j in range(len(points)):
        residual[j] = innovation_update(points[j],reference,coefficients,ring,counter)
    return residual


def shape_statistics(values):
    z = np.asarray(values,dtype=float)
    centered = z-z.mean()
    variance = max(float(np.mean(centered*centered)),1e-12)
    standard = centered/np.sqrt(variance)
    squared = z*z
    result = {'mean':float(z.mean()),'sd':float(z.std()),'kurtosis':float(np.mean(standard**4)),
              'tail_abs_over_3':float(np.mean(np.abs(z)>3))}
    for lag in [1,2,4,8]:
        result[f'acf_{lag}'] = acf(z,lag)
    result['squared_acf_1'] = acf(squared,1)
    result['squared_acf_2'] = acf(squared,2)
    result['absolute_acf_1'] = acf(np.abs(z),1)
    return result


def main():
    started = time.perf_counter()
    data = prepare_dev(ROOT)
    out = ROOT/'artifacts/next/historical_holdout'
    out.mkdir(exist_ok=True)
    completed_path = out/'RESULTS.json'
    if completed_path.exists():
        completed = json.loads(completed_path.read_text(encoding='utf-8'))
        if completed['status'] != 'PASS' or completed['policy']['split_sha256'] != data.guard.split.digest or completed['policy']['dev_manifest_sha256'] != sha256(data.directory/'MANIFEST.json'):
            raise RuntimeError('Completed H-only study identity changed')
        for name,digest in completed['input_files_sha256'].items():
            if sha256(out/name) != digest:
                raise RuntimeError('Completed H-only study data changed')
        for name,digest in completed['scientific_source_sha256'].items():
            if sha256(ROOT/name) != digest:
                raise RuntimeError('Historical scientific implementation changed')
        print('HISTORICAL HOLDOUT already complete; verified and retained',flush=True)
        return
    policy = {'purpose':'H-only forward holdout diagnostic, not competition CV or candidate selection authority',
        'historical_training_prefix_fraction':.75,'orders':list(AR_ORDERS),'normalization_order':8,'normalization_modes':list(MODES),
        'holdout':'last quarter of historical segment; no actual online observation, tau or target is accessed',
        'series_ids':'all frozen DEV','script_sha256':sha256(Path(__file__)),
        'split_sha256':data.guard.split.digest,'dev_manifest_sha256':sha256(data.directory/'MANIFEST.json')}
    policy_path = out/'POLICY.json'
    if policy_path.exists() and json.loads(policy_path.read_text(encoding='utf-8')) != policy:
        raise RuntimeError('Historical diagnostic policy changed')
    write_json(policy_path,policy)
    if (out/'RESULTS.json').exists():
        print('HISTORICAL HOLDOUT already complete',flush=True)
        return
    regime = pd.read_parquet(ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet',
                             columns=['dataset_id','historical_regime']).set_index('dataset_id').historical_regime.to_dict()
    whitening, normalization = [], []
    offsets = data.series['historical_offsets']
    for j,sid in enumerate(data.ids):
        h = data.historical[offsets[j]:offsets[j+1]]
        cut = int(len(h)*.75)
        for order in AR_ORDERS:
            tick = time.perf_counter()
            f = fit_historical(h[:cut],order)
            initialization = time.perf_counter()-tick
            residual = residual_replay(h[cut:],f['reference'],f['coefficients'],f['ring'],f['counter'])
            train_r = f['historical_innovations']
            sd = max(float(train_r.std()),1e-8)
            standardized = (residual-train_r.mean())/sd
            whitening.append({'dataset_id':int(sid),'regime':int(regime[int(sid)]),'order':order,
                'historical_prefix_count':cut,'historical_holdout_count':len(h)-cut,
                'historical_fit_seconds':initialization,'pole_projected':f['pole_projected'],
                'raw_fitted_pole_radius':f['original_pole_radius'],
                'relative_holdout_mse':float(np.mean(standardized**2)),**shape_statistics(standardized)})
            if order == 8:
                for mode in MODES:
                    norm = fit_normalization(train_r,mode)
                    z,_ = replay_normalization(residual,norm['parameters'],norm['conditional'],norm['sorted_h'],norm['mode'])
                    normalization.append({'dataset_id':int(sid),'regime':int(regime[int(sid)]),'normalization':mode,
                        'arch_slope1':float(norm['parameters'][3]),'arch_slope2':float(norm['parameters'][4]),
                        **shape_statistics(z)})
        if (j+1) % 1000 == 0:
            print('HISTORICAL FORWARD HOLDOUT',j+1,round(time.perf_counter()-started,1),'seconds',flush=True)
    wf,nf = pd.DataFrame(whitening),pd.DataFrame(normalization)
    wf.to_parquet(out/'WHITENING_PER_DEV_SERIES.parquet',index=False)
    nf.to_parquet(out/'NORMALIZATION_PER_DEV_SERIES.parquet',index=False)
    summaries = []
    for frame,key in [(wf,'order'),(nf,'normalization')]:
        for value,group in frame.groupby(key):
            summaries.append({'kind':key,'variant':str(value),'series':len(group),
                'mean_abs_acf1':float(group.acf_1.abs().mean()),
                'mean_abs_acf8':float(group.acf_8.abs().mean()),
                'mean_abs_squared_acf1':float(group.squared_acf_1.abs().mean()),
                'mean_abs_squared_acf2':float(group.squared_acf_2.abs().mean()),
                'median_kurtosis':float(group['kurtosis'].median()),
                'mean_tail_abs_over_3':float(group.tail_abs_over_3.mean())})
    pd.DataFrame(summaries).to_csv(out/'SUMMARY.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','timestamp':utc_now(),'policy':policy,'summary':summaries,
        'input_files_sha256':{name:sha256(out/name) for name in ['WHITENING_PER_DEV_SERIES.parquet','NORMALIZATION_PER_DEV_SERIES.parquet']},
        'scientific_source_sha256':{name:sha256(ROOT/name) for name in ['src/next/whitening.py','src/next/normalization.py','src/next/regimes.py']},
        'elapsed_seconds':time.perf_counter()-started,'seal_rows':0,'actual_online_values_used':False,
        'targets_used':False,'reduced_new_usage':0,
        'limitation':'H quarters are a method diagnostic. They may not be stationary and do not substitute for held-out break labels or full primary CV.'})
    checkpoint(ROOT,'historical-whitening-diagnostics',['H-only forward holdout 5 AR orders and 8 normalizers'],
        ['initial candidate queue','historical-null calibration hypothesis','mechanism error analysis'],
        artifacts=['artifacts/next/historical_holdout/SUMMARY.csv'],best=.611292686371357)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
