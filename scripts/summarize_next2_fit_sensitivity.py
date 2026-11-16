"""Compare existing CV fits without averaging their predictions or treating fits as independent."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next2.counterexamples import policy
from src.next2.synthetic_analysis import calibration_map, background, threshold, alarm_summary, score
from src.next2.blend import combine
from src.next.alarm_calibration import wilson_interval
from src.next2.session import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now

OUT = ROOT / 'artifacts/next2/finalist_fit_sensitivity'
J = 'J102_SERIAL_VARIANCE_512'
C = 'C205_DEPENDENCE'
B = 'BL2_C205_DEPENDENCE_80'


def main():
    menu_path = ROOT / 'configs/next2/FINALIST_FIT_SENSITIVITY_LOCK.json'
    menu = json.loads(menu_path.read_text())
    lock = policy()
    assert menu['counterexample_lock_sha256'] == sha256(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    mapping = calibration_map(lock)
    completed, pending, receipts = {}, [], {}
    for ref in menu['models']:
        eid, fold = ref['experiment_id'], ref['fold']
        folder = ROOT / ref['reuse_directory'] if 'reuse_directory' in ref else OUT / f'fold_{fold}' / eid
        receipt = folder / 'RESULTS.json'
        if not receipt.exists():
            pending.append((eid, fold))
            continue
        r = json.loads(receipt.read_text())
        assert r['status'] == 'PASS' and r['identity']['model_sha256'] == ref['model_sha256']
        assert sha256(ROOT / ref['model_path']) == ref['model_sha256']
        if 'reuse_directory' in ref:
            assert sha256(receipt) == ref['reuse_receipt_sha256']
        receipts[receipt.relative_to(ROOT).as_posix()] = sha256(receipt)
        completed[eid, fold] = folder

    verified = {}
    def load(eid, fold, kind, scenario):
        if eid == B:
            left, tau = load(J, fold, kind, scenario)
            right, other_tau = load(C, fold, kind, scenario)
            np.testing.assert_array_equal(tau, other_tau)
            assert left.shape == right.shape
            return combine(left, right, .8), tau
        folder = completed[eid, fold] / kind / scenario
        if folder not in verified:
            receipt = folder / 'RESULTS.json'
            r = json.loads(receipt.read_text())
            assert r['status'] == 'PASS' and r['identity']['scenario'] == scenario
            for name, digest in r['files_sha256'].items():
                assert sha256(folder / name) == digest
            verified[folder] = True
            receipts[receipt.relative_to(ROOT).as_posix()] = sha256(receipt)
        return (np.load(folder / 'prediction.npy', mmap_mode='r', allow_pickle=False),
                np.load(folder / 'synthetic_tau.npy', allow_pickle=False))

    jobs = sorted(completed)
    jobs += [(B, fold) for fold in range(5) if (J, fold) in completed and (C, fold) in completed]
    counter_rows, long_rows = [], []
    for eid, fold in jobs:
        calibration = {}
        for scenario in mapping.values():
            p, tau = load(eid, fold, 'calibration', scenario)
            assert np.all(tau == -1) and p.shape == (lock['calibration_null_series_per_regime'], lock['online_length'])
            calibration[scenario] = p.max(axis=1)
        for spec in lock['scenarios']:
            scenario = spec['id']
            p, tau = load(eid, fold, 'counterexample', scenario)
            auc = score(p, tau)
            for alpha in lock['long_null']['threshold_alphas']:
                level = threshold(calibration[mapping[background(spec)]], alpha)
                summary, crossed, _ = alarm_summary(p, tau, level)
                row = {'candidate': eid, 'fitted_fold': fold, 'scenario': scenario, 'alpha': alpha,
                       'ts_auc': auc, 'threshold': level, **summary,
                       'artifact_kind': 'existing full-CV fitted model; synthetic diagnostic only'}
                if scenario == 'O1':
                    n = len(p) // 2
                    row.update(control_alarm_rate=float(crossed[:n].mean()), outlier_alarm_rate=float(crossed[n:].mean()),
                               outlier_alarm_excess=float(crossed[n:].mean() - crossed[:n].mean()))
                counter_rows.append(row)
        for scenario in lock['long_null']['regimes']:
            p, tau = load(eid, fold, 'long_null', scenario)
            assert np.all(tau == -1)
            baseline = p[:, :1024].max(axis=1).mean()
            for horizon in lock['long_null']['horizons']:
                maximum = p[:, :horizon].max(axis=1)
                for alpha in lock['long_null']['threshold_alphas']:
                    level = threshold(calibration[scenario], alpha)
                    count = int((maximum > level).sum()) if level is not None else 0
                    lo, hi = wilson_interval(count, len(p))
                    long_rows.append({'candidate': eid, 'fitted_fold': fold, 'regime': scenario, 'horizon': horizon,
                        'alpha': alpha, 'threshold': level, 'eventual_alarm_rate': count / len(p),
                        'eventual_alarm_count': count, 'series': len(p), 'alarm_wilson95_low': lo, 'alarm_wilson95_high': hi,
                        'mean_max_score': float(maximum.mean()), 'mean_max_growth_from_1024': float(maximum.mean()-baseline),
                        'artifact_kind': 'existing full-CV fitted model; synthetic diagnostic only'})
        print('NEXT2 FIT SUMMARY', eid, fold, flush=True)
    counter = pd.DataFrame(counter_rows)
    long = pd.DataFrame(long_rows)
    counter.to_csv(OUT / 'COUNTEREXAMPLES_BY_FIT.csv', index=False)
    long.to_csv(OUT / 'LONG_NULL_BY_FIT.csv', index=False)
    contrasts = []
    fields = ('ts_auc', 'postbreak_first_detection_rate', 'null_eventual_alarm_rate', 'outlier_alarm_excess')
    for (scenario, alpha), group in counter.groupby(['scenario', 'alpha']):
        reference = group[group.candidate == J].set_index('fitted_fold')
        for eid, subset in group.groupby('candidate'):
            if eid == J:
                continue
            matched = subset.set_index('fitted_fold').join(reference, lsuffix='_new', rsuffix='_ref', how='inner')
            for field in fields:
                if field+'_new' not in matched:
                    continue
                delta = (matched[field+'_new'] - matched[field+'_ref']).dropna()
                if len(delta):
                    contrasts.append({'candidate': eid, 'reference': J, 'scenario': scenario, 'alpha': alpha,
                        'metric': field, 'matched_training_fits': len(delta), 'mean_delta_across_fits': float(delta.mean()),
                        'min_delta_across_fits': float(delta.min()), 'max_delta_across_fits': float(delta.max()),
                        'positive_deltas': int((delta > 0).sum()), 'negative_deltas': int((delta < 0).sum()),
                        'interpretation': 'Descriptive sensitivity across overlapping training fits, not independent replicates or a new ensemble.'})
    pd.DataFrame(contrasts).to_csv(OUT / 'MATCHED_FIT_CONTRASTS.csv', index=False)
    result = {'status': 'PASS' if not pending else 'PARTIAL_RESEARCH_SNAPSHOT', 'created_utc': utc_now(),
        'menu_sha256': sha256(menu_path), 'script_sha256': sha256(Path(__file__)), 'new_model_fits': 0,
        'completed_original_fits': len(completed), 'derived_fixed_80_20_blends': len(jobs)-len(completed),
        'pending': pending, 'source_receipts_sha256': receipts,
        'files_sha256': {name: sha256(OUT/name) for name in ('COUNTEREXAMPLES_BY_FIT.csv','LONG_NULL_BY_FIT.csv','MATCHED_FIT_CONTRASTS.csv')},
        'limitations': 'Same frozen synthetic cohorts across fits. Overlapping CV training sets; no independent-fit CI, no synthetic tuning, no new prediction ensemble.'}
    write_json(OUT / 'SUMMARY_RESULTS.json', result)
    checkpoint('Summarized fixed training-fit synthetic sensitivity', ['Finish remaining prediction jobs', 'Inspect robustness directions'],
               artifacts=['artifacts/next2/finalist_fit_sensitivity/SUMMARY_RESULTS.json'])
    print(result['status'], len(completed), 'original fits', len(jobs)-len(completed), 'blends', flush=True)


if __name__ == '__main__':
    main()
