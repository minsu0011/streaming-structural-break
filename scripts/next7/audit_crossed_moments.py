"""Audit the exact resampling moments after all12 frozen cohorts complete."""
from pathlib import Path
import argparse
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.worker_environment import environment
os.environ.update(environment())
import numpy as np
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, require_guard
from src.next7.crossed_moments import exact_product_moments

SOURCES = ('src/next7/crossed_moments.py', 'scripts/next7/audit_crossed_moments.py')
COHORTS = [(case, length) for case in ('N0', 'N1', 'N2', 'N3') for length in (1000, 2048, 5000)]
METRICS = {'qualified': 'G2_QUALIFIED_EVER_LATCH',
           'terminal': 'FIXED_HORIZON_MAX_EXCEEDS_TERMINAL_CUT',
           'early_only': 'EARLIER_CROSSING_RETAINED_ONLY'}


def lock():
    path = BASE / 'CROSSED_MOMENT_AUDIT_LOCK_NEXT7.json'
    sources = {name: digest(ROOT/name) for name in SOURCES}
    if path.exists():
        assert read_json(path)['sources_sha256'] == sources
        return
    test = BASE/'audit/CROSSED_MOMENT_TESTS_001.xml'
    xml = ET.parse(test)
    assert len(xml.findall('.//testcase')) == 8 and not xml.findall('.//failure') and not xml.findall('.//error')
    save(path.name, {'status': 'FROZEN', 'created_utc': utc_now(), 'sources_sha256': sources,
        'tests_sha256': digest(test), 'fixed_cohorts': COHORTS, 'metrics': METRICS,
        'purpose': 'Exact finite-multinomial mean/variance audit of the original same-count quadratic functional. Independent-axis moments are algebraic comparisons, not replacement confidence intervals.',
        'scope': 'All12 complete crossed cohorts, three existing binary partitions and three registered horizons; no partial-cohort result or new inference.',
        'unchanged_original_statistics_lock_sha256': digest(BASE/'CROSSED_STATISTICS_LOCK.json'),
        'new_draws': 0, 'new_intervals': 0, 'new_models': 0, 'new_thresholds': 0,
        'population_coverage_or_promotion_claim': False})


def main(lock_only):
    lock()
    if lock_only:
        return
    require_guard()
    # Check every dependency before opening a work activity or emitting results.
    for case, length in COHORTS:
        assert read_json(BASE/f'conditional_statistics/{case}/H_{length}/DONE.json')['status'] == 'COMPLETE'
    outputs, inputs = [], {}
    with activity('Exact finite-multinomial crossed-grid uncertainty audit', 'UNCERTAINTY_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        for case, length in COHORTS:
            directory = BASE/f'conditional_statistics/{case}/H_{length}'
            done = read_json(directory/'DONE.json')
            for name in ('GRID_FOR_FIGURES.npz', 'BOOTSTRAP_DRAWS.npz'):
                assert digest(directory/name) == done['files_sha256'][name]
                inputs[(directory/name).relative_to(BASE).as_posix()] = digest(directory/name)
            with np.load(directory/'GRID_FOR_FIGURES.npz', allow_pickle=False) as grid, np.load(directory/'BOOTSTRAP_DRAWS.npz', allow_pickle=False) as boot:
                for name, metric in METRICS.items():
                    tensor = grid[name]
                    assert tensor.shape == (512, 512, 3) and np.isin(tensor, (0, 1)).all()
                    for j, horizon in enumerate((1024, 4096, 16384)):
                        moments = exact_product_moments(tensor[:, :, j])
                        draws = boot[f'{metric}_{horizon}']
                        assert draws.shape == (2000,) and np.isfinite(draws).all()
                        sd = moments['same_axis_exact_SD']
                        empirical_sd = float(draws.std(ddof=1))
                        outputs.append({'case': case, 'H_length': length, 'horizon': horizon,
                            'metric': metric, **moments, 'original2000_draw_mean': float(draws.mean()),
                            'original2000_draw_sample_SD': empirical_sd,
                            'original_draw_mean_minus_exact': float(draws.mean()) - moments['same_axis_exact_mean'],
                            'original_draw_SD_over_exact': empirical_sd/sd if sd > 0 else None,
                            'new_interval_or_calibration': False, 'coverage_claim': False})
        assert len(outputs) == 108
        csv_write('CROSSED_RESAMPLING_EXACT_MOMENTS_NEXT7.csv', outputs)
        save('CROSSED_MOMENT_AUDIT_DONE_NEXT7.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'lock_sha256': digest(BASE/'CROSSED_MOMENT_AUDIT_LOCK_NEXT7.json'), 'inputs_sha256': inputs,
            'cohorts': 12, 'rows': len(outputs), 'partial_cohorts_included': False,
            'new_random_draws': 0, 'existing_draws_modified': False, 'original_intervals_replaced': False,
            'output_sha256': digest(BASE/'CROSSED_RESAMPLING_EXACT_MOMENTS_NEXT7.csv')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-only', action='store_true')
    main(parser.parse_args().lock_only)
