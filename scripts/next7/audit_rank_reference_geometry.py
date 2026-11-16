"""Audit deterministic rank moments and quantized tail normalizers, using H only."""
from pathlib import Path
import csv
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.worker_environment import environment
os.environ.update(environment())
import numpy as np
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note
from src.next7.conditional_null import source_components


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def reference_moment(points):
    values, multiplicities = np.unique(points, return_counts=True)
    n = len(points)
    tie_mass = sum(int(m) ** 3 - int(m) for m in multiplicities)
    no_ties = (n - 1) / (3 * (n + 1))
    loss = tie_mass / (3 * n * (n + 1) ** 2)
    left = np.r_[0, np.cumsum(multiplicities)[:-1]]
    right = np.cumsum(multiplicities)
    u = (left + right + 1.) / (n + 1.) - 1.
    direct = float(np.sum(multiplicities * u ** 2) / n)
    assert abs(direct - (no_ties - loss)) < 5e-16
    return no_ties - loss, int(np.sum(multiplicities > 1)), tie_mass, direct


def main():
    lock = {'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': digest(__file__),
        'origin': 'Posthoc interpretation audit of the already reported near-constant rank-second-moment covariate and discrete rank-tail IQR.',
        'data': 'Only original partition33 H5000 paths and the three locked suffix lengths. No online innovations, labels, new draws, fits or detector changes.',
        'rank_moment_identity': 'E_H[u^2]=(n-1)/(3(n+1))-sum_m(m^3-m)/(3n(n+1)^2), where m ranges over multiplicities of distinct canonical float32 values.',
        'tail_geometry': 'A16-point mean of binary tail indicators is an integer/16. Linear empirical quartiles are on the integer/64 lattice; their IQR is on the same lattice. The median is on the integer/32 lattice. Stored spread is max(IQR/1.349,.02).',
        'limits': 'Finite-sample arithmetic identities. No causal claim about false memory and no threshold/floor selection. Existing correlation tables remain unchanged.'}
    save('RANK_REFERENCE_GEOMETRY_AUDIT_LOCK.json', lock)
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    records, summary, frequencies, inputs = [], [], [], {}
    with activity('H-only rank moment and tail-IQR quantization audit', 'H_REFERENCE_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        for case in ('N0','N1','N2','N3'):
            historical = []
            for seed in range(512):
                h, future, _, _ = source_components(case, seed, suite, horizon=0)
                assert len(h) == 5000 and len(future) == 0
                historical.append(h)
            for length in (1000,2048,5000):
                directory = BASE / f'normalizer_transport/H_parameters/{case}/H_{length}'
                done = read_json(directory / 'DONE.json')
                p = directory / 'PARAMETERS.csv'
                assert digest(p) == done['files_sha256'][p.name]
                inputs[p.relative_to(BASE).as_posix()] = digest(p)
                parameter_rows = [r for r in rows(p) if r['channel'] == 'rank_tail']
                assert len(parameter_rows) == 512
                cohort = []
                for seed, r in enumerate(parameter_rows):
                    assert int(r['reference_index']) == seed
                    h = historical[seed][-length:]
                    prefix = int(.6 * length)
                    prefix_fit, full_fit = int(.6 * prefix), int(.6 * length)
                    ep, tp, mp, directp = reference_moment(h[:prefix_fit])
                    ef, tf, mf, directf = reference_moment(h[:full_fit])
                    stored = float(r['prefix_minus_full_rank_second_moment'])
                    assert abs(stored - (ep - ef)) < 5e-16
                    entry = {'case': case, 'H_length': length, 'reference_index': seed,
                        'prefix_fit': prefix_fit, 'full_fit': full_fit,
                        'prefix_tied_value_groups': tp, 'full_tied_value_groups': tf,
                        'prefix_tie_cubic_mass': mp, 'full_tie_cubic_mass': mf,
                        'stored_moment_difference': stored,
                        'analytic_moment_difference': ep - ef,
                        'formula_absolute_error': abs(stored - (ep - ef)),
                        'no_tie_difference': (prefix_fit-1)/(3*(prefix_fit+1)) - (full_fit-1)/(3*(full_fit+1)),
                        'online_points_or_labels_used': 0}
                    for kind in ('prefix','full'):
                        center, spread = float(r[kind+'_center']), float(r[kind+'_spread'])
                        assert abs(center*32 - round(center*32)) < 1e-12
                        floor = spread == .02
                        if not floor:
                            assert abs(spread*1.349*64 - round(spread*1.349*64)) < 1e-11
                        entry[kind+'_tail_center'] = center
                        entry[kind+'_tail_spread'] = spread
                        entry[kind+'_tail_spread_floored'] = floor
                        entry[kind+'_tail_IQR_lattice_integer_when_unfloored'] = '' if floor else round(spread*1.349*64)
                    cohort.append(entry)
                records.extend(cohort)
                summary.append({'case': case, 'H_length': length, 'histories':512,
                    'prefix_histories_with_ties':sum(r['prefix_tied_value_groups']>0 for r in cohort),
                    'full_histories_with_ties':sum(r['full_tied_value_groups']>0 for r in cohort),
                    'stored_moment_difference_range':float(np.ptp([r['stored_moment_difference'] for r in cohort])),
                    'max_formula_absolute_error':max(r['formula_absolute_error'] for r in cohort),
                    'rank_moment_interpretation':'At fixed fit sizes, only float32 tie multiplicities and floating-point reduction affect this covariate; it is not an independently estimated distributional moment.',
                    'new_inference_rule':False})
                for kind in ('prefix','full'):
                    spreads = np.array([r[kind+'_tail_spread'] for r in cohort])
                    for value, count in zip(*np.unique(spreads,return_counts=True)):
                        frequencies.append({'case':case,'H_length':length,'reference':kind,
                            'spread':float(value),'histories':int(count),'fraction':float(count/512),
                            'floor_active':bool(value==.02),
                            'IQR_lattice_integer_when_unfloored':'' if value==.02 else round(value*1.349*64)})
        outputs = {'RANK_REFERENCE_GEOMETRY_PER_H_NEXT7.csv':records,
            'RANK_REFERENCE_GEOMETRY_SUMMARY_NEXT7.csv':summary,
            'RANK_TAIL_NORMALIZER_FREQUENCIES_NEXT7.csv':frequencies}
        for name, values in outputs.items():
            csv_write(name, values)
        save('RANK_REFERENCE_GEOMETRY_AUDIT_DONE.json', {'status':'COMPLETE','created_utc':utc_now(),
            'lock_sha256':digest(BASE/'RANK_REFERENCE_GEOMETRY_AUDIT_LOCK.json'),
            'histories':len(records),'rank_moment_formula_parity':True,'tail_quantization_parity':True,
            'maximum_moment_error':max(r['formula_absolute_error'] for r in records),
            'inputs_sha256':inputs,'outputs_sha256':{name:digest(BASE/name) for name in outputs},
            'new_draws':0,'new_candidates':0,'prior_correlations_preserved':True})
    note('Rank reference geometry audited', 'All6144 histories: exact rank moments and discrete tail-IQR support, no inference changes', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()
