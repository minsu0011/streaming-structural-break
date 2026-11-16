"""One frozen100-replicate confidence-interval calibration cell."""
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity, note
from src.next8.rank_ci_calibration import cells, evaluate_sample


def main(cell):
    lock = read_json(BASE / 'RANK_CI_CALIBRATION_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    policy = read_json(BASE / 'RANK_CI_CALIBRATION_POLICY_NEXT8.json')
    assert policy['replicates'] == 100 and policy['bootstrap_draws'] == 2000
    out = BASE / 'rank_ci_calibration' / f'cell_{cell:02d}'
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Partial calibration retained; no silent repetitions')
    groups, rho, shift = cells()[cell]
    with activity('Rank interval sampling calibration cell' + str(cell), 'STATISTICAL_VALIDATION', 'B1_CONDITIONAL_DIRECTION'):
        records = []
        for replicate in range(100):
            with activity(f'Known-truth CI cell{cell} replicate{replicate}', 'STATISTICAL_VALIDATION', 'B1_CONDITIONAL_DIRECTION'):
                record = evaluate_sample(groups, rho, shift, replicate)
                atomic(out / f'REP_{replicate:03d}.json', record)
                records.append(record)
        atomic(out / 'DONE.json', {'status': 'COMPLETE', 'timestamp': utc_now(), 'cell': cell,
            'groups': groups, 'rho': rho, 'mean_shift': shift, 'replicates': len(records),
            'files_sha256': {f'REP_{k:03d}.json': digest(out / f'REP_{k:03d}.json') for k in range(100)},
            'policy_sha256': digest(BASE / 'RANK_CI_CALIBRATION_POLICY_NEXT8.json'),
            'no_RP_scores_or_intervals_changed': True})
    note('Confidence interval sampling control complete', 'cell' + str(cell), 'B1_CONDITIONAL_DIRECTION')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', type=int, choices=range(12), required=True)
    main(parser.parse_args().cell)
