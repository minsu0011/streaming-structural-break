import numpy as np
import pytest
from scripts.next7.synthetic_uncertainty import estimate, group_statistics


def test_ratio_uses_only_observable_paths_and_reports_empty_draw():
    point, draws, support = estimate([1., None, 0.], np.array([[2., 0., 1.], [0., 3., 0.]]))
    assert point == .5 and support == 2
    assert draws[0] == pytest.approx(2. / 3.)
    assert np.isnan(draws[1])


def test_paired_recovery_uses_same_observable_seed_intersection():
    def row(seed):
        return {'seed': seed, 'observed_break': False, 'horizon': 1024, 'tau_offline_only': -1,
            'any_latch': False, 'control_any_latch': False, 'excess_latch_vs_paired_control': False}
    reference = [row(i) for i in range(512)]
    candidate = [row(i) for i in range(512)]
    reference[0]['event_retained_after_recovery'] = True
    reference[1]['event_retained_after_recovery'] = False
    candidate[1]['memory_retained_after_current_recovery'] = True
    candidate[2]['memory_retained_after_current_recovery'] = False
    individual, paired, _ = group_statistics({'NEXT6_HARD4_EVENT': reference, 'G1_RAW_EVENT': candidate},
        {}, np.ones((5, 512)))
    item = [r for r in paired if r['metric'] == 'memory_retained_after_recovery'][0]
    assert item['eligible_seeds'] == 1
    assert item['estimate'] == item['ci_2_5'] == item['ci_97_5'] == 1.
    means = [r['estimate'] for r in individual if r['metric'] == 'memory_retained_after_recovery']
    assert means == [.5, .5]  # Difference of differently eligible means would be wrong.
