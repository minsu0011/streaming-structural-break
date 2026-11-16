import itertools
import numpy as np
from src.next7.bootstrap_support import exact_support
from src.next7.null_growth import joint_ratio


def test_stitching_can_create_a_confirmation_absent_from_every_H_window():
    vectors = np.zeros((6, 9))
    vectors[:, 0] = [3., 3., -1., -1., 3., 3.]
    internal, stitched, upper = exact_support(vectors, 2., 2.)
    assert internal == 0. and stitched == 1.5 and upper == 1.5


def test_exact_envelope_matches_all_two_chunk_sequences():
    v = ((np.arange(63).reshape(7, 9) * 3) % 11 - 5).astype(float)
    internal, stitched, upper = exact_support(v, 2., 2.)
    observed = []
    for a, b in itertools.product(range(len(v) - 3), repeat=2):
        path = np.vstack((v[a:a + 4], v[b:b + 4]))
        observed += [joint_ratio(path[j:j + 4], 2., 2.) for j in range(5)]
    assert max(1., max(observed)) == upper


def test_three_chunk_paths_cannot_exceed_two_chunk_support_envelope():
    v = np.zeros((6, 9))
    v[:, 0] = [3., 3., -1., -1., 3., 3.]
    v[:, 8] = [-2., 1., -1., 3., -4., 2.]
    upper = exact_support(v, 2., 2.)[-1]
    for starts in itertools.product(range(3), repeat=3):
        path = np.vstack([v[a:a + 4] for a in starts])
        assert all(joint_ratio(path[j:j + 4], 2., 2.) <= upper for j in range(9))


def test_floor_one_respects_the_actual_admission_rule():
    assert exact_support(np.zeros((8, 9)), 2., 2.) == (0., 0., 1.)


def test_float_support_uses_the_actual_bootstrap_physical_ring_order():
    v = np.zeros((7, 9))
    v[:, 8] = [4., 2., .3, 1e-15, -4., 1.7, 3.9]
    maximum = 1.
    for a, b in itertools.product(range(len(v) - 3), repeat=2):
        ring = np.zeros((4, 9))
        path = np.vstack((v[a:a + 4], v[b:b + 4]))
        for t, row in enumerate(path):
            ring[t % 4] = row
            if t >= 3:
                maximum = max(maximum, joint_ratio(ring, 2., 1.))
    assert exact_support(v, 2., 1.)[-1] == maximum
