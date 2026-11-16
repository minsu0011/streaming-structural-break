import numpy as np
from src.next7.viewer_evidence import evidence_flags


def path(scale_ratio, admission, *, core_ratio=0., supported=True):
    values = np.zeros((80, 45))
    for t in range(80):
        blocks = (t + 1) // 16
        complete = (t + 1) % 16 == 0
        values[t, 16] = blocks
        values[t, 15] = complete
        values[t, 43] = supported
        values[t, 42] = admission
        values[t, 9] = -4
        if blocks >= 4:
            values[t, 11] = scale_ratio
            values[t, 13] = core_ratio
            values[t, 40] = complete and supported and max(scale_ratio, core_ratio) > admission
    return values


def test_exact_equality_stays_rejected_even_when_display_rounding_hides_margin():
    exact = 4 / 2.0409484800482542
    for admission, status in ((exact, 3), (np.nextafter(exact, 0), 5)):
        values = path(exact, admission)
        flags = evidence_flags(values, 2.0409484800482542)
        assert flags[63, 0] == status
        assert flags[64, 0] == status
        assert flags[63, 2] == 1 and flags[64, 2] == 0
        assert flags[64, 1] == 63


def test_scale_rejection_does_not_erase_core_qualification_or_existing_memory():
    values = path(1.4, 1.5, core_ratio=2.1)
    values[63:, 30] = .7
    flags = evidence_flags(values, 2.)
    assert flags[64, 0] == 4
    assert flags[64, 5] == 1 and flags[64, 6] == 0
    assert flags[64, 7] == 0 and flags[64, 8] == 1
    assert flags[64, 3] == 4


def test_support_and_four_block_warmup_precede_scale_status():
    supported = evidence_flags(path(.8, 1.), 2.)
    assert supported[0, 0] == supported[62, 0] == 1
    assert supported[63, 0] == 2
    unsupported = evidence_flags(path(1.8, 1., supported=False), 2.)
    assert np.all(unsupported[:, 0] == 0)


def test_explanation_prefix_does_not_read_future_state_rows():
    values = path(1.4, 1.5, core_ratio=2.1)
    expected = evidence_flags(values, 2.)
    np.testing.assert_array_equal(evidence_flags(values[:64], 2.), expected[:64])
    changed = values.copy()
    changed[79:, 9] = 4.
    changed[79:, 11] = .2
    changed[79:, 13] = 0.
    changed[79:, 40] = 0.
    np.testing.assert_array_equal(evidence_flags(changed, 2.)[:64], expected[:64])
