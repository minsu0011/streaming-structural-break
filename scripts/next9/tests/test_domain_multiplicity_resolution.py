from src.next9.eligibility import holm


def test_registered_bootstrap_resolution_can_test_full_pair_family():
    minimum_p=1/2001
    # Full possible A budget12 times8pairs is still testable at2000draws.
    family={k:minimum_p if k==0 else 1. for k in range(12*8)}
    assert holm(family)[0]<.05
    impossible={k:minimum_p if k==0 else 1. for k in range(9*8*20)}
    assert holm(impossible)[0]>.05
