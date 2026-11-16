import src.next9
import pickle
import numpy as np
from src.next9.data import FitScope
from src.next9.transitions import fit_law, summaries, SUMMARY_NAMES


def test_transition_law_never_updates_on_validation():
    state = np.tile(np.arange(4), 128)
    ids = np.repeat(np.arange(64), 8)
    scope = FitScope(frozenset(range(64)), frozenset(range(64,128)), frozenset(), 'toy')
    run = np.ones(len(state)); slope = np.zeros(len(state)); context = np.zeros((len(state),6))
    law = fit_law(state, state, run, slope, context, ids, ids, scope, 'T1_SELF_MARKOV')
    before = pickle.dumps(law)
    for order in (np.arange(len(state)), np.arange(len(state))[::-1]):
        p = law.probabilities(state[order], run[order], slope[order], context[order])
        assert (p.argmax(axis=1) == state[order]).all()
    assert before == pickle.dumps(law)


def test_surprise_uses_previous_issued_distribution():
    s = np.array([2,0,3])
    previous = np.array([[.1,.1,.7,.1],[.8,.1,.05,.05],[.25]*4])
    current = np.full((3,4),.25)
    out = summaries(s, np.ones(3), np.ones(3), np.ones(3), np.ones(3), previous, current)
    assert len(SUMMARY_NAMES) == 8
    np.testing.assert_allclose(out[:2,2], -np.log([.7,.8]))
    np.testing.assert_array_equal(out[2,1:], 0)
    changed = summaries(s, np.ones(3), np.ones(3), np.ones(3), np.ones(3), previous, np.tile([.1,.2,.3,.4],(3,1)))
    np.testing.assert_array_equal(out[:,2], changed[:,2])
