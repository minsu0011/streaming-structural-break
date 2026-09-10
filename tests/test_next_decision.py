from pathlib import Path
import json
from src.next.decision import evaluate_candidate


POLICY = json.loads((Path(__file__).resolve().parents[1]/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))


def _passing_evidence():
    return {'mean_delta':.002,'median_delta':.001,'worst_delta':0.,'bootstrap_replicates':2000,'bootstrap_probability':.9,
        'alternate_mean_deltas':[.002,0.,-.0001],'historical_worst_delta':-.001,
        'required_supported_mechanisms':['a','b'],'mechanism_deltas':{'a':.003,'b':-.004},
        'point_latency_ratio':1.5,'amortized_latency_ratio':1.4,'engineering_pass':True}


def test_promotion_needs_every_gate_and_final_replication_count():
    evidence = _passing_evidence()
    assert evaluate_candidate(POLICY,evidence)['promoted']
    evidence['bootstrap_replicates']=500
    result = evaluate_candidate(POLICY,evidence)
    assert not result['promoted'] and result['status']=='PENDING_REQUIRED_EVIDENCE'


def test_high_mean_does_not_waive_bootstrap_or_mechanism_collapse():
    evidence = _passing_evidence()
    evidence.update(mean_delta=.005,bootstrap_probability=.84)
    result = evaluate_candidate(POLICY,evidence)
    assert not result['new_generation_seal_worthy']
    evidence['bootstrap_probability']=.99
    evidence['mechanism_deltas']['b']=-.010001
    assert not evaluate_candidate(POLICY,evidence)['promoted']


def test_missing_supported_mechanism_and_latency_remain_pending():
    evidence = _passing_evidence()
    evidence['mechanism_deltas'].pop('b')
    evidence.pop('point_latency_ratio')
    result = evaluate_candidate(POLICY,evidence)
    assert result['status']=='PENDING_REQUIRED_EVIDENCE'
    assert set(result['pending_gates'])=={'supported_mechanism_worst_delta','point_latency_ratio'}


def test_conditional_new_generation_requires_matched_statistic_evidence():
    evidence = _passing_evidence()
    evidence.update(mean_delta=.0035,bootstrap_probability=.96,alternate_mean_deltas=[.002,.002,.002],historical_worst_delta=0.)
    assert not evaluate_candidate(POLICY,evidence)['new_generation_seal_worthy']
    evidence.update(matched_statistic_mean_delta=.0011,matched_supported_mechanism_gain=.0031)
    result = evaluate_candidate(POLICY,evidence)
    assert result['new_generation_seal_worthy'] and not result['actual_seal_opening_permitted']
