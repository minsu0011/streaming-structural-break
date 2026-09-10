import json
from pathlib import Path
from src.next2.decision_gates import assess

POLICY = json.loads((Path(__file__).resolve().parents[1]/'configs/next2/RESEARCH_POLICY_LOCK.json').read_text())


def evaluate(delta=.004, p=1., alt=(.001,.001,.001), mechanism=True, complexity=True, engineering=True):
    return assess({'mean':.62+delta,'median':.62,'worst_fold':.60},
        {'mean':.62,'median':.62,'worst_fold':.60},
        {'probability_delta_positive':p,'probability_delta_above_minus_0005':p},
        [{'mean_delta':d,'worst_delta':0.} for d in alt], {'minimum_matched_fold_delta':0.},
        complexity_decreased=complexity,engineering_pass=engineering,independent_mechanism_gain=mechanism,policy=POLICY)


def test_generation_literal_disjunction():
    assert evaluate(.004,mechanism=False)['generation']['status']=='PASS'
    assert evaluate(.003,mechanism=False)['generation']['status']=='FAIL'
    assert evaluate(.003,mechanism=True)['generation']['status']=='PASS'
    assert evaluate(.003,p=.95)['generation']['status']=='FAIL'
    assert evaluate(.003,alt=(.001,0.,.001))['generation']['status']=='FAIL'


def test_missing_evidence_is_not_a_pass():
    r=evaluate(.004,engineering=None,complexity=None)
    assert r['performance_scientific']['status']=='PASS'
    assert r['performance']['status']=='PENDING'
    assert r['distilled']['status']=='PENDING'
    assert evaluate(.001,engineering=None)['performance']['status']=='FAIL'


def test_noninferiority_and_performance_are_different():
    r=evaluate(0.,p=.95)
    assert r['distilled']['status']=='PASS'
    assert r['performance_scientific']['status']=='FAIL'
    assert evaluate(0.,p=.89)['distilled']['status']=='FAIL'
    assert evaluate(.002,alt=(.001,.001,-.02))['performance_scientific']['status']=='PASS'
    assert evaluate(.002,alt=(.001,.001,-.02))['distilled']['status']=='FAIL'
