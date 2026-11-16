import src.next9
import numpy as np
from src.next9.science_summary import aggregate_contrast,score_promotion


def test_original_equal_fold_mean_and_paired_resampling():
    records=[];samples=[]
    for k,d in enumerate([.001,.002,.003,.004,.005]):
        records.append(dict(fold=k,major_domain_supported=True,points={'M7':{'TS_AUC':.6+d},'M0':{'TS_AUC':.6}}))
        noise=np.linspace(-.02,.02,2000)
        samples.append(dict(M7=.6+d+noise,M0=.6+noise))
    r=aggregate_contrast(records,samples,'M7','M0')
    assert abs(r['point']-.003)<1e-14 and abs(r['ci_low']-.003)<1e-14
    assert r['folds']==[0,1,2,3,4]
    records[0]['points']['M7']['TS_AUC']=None
    q=aggregate_contrast(records,samples,'M7','M0')
    assert q['folds']==[1,2,3,4] and abs(q['point']-.0035)<1e-14


def test_promotion_does_not_ignore_losing_supported_domain():
    core=dict(status='SUPPORTED',folds=list(range(5)),point=.002,ci_low=.001,positive_folds=5,worst_fold=.001)
    early={**core,'point':0.}
    domain={**core,'point':-.0011}
    gate=score_promotion(core,early,[dict(mode='NORMAL',domain='TIME_6',contrast=domain)])
    assert gate['core_score_gate'] and not gate['score_gates_pass']
    assert not gate['canonical_role_changed']
