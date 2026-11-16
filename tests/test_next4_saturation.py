import pytest
from src.next4.saturation import assess


POLICY={'saturation':{'structural_hypotheses_min':10,'best_gain_strict_below':.0007}}


def full(index,gain=.0002):
    return {'candidate':str(index),'scope':'full','status':'COMPLETE','mean_delta':gain,
        'fold_scores':[.61,.60],'reference_fold_scores':[.60,.61]}


@pytest.mark.parametrize('count,gain,stop',[(9,.0002,False),(10,.0002,True),(10,.0007,False),(11,.0013,False)])
def test_frozen_strict_threshold_with_completed_evidence(count,gain,stop):
    report=assess([full(i,gain) for i in range(count)],[],[],POLICY)
    assert report['stop_new_scientific_fits'] is stop


def test_failed_screen_counts_once_and_pending_screen_never_counts():
    screens=[{'candidate':str(i),'scope':'screen','status':'COMPLETE','screen_pass':False} for i in range(10)]
    screens.append({'candidate':'100','scope':'screen','status':'COMPLETE','screen_pass':True})
    report=assess([full(0)],screens,[],POLICY)
    assert report['completed_structural_hypotheses']==10 and report['case_G']


def test_uncertainty_warnings_do_not_impose_unregistered_hard_stop():
    boot=[{'status':'PASS','paired_all_time':{'ci_2_5':-.001,'ci_97_5':.001}}]*2
    report=assess([full(0),full(1)],[],boot,POLICY)
    assert report['risk']=='HIGH' and not report['stop_new_scientific_fits']
