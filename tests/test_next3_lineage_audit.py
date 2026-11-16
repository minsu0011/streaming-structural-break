import pytest
from scripts.audit_next3_lineage import validate_fold_record


def test_heuristic_summary_schema_accepts_only_omitted_pass_field():
    record={'fold':0,'ts_auc':.623,'files_sha256':{'fold_0.npy':'frozen'}}
    validate_fold_record(record,{**record,'status':'PASS'})
    validate_fold_record({**record,'status':'PASS'},{**record,'status':'PASS'})


@pytest.mark.parametrize('change',[{'ts_auc':.624},{'status':'FAIL'},{'unreviewed_extra':True}])
def test_scientific_or_status_drift_is_not_accepted_as_schema_difference(change):
    record={'fold':0,'ts_auc':.623,'files_sha256':{'fold_0.npy':'frozen'}}
    with pytest.raises(ValueError,match='receipt differ'):validate_fold_record(record,{**record,'status':'PASS',**change})
