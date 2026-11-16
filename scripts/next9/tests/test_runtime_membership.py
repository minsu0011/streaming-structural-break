import src.next9
import hashlib
import numpy as np
import pytest
from src.next9.common import ids_hash
from src.next9.runtime_membership import validate_reference_membership


def case():
    ids=[9,2,17]
    scope=dict(training_series=3,training_ids_sha256=ids_hash(ids))
    training=dict(training_series=3,training_ids_sha256=hashlib.sha256(np.asarray(sorted(ids),dtype=np.int32).tobytes()).hexdigest())
    return ids,scope,training


def test_legacy_identity_is_exact_not_equal_serialization():
    ids,scope,training=case()
    assert scope['training_ids_sha256']!=training['training_ids_sha256']
    assert validate_reference_membership(training,scope,ids,'NORMAL')['status']=='PASS'


def test_wrong_membership_and_time_domain_rejected():
    ids,scope,training=case()
    with pytest.raises(AssertionError):validate_reference_membership(training,scope,[9,2,18],'NORMAL')
    with pytest.raises(AssertionError):validate_reference_membership(training,scope,ids,'EARLY')


def test_early_identity_and_cutoff():
    ids,scope,training=case()
    training.update(training_ids_sha256=scope['training_ids_sha256'],early_only_reference_new_fit_not_canonical=True,
                    training_max_online_time=199,validation_or_late_training_labels_used=False)
    assert validate_reference_membership(training,scope,ids,'EARLY')['status']=='PASS'
    training['training_max_online_time']=200
    with pytest.raises(AssertionError):validate_reference_membership(training,scope,ids,'EARLY')
