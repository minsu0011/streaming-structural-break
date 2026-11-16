"""Compare exact memberships across legacy NumPy and current JSON hashes."""
import hashlib
import numpy as np
from src.next9.common import ids_hash


def validate_reference_membership(training, scope, selection_ids, mode):
    ids = list(map(int, selection_ids))
    assert len(ids) == len(set(ids)) == scope['training_series']
    assert ids_hash(ids) == scope['training_ids_sha256']
    legacy = hashlib.sha256(np.asarray(sorted(ids), dtype=np.int32).tobytes()).hexdigest()
    early = mode == 'EARLY'
    assert mode in ('NORMAL', 'EARLY')
    assert bool(training.get('early_only_reference_new_fit_not_canonical', False)) == early
    expected = scope['training_ids_sha256'] if early else legacy
    assert training['training_ids_sha256'] == expected
    assert training['training_series'] == len(ids)
    if early:
        assert training['training_max_online_time'] <= 199
        assert not training['validation_or_late_training_labels_used']
    return dict(status='PASS', current_membership_sha256=scope['training_ids_sha256'],
                legacy_numpy_int32_sha256=legacy, actual_sha256=training['training_ids_sha256'], mode=mode)
