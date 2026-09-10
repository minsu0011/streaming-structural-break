import numpy as np
import pytest
from src.next.final_fitting import fit_all_development,training_feature_names


def test_incomplete_dev_identity_cannot_use_all_dev_fit(tmp_path):
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    candidate = json.loads((root/'configs/next_variance_candidates.json').read_text(encoding='utf-8'))['candidates'][6]
    names = training_feature_names(candidate)
    class Guard:
        dev_ids = {1,2,3}
        def admit(self,ids,**kwargs):
            return ids
    metadata = {'dataset_id':np.array([1,2]),'time_online':np.array([0,0]),'target':np.array([0,1])}
    with pytest.raises(RuntimeError,match='complete DEV identity'):
        fit_all_development(np.zeros((2,len(names)),dtype=np.float32),metadata,candidate,Guard(),names)


def test_missing_chronological_training_row_is_rejected():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    candidate = json.loads((root/'configs/next_variance_candidates.json').read_text(encoding='utf-8'))['candidates'][6]
    names = training_feature_names(candidate)
    class Split:
        def assignment(self,sid):
            return 'DEVELOPMENT_POOL',sid%5
    class Guard:
        dev_ids = {1,2}
        split = Split()
        def admit(self,ids,**kwargs):
            return ids
    metadata = {'dataset_id':np.array([1,1,2,2]),'time_online':np.array([0,2,0,1]),'target':np.array([0,1,0,1])}
    with pytest.raises(ValueError,match='chronological'):
        fit_all_development(np.zeros((4,len(names)),dtype=np.float32),metadata,candidate,Guard(),names)
