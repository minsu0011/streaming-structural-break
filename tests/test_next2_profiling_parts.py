import json
from pathlib import Path
import numpy as np
import pytest
from src.next2.io import model_class
from src.next2.blend import FixedBlendBundle
from src.next2.runtime import prepare
from src.next2.pruned_runtime import PreparedCompactPredictor
from src.next2.used_feature_runtime import PreparedUsedPredictor
from src.next.serial_variance_runtime import PreparedSerialVariancePredictor
from src.next.fast_variance_initialization import PreparedFastVariancePredictor
from src.next2.profiling_parts import Parts

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('which',['J_original','J_compact','J_used','Y_original','Y_compact','B_original','B_compact','blend'])
def test_separate_calls_match_fused_and_tree_does_not_advance_state(which):
    lock=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    role={'J':'FROZEN_PRIMARY','Y':'FROZEN_SCIENTIFIC_REFERENCE','B':'FROZEN_ROBUST_REFERENCE'}
    if which=='blend':
        model=FixedBlendBundle.load(ROOT/'artifacts/next2/refits/BL2_C205_DEPENDENCE_80/components_v1/model.joblib')
        engine=prepare(model)
    else:
        prefix,kind=which.split('_')
        ref=lock['references'][role[prefix]]
        model=model_class(ref['candidate']).load(ROOT/ref['model_path'])
        constructor=PreparedUsedPredictor if kind=='used' else PreparedCompactPredictor if kind=='compact' else PreparedSerialVariancePredictor if prefix=='J' else PreparedFastVariancePredictor
        engine=constructor(model)
    rng=np.random.default_rng(517)
    h=rng.normal(size=512);o=rng.normal(size=200)
    fused=engine.new_state(h);parts=Parts(engine.new_state(h))
    for v in o:
        expected=fused.predict_one(v)
        parts.feature_update(v)
        assert parts.model_predict()==expected
        assert parts.model_predict()==expected
