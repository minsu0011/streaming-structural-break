"""Initialize deterministic schema cache before taking immutable model hashes."""
import numpy as np
from src.next9.runtime_loading_v5 import load_heads as _load_heads, load_baseline, reference_paths


def initialize_schema_cache(heads):
    s=heads.spec
    if s['rule_specs'] or not s['candidates']:
        return heads
    candidates=np.asarray(s['candidates'],dtype=np.int64)
    maps=np.asarray([s['run_maps'][str(j)] for j in candidates],dtype=np.int64)
    assert maps.shape[0]==len(candidates) and maps.shape[1] in (10,13)
    assert np.all((maps>=0)&(maps<13))
    names=tuple(f'F{j}_{suffix}_{k}' for j in candidates
                for suffix,count in [('fixed',5),('own',9),('S1_DIRECTION',4),
                                     ('S2_DIRECTION_RUN',13),('S3_DIRECTION_SLOPE',7),('S4_REVERSAL',5)]
                for k in range(count))
    assert names==tuple(s['input_names'])
    if hasattr(heads,'_empty_rule_layout'):
        old=heads._empty_rule_layout
        np.testing.assert_array_equal(old[0],candidates)
        np.testing.assert_array_equal(old[1],maps)
        assert old[2]==names
    else:
        heads._empty_rule_layout=(candidates,maps,names)
    return heads


def load_heads(fold,mode):
    return initialize_schema_cache(_load_heads(fold,mode))
