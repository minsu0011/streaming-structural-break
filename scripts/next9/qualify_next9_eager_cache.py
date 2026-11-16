"""Diagnose lazy-cache-only pickle change and check all ten fitted procedures."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,digest,activity,utc_now
import pickle,hashlib,copy
import numpy as np
from src.next9.runtime_loading_v5 import load_heads,load_baseline
from src.next9.runtime_loading_v6 import initialize_schema_cache
from src.next9.streaming import ResearchStream
from src.next9.negative_controls_v2 import generate


def sha(obj):return hashlib.sha256(pickle.dumps(obj,protocol=5)).hexdigest()


def main():
    records=[]
    for fold in range(5):
        for mode in ['NORMAL','EARLY']:
            with activity(f'F{fold} {mode} diagnose schema-only lazy cache and full fitted immutability',
                          'RUNTIME_REGRESSION','B9_NORMALIZATION_STATE_AVAILABILITY'):
                heads=load_heads(fold,mode);before=sha(heads)
                eager=initialize_schema_cache(copy.deepcopy(heads));eager_before=sha(eager)
                h,o,metadata=generate('GAUSSIAN','NONE',9101604,512)
                baseline=load_baseline(fold,mode);base=baseline.new_state(h)
                a=ResearchStream(heads,h);b=ResearchStream(eager,h)
                for value in o:
                    p=base.predict_one(float(value))
                    left=a.predict_one(float(value),p);right=b.predict_one(float(value),p)
                    assert left.keys()==right.keys()
                    for name in left:
                        assert np.float64(left[name]).tobytes()==np.float64(right[name]).tobytes(),name
                assert sha(eager)==eager_before,'Eager full model object changed'
                assert set(vars(heads))-set(vars(eager))==set()
                stripped=copy.copy(heads);del stripped._empty_rule_layout
                assert sha(stripped)==before,'Something besides deterministic layout changed'
                assert sha(heads)==sha(eager),'Lazy and eager final full objects differ'
                records.append(dict(fold=fold,mode=mode,points=len(o),methods=list(left),
                    exact_predictions=True,only_lazy_layout_added=True,eager_full_pickle_unchanged=True,
                    before_sha256=before,after_sha256=sha(heads)))
    atomic(BASE/'EAGER_SCHEMA_CACHE_QUALIFICATION_NEXT9.json',dict(status='PASS',timestamp=utc_now(),records=records,
        no_model_parameter_or_prediction_changes=True,source_sha256={p:digest(ROOT/p) for p in
            ['src/next9/runtime_loading_v6.py','scripts/next9/qualify_next9_eager_cache.py']}))


if __name__=='__main__':main()
