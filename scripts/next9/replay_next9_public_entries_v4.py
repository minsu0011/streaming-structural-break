"""Actual public generators: prospective NEXT9 demo plus frozen canonical roles."""
from pathlib import Path
import sys
import os
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
if os.environ.get('NEXT9_PORTABLE_REPLAY')=='1':os.environ['NUMBA_CACHE_DIR']=str(ROOT/'nc')
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

def main(original=None):
    if original:
        from src.next8.rawfree_boundary import install
        denied,opened=install(ROOT,original=original)
        assert read_json(BASE/'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json')['status']=='PASS'
    else:
        from src.next9.boundary import install_workspace_boundary
        denied=install_workspace_boundary();opened=set()
    import numpy as np
    from scripts.next9.infer_next9_v4 import infer
    from src.next9.negative_controls_v2 import generate,BACKGROUNDS,INTERVENTIONS,SEEDS,H_LENGTHS
    kind='FINAL_AUDIT' if original else 'PUBLIC_ENTRY_REPLAY'
    checks=[]
    for background in BACKGROUNDS:
        with activity(f'Actual NEXT9 public infer generator {background},42fixedcases',kind,'B9_NORMALIZATION_STATE_AVAILABILITY'):
            # One generator handles all42 consecutive series, explicitly testing
            # initialization/reset between different H lengths and interventions.
            cases=[(background,i,s,h) for i in INTERVENTIONS for s in SEEDS for h in H_LENGTHS]
            def inputs():
                for args in cases:
                    historical,online,_=generate(*args)
                    yield historical,(float(x) for x in online)
            stream=infer(inputs(),BASE)
            for args in cases:
                assert next(stream) is None
                values=np.asarray([next(stream) for _ in range(512)],np.float64)
                b,i,s,h=args;name=f'{b}_{i}_{s}_H{h}'
                directory=BASE/'portable_fixture/outer_0/NORMAL'/name
                done=read_json(directory/'DONE.json');assert digest(directory/'PREDICTIONS.npz')==done['prediction_sha256']
                with np.load(directory/'PREDICTIONS.npz') as z:np.testing.assert_array_equal(values,z['M7'])
                checks.append(dict(case=name,points=512,bitwise_fixture_equal=True,H_then_scalar_generator=True))
            try:next(stream)
            except StopIteration:pass
            else:raise AssertionError('Unexpected extra generator output')
    with activity('Actual three unchanged canonical public generators vs historical synthetic fixture',kind,'B9_NORMALIZATION_STATE_AVAILABILITY'):
        from src.next8.portable_replay import calculate_canonical
        _,canonical=calculate_canonical()
    assert not denied
    path='PUBLIC_ENTRY_FRESH_ARCHIVE_NEXT9.json' if original else 'PUBLIC_ENTRY_REPLAY_NEXT9.json'
    atomic(BASE/path,dict(status='PASS',timestamp=utc_now(),NEXT9_checks=checks,canonical_checks=canonical,
        NEXT9_fold=0,NEXT9_mode='NORMAL',prospective_demo_not_selected_by_score=True,canonical_role_changed=False,
        original_workspace_denied=bool(original),no_real_data=True,denied=denied,opened_payload_files=sorted(opened)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original');main(p.parse_args().original)
