"""One-factor age expansion for the successful two-channel variance family."""
from pathlib import Path
import copy
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.extensions import extension_hash
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.utils.artifacts import write_json,sha256,utc_now


if __name__=='__main__':
    path = ROOT/'configs/next_long_variance_candidates.json'
    if path.exists():
        print('Long-variance queue already frozen')
    else:
        candidates = []
        for eid,parent_id in [('Y001_LONG_VARIANCE_ARCH1','V007_VARIANCE_FORECAST'),('Y002_LONG_VARIANCE_ARCH2','Q009_VARIANCE_ARCH2_PAIR'),('Y003_LONG_VARIANCE_EWMA','Q010_VARIANCE_EWMA_PAIR')]:
            path_parent = ROOT/'artifacts/next/experiments'/parent_id/'RESULTS.json'
            candidate = copy.deepcopy(json.loads(path_parent.read_text(encoding='utf-8'))['candidate'])
            candidate.update(experiment_id=eid,parent_id=parent_id,generation=12,phase='VARIANCE_AGE_GRID',
                hypothesis='The variance-pair family gains from sequential integration. Test whether extending the fixed logarithmic age grid captures slower changes beyond 128 points.',
                change='Only max_age changes from 128 to 512; add the preimplemented 256/512 ages. Same channels, H fit, learner and weighting.',
                expected_effect='Stronger accumulated weak scale evidence at long break ages.',
                risks='Long windows mix pre/post observations near a recent break, increase state memory and can delay a response. The original screen still decides full-CV admission.')
            candidate['extension']['settings']['max_age']=512
            candidate['extension_frozen_hash']=extension_hash(candidate['extension'])
            candidates.append(candidate)
        write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':12,
            'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
            'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),'thresholds_splits_seeds_unchanged':True,
            'grid':'Existing independent variance-bank age option 512; the frozen original core-bank grid is unchanged.',
            'candidates':candidates})
        print('FROZEN LONG VARIANCE',len(candidates))
