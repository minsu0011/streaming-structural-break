"""Freeze two forward-H calibration variants before their screen outcomes."""
from pathlib import Path
import copy
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.registered_extensions import register_extension,registration_hash
from src.next.extensions import extension_hash
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.utils.artifacts import write_json,sha256,utc_now


if __name__=='__main__':
    path = ROOT/'configs/next_forward_variance_candidates.json'
    if path.exists():
        print('Forward-H variance queue already frozen')
    else:
        parent_path = ROOT/'artifacts/next/experiments/V007_VARIANCE_FORECAST/RESULTS.json'
        parent = json.loads(parent_path.read_text(encoding='utf-8'))['candidate']
        candidates = []
        for eid,fraction in [('H101_FORWARD_VARIANCE_HALF',.5),('H102_FORWARD_VARIANCE_75',.75)]:
            candidate = copy.deepcopy(parent)
            candidate.update(experiment_id=eid,parent_id='V007_VARIANCE_FORECAST',generation=14,phase='FORWARD_H_CALIBRATION',
                hypothesis='Reusing H to fit both AR/ARCH parameters and null channel calibration can underestimate forward uncertainty. Hold out a chronological H tail to calibrate the new channels.',
                change=f'Fit new AR8/ARCH1 only on the first {fraction:.0%} of H, then causally process the rest of H and calibrate there. The old 51 ABCD features retain their unchanged full-H recipe.',
                expected_effect='Improve comparability of new evidence across unseen series and historical regimes.',
                risks='Fewer H observations estimate parameters and calibration; H itself may drift. No refit after tail calibration, no online parameter updates, and no original split or promotion-gate changes.',
                extension={'kind':'forward_variance','settings':{'order':8,'normalization':'arch1','max_age':128,'fit_fraction':fraction}},
                extension_registration={'kind':'forward_variance','module':'forward_variance_evidence'},registration_source_sha256=registration_hash())
            candidate['keep_names'] = [name.replace('conditional_variance__','forward_variance__') for name in parent['keep_names']]
            register_extension(candidate['extension'],candidate['extension_registration'])
            candidate['extension_frozen_hash']=extension_hash(candidate['extension'])
            candidates.append(candidate)
        write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':14,
            'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
            'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),'thresholds_splits_seeds_unchanged':True,
            'parent_result_sha256':sha256(parent_path),'sequential_summaries_and_learner_unchanged':True,
            'calibration_burn':'Discard min(32, tail_length//4) points from the observed H tail only.',
            'candidates':candidates})
        print('FROZEN FORWARD-H VARIANCE',len(candidates))
