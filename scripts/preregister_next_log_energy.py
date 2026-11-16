"""Freeze three mechanistic alternatives to the successful ARCH forecast channel."""
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
    path = ROOT/'configs/next_log_energy_candidates.json'
    if path.exists():
        print('Log-energy queue already frozen')
    else:
        parent_path = ROOT/'artifacts/next/experiments/V007_VARIANCE_FORECAST/RESULTS.json'
        parent = json.loads(parent_path.read_text(encoding='utf-8'))['candidate']
        candidates = []
        for eid,alignment,cap in [('Z001_LOG_ENERGY_CURRENT','current',False),('Z002_LOG_ENERGY_LAGGED','lagged',False),('Z003_LOG_ENERGY_CAPPED','current',True)]:
            candidate = copy.deepcopy(parent)
            candidate.pop('keep_names')
            candidate.update(experiment_id=eid,parent_id='V007_VARIANCE_FORECAST',generation=11,phase='LOG_ENERGY_MECHANISM',
                hypothesis='ARCH1 forecast evidence is a monotone compression of lagged clipped energy when alpha is positive. Test whether robust energy compression or forecasting alignment accounts for the gain.',
                change=f'Retain raw energy and replace log ARCH variance forecast with {alignment} log1p(energy/H_variance); historical 99th-percentile clipping={cap}.',
                expected_effect='Compress rare large residuals before sequential aggregation; current alignment may improve response without the one-point forecasting lag.',
                risks='Log energy changes the null distribution and can lose scale sensitivity. These are prospective mechanism tests motivated by prior DEV results, with the original screen and promotion gates retained.',
                groups=['ENERGY_LOCAL','ENERGY_GLR','ENERGY_BAYES'],
                extension={'kind':'log_energy','settings':{'order':8,'alignment':alignment,'cap_at_h99':cap,'max_age':128}},
                extension_registration={'kind':'log_energy','module':'log_energy_evidence'},registration_source_sha256=registration_hash())
            register_extension(candidate['extension'],candidate['extension_registration'])
            candidate['extension_frozen_hash'] = extension_hash(candidate['extension'])
            candidates.append(candidate)
        write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':11,
            'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
            'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),'thresholds_splits_seeds_unchanged':True,
            'parent_result_sha256':sha256(parent_path),'candidate_capacity_and_loss_unchanged':True,
            'sequential_summaries':'Exactly the existing 13 channel summaries; 26 new features and 51 unchanged old ABCD features for every variant.',
            'candidates':candidates})
        print('FROZEN LOG ENERGY',len(candidates))
