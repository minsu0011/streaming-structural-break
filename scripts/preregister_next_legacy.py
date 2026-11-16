"""Freeze AR12/ARCH2/Gaussian comparisons in the unchanged legacy ABCD bank."""
from pathlib import Path
from dataclasses import asdict
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.engine import EngineConfig
from src.next.legacy_extension import LegacyConfig
from src.next.extensions import extension_hash
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.next.ranking_objectives import implementation_hash as ranking_hash
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    path = ROOT/'configs/next_legacy_candidates.json'
    if path.exists():
        print('Legacy extension queue already frozen')
        return
    anchor_path = ROOT/'artifacts/next/LEGACY_BANK_ANCHOR_PARITY.json'
    anchor = json.loads(anchor_path.read_text(encoding='utf-8'))
    if anchor['status']!='PASS' or not anchor['float32_features_exact']:
        raise RuntimeError('Exact fixed-screen legacy anchor feature parity required before preregistration')
    candidates = []
    for eid,parent,order,norm,learner,ranking in [
        ('P001_LEGACY_AR12_ARCH1','R013_ARCH_REFERENCE',12,'arch1','M4_lightgbm',False),
        ('P002_LEGACY_AR12_NONE','P001_LEGACY_AR12_ARCH1',12,'none','M4_lightgbm',False),
        ('P003_LEGACY_AR8_ARCH2','R013_ARCH_REFERENCE',8,'arch2','M4_lightgbm',False),
        ('P004_LEGACY_AR12_ARCH2','P001_LEGACY_AR12_ARCH1',12,'arch2','M4_lightgbm',False),
        ('P005_LEGACY_AR8_GAUSSIAN','R013_ARCH_REFERENCE',8,'gaussian','M4_lightgbm',False),
        ('P006_LEGACY_AR12_GAUSSIAN','P001_LEGACY_AR12_ARCH1',12,'gaussian','M4_lightgbm',False),
        ('P007_LEGACY_AR12_HISTGB','P001_LEGACY_AR12_ARCH1',12,'arch1','M3_histgb',False),
        ('P008_LEGACY_AR12_RANK','P001_LEGACY_AR12_ARCH1',12,'arch1','histogram_ranknet',True)]:
        extension = {'kind':'legacy_bank','settings':asdict(LegacyConfig(order,norm))}
        item = {'experiment_id':eid,'parent_id':parent,'generation':9,'phase':'LEGACY_WHITENING_EXTENSION',
            'hypothesis':'The original ABCD feature bank and H median/MAD calibration may express improved whitening more effectively than the new standalone sequential bank.',
            'change':'Keep the exact legacy 51 ABCD features and H calibration. Change only input AR order, variance normalization or learner as explicitly identified by the parent comparison.',
            'expected_effect':'Isolate AR12/ARCH2/innovation-Gaussian effects from simultaneous changes in the downstream statistic bank.',
            'risks':'Legacy AR fits constrain coefficient L2 rather than poles; H-fit optimism persists; ARCH2 or Gaussian mapping can remove change signal.',
            'engine_config':asdict(EngineConfig(order=order)),'groups':['A','B','C','D'],'base':'none',
            'learner':learner,'weighting':'W3','params':{},'extension':extension,'extension_frozen_hash':extension_hash(extension),
            'legacy_anchor_parity_sha256':sha256(anchor_path)}
        if ranking:
            item.update(ranking_objective={'kind':'histogram_ranknet','bins':128},ranking_source_sha256=ranking_hash())
        candidates.append(item)
    write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':9,
        'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
        'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),'thresholds_splits_seeds_unchanged':True,
        'anchor':'AR8/ARCH1 reproduces all 51 original ABCD features over the complete 3000-series fixed screen. R013 and E001 already establish the unchanged learner behavior; no duplicate anchor fit.',
        'AR12_policy':'Separate extension of the old ridge 0.001 * trace / order and coefficient L2 <=4 formulas; no protected source edit and no online refit.',
        'ARCH2_policy':'H-only squared-innovation lag-1/2 joint ridge fit, 99% winsor cap, nonnegative slopes with sum <=0.95, mean*(1-sum) intercept, fixed H parameters and two prior squared innovations online.',
        'Gaussian_policy':'Apply the unchanged historical quantile-knot mapping to AR innovations before the original feature engine; H only, binary-search interpolation online.',
        'candidates':candidates})
    print('FROZEN LEGACY EXTENSION',len(candidates))


if __name__=='__main__':
    main()
