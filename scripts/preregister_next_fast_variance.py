"""Bounded tree-budget study and inherited three-role selection tolerances."""
from pathlib import Path
import copy
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.utils.artifacts import write_json,sha256,utc_now


if __name__=='__main__':
    role_path = ROOT/'configs/next_final_role_policy.json'
    if not role_path.exists():
        write_json(role_path,{'status':'LOCKED','created_utc':utc_now(),
            'old_role_policy_sha256':sha256(ROOT/'FINAL_CANDIDATE_SELECTION_POLICY.json'),
            'promotion_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
            'primary':'Among candidates passing every unchanged NEXT promotion and engineering gate, maximize full original-CV mean; ties by worst fold then median then lexical ID. If none passes, retain the frozen PRIMARY.',
            'robust':'Among other eligible fully evaluated and independently reproduced candidates, require mean >= selected primary mean - .005 and median >= selected primary median - .005; maximize worst original-CV fold, then minimize fold standard deviation, then lexical ID.',
            'fast':'Among other eligible fully evaluated and independently reproduced candidates, require mean >= selected primary mean - .010 and worst >= selected primary worst - .010. Shortlist at most five by verified isolated production latency. Select lowest measured scalar-call median, then H-amortized cost, then highest mean.',
            'backup_eligibility':'New candidates must also pass the unchanged NEXT promotion gates before a final role; existing frozen artifacts remain preserved. All finalist runtime, causality, determinism and reproduction requirements apply.',
            'backup_tolerances':'The .005 robust and .010 fast relative-score tolerances are inherited verbatim from the previous frozen role policy; they are not selected from NEXT candidate outcomes.',
            'maximum_roles':3,'fallback':'If no distinct backup satisfies all requirements, record none instead of inventing an additional candidate.',
            'authority':'DEV only. Seal, reduced and public results cannot enter selection.'})
    path = ROOT/'configs/next_fast_variance_candidates.json'
    if path.exists():
        print('Fast-variance tree-budget queue already frozen')
    else:
        parent_path = ROOT/'artifacts/next/experiments/V007_VARIANCE_FORECAST/RESULTS.json'
        parent = json.loads(parent_path.read_text(encoding='utf-8'))['candidate']
        candidates = []
        for eid,trees in [('F101_VARIANCE_TREES50',50),('F102_VARIANCE_TREES25',25)]:
            candidate = copy.deepcopy(parent)
            candidate.update(experiment_id=eid,parent_id='V007_VARIANCE_FORECAST',generation=13,phase='VARIANCE_TREE_BUDGET',
                hypothesis='The two-channel variance improvement may permit a smaller tree ensemble while retaining most of its DEV gain.',
                change=f'Only n_estimators changes from 100 to {trees}; exact same 77 features, H model, W3 weights and tree structure constraints.',
                expected_effect='Reduce scalar tree prediction cost and artifact size.',
                risks='Underfitting may reduce mean or collapse a fold. The fixed screen and every NEXT promotion gate remain unchanged.',params={'n_estimators':trees})
            candidates.append(candidate)
        write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':13,
            'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
            'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),'thresholds_splits_seeds_unchanged':True,
            'role_policy_sha256':sha256(role_path),'parent_result_sha256':sha256(parent_path),
            'grid':'Exactly 25 and 50 trees; reuse the already completed 100-tree parent. No learning-rate, depth, leaf-count or weight tuning.',
            'candidates':candidates})
        print('FROZEN FAST VARIANCE',len(candidates))
