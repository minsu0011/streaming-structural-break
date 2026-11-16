"""Freeze conditional expert admission and deduplicate existing three-expert fits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress


def main():
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json'
    ids=read_json(ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json')['families']['F']['candidate_ids']
    atomic_json(BASE/'families/F_MENU_LOCK.json',{'status':'FROZEN_CONDITIONAL_FAMILY_BEFORE_FIRST_NEW_F_SCORE','created_utc':utc_now(),
        'family':'F','candidate_ids':ids,'prior_work_reference':'Every A/C/D/H candidate already uses 0.9 BL80 + 0.1 standalone expert, exactly a fixed 72/18/10 J/C/new expert composition with intermediate float32 rounding.',
        'F1_disposition':'DEDUPLICATE_EXISTING_REGISTERED_THREE_EXPERT_COMPOSITION; reuse admitted source candidate score, no 70/20/10 vs 72/18/10 micro-weight experiment',
        'third_expert_pool_families':['A','C','D','H'],
        'third_expert_admission':{'full_mean_delta_min':.0003,'standalone_mean_min':.54,'paired_ci_lower_strict_min':0.,
            'paired_probability_min':.9,'mean_within_time_rank_correlation_max_strict':.98,
            'correlation_references':['J102','C205','BL80'],'completed_fixed_forensic_required':True},
        'third_selection':'Among admitted experts, largest preregistered fixed-blend full mean gain; exact tie broken by experiment ID; choose once after the pool finishes screen/full decisions and required diagnostics',
        'replacement_pool':['B1_SPLIT_ENERGY_BRANCHES','B2_LOG_ENERGY_SEMITAIL','G1_H_CONTEXT_BASE'],
        'replacement_admission':'Completed full result satisfies original performance numeric mean/median/worst gate; not assumed promoted',
        'replacement_selection':'Largest full mean delta among admitted augmented-J base candidates, tie broken by experiment ID; choose once',
        'F2_recipe':'0.9 * existing selected augmented-J 80/20 C205 composite float32 score + 0.1 * admitted third-expert float32 score; float64 weighted sum rounded float32',
        'F2_new_fits_for_primary_scores':0,'F2_identity_is_new_fixed_combination':True,
        'F2_screen_requirement':'Selected components must both have the original locked screen OOF; apply the unchanged NEXT4 screen gate to their fixed combination before full scoring',
        'F2_full_candidate_budget_charge':1,'nested_scope':'No learned meta coefficients. Each component prediction is honest outer OOF. Expert nomination itself is adaptive on DEV and remains part of selection uncertainty.',
        'stop_rule':'No third expert admitted or no augmented-J candidate admitted means explicit conditional skip. Otherwise one fixed F2 recipe only. No weights, substitutes, rank-average or fallback after its score.',
        'promotion_authority':'Unchanged NEXT4 complete performance/robustness/generation gates; no inheritance of a component promotion claim',
        'new_F_scores_before_lock':0,'policy_sha256':digest(policy_path)},immutable=True)
    progress('Conditional multi-expert family frozen','F1 deduplicates existing 72/18/10 compositions; F2 is at most one structurally new augmented-J plus independently admitted expert combination')
    print('NEXT4 FAMILY F POLICY FROZEN; selection pending closed independent-expert evidence',flush=True)


if __name__=='__main__':main()
