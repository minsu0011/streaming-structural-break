"""Register a distinct H-only memory-admission rule before any G2 outcomes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,save,read_json,digest,utc_now,activity,note


def main():
    assert (BASE/'G1_BOOTSTRAP_DONE.json').exists()
    assert not (BASE/'G2_PROTOCOL_LOCK.json').exists()
    with activity('register H-only null-growth memory hypothesis','PREREGISTRATION','F4_NULL_GROWTH_CORRECTION'):
        policy={
            'status':'FROZEN','created_utc':utc_now(),'id':'G2_H_NULL_MAXIMUM_ADMISSION',
            'family':'F4_NULL_GROWTH_CORRECTION','parent':'G1 S1_BLOCK_LOG_MAD / C1_SOFT_3OF4 + frozen NEXT6 hard4',
            'hypothesis':'Conditioning permanent memory admission on H-only maximum evidence at the number of observed blocks can reduce false latch without rewriting the current detector.',
            'one_structural_change':'Only admission to event memory; all G1 current-state values, raw fresh-confirmation rules, scale features, fixed cutoffs and BL80 source unchanged.',
            'failure_evidence':{'G1_null_N0_16384_latch':.615234375,
                'G1_RP_M_EVENT_delta':-.001374571894001675,
                'false_latched_negative_pair_contribution':-.01199901087930746,
                'true_latched_positive_pair_contribution':.010647958159426887},
            'H_null_reference':{
                'reference_prefix':'first floor(.6*len(H)) observations',
                'pseudo_online':'remaining40% of H, replayed causally using only that reference prefix',
                'source_states':'Frozen SafeState(prefix) and ScaleState(prefix); extract only completed16-point core8 node z and S1 signed-scale z.',
                'ratio_denominator':'Use the same core and full-H-routed scale cutoffs as the actual G1 live state.',
                'raw_normality_assumption':'H is treated as a normal reference; hidden H changes can invalidate that assumption.',
                'prefix_shortening_limit':'Calibration uses a shorter reference than live inference, a conservative-motivated approximation that must be checked empirically; not a distribution-free theorem.',
                'insufficient_support':'If reference_prefix<512 or fewer than8 complete pseudo blocks or either reference invalid, no new G2 event can latch; G1 current state still runs.',
            },
            'H_only_bootstrap':{
                'replicates':512,'seed':20260913,
                'shared_draws':'One fixed512x256 uniform matrix reused for every H; no dataset_id, hash-dependent seed or seed shopping.',
                'resample_unit':'Moving segment of4 consecutive H pseudo-online block vectors; start=floor(U*(n_H_blocks-3)); concatenate sampled segments.',
                'null_path_length_blocks':1024,
                'grid_blocks':[4,8,16,32,64,128,256,512,1024],
                'score_per_block':'Same max(core hard4/core_cut, scale C1/scale_cut) from the latest4 block vectors; no score before4 complete blocks.',
                'grid_statistic':'maximum of the null score through each grid block count',
                'order_statistic':'ceil(.95*(512+1))=488, 1-based; no percentile tuning',
                'null_admission_cut':'max(1, selected H bootstrap maximum order statistic)',
                'dependence':'Preserves local four-block segments only; joins and long dependence are approximate.',
                'information_limit':'512 resamples do not create512 independent historical calibration observations.',
            },
            'online_memory':{
                'routing':'Smallest grid value >= the number of blocks observed so far; never actual future total length.',
                'qualification':'A G1 fresh confirmation at a completed block AND current G1 ratio strictly > routed H-null admission cut.',
                'strength':'1-exp(-current_ratio/admission_cut)',
                'update':'max(previous_qualified_ever,qualified_strength), only at qualifying fresh confirmations',
                'episodes':'False-to-true qualification at complete blocks, capped7; first/last qualified times and last qualified direction stored.',
                'current_state':'All G1 current-state coordinates remain identical.',
                'raw_vs_qualified':'Keep raw G1 confirmation diagnostics separate from qualified event memory; do not count raw confirmations as qualified latches.',
                'after_1024_observed_blocks':'No supported extrapolation. Admission cut=max(12/core_cut,4/scale_cut)+1e-9, so no new event qualifies after16384 observations. Existing qualified memory remains. This is an explicit late-change limitation.',
                'probability_claim':'None. Prefix-dependent H bootstrap thresholds are not an anytime FWER guarantee.',
            },
            'screen':{
                'partition':31,'indices':[0,63],'cases':['N0','N2','V4','O1','RETURN_NORMAL'],
                'null_horizon':16384,'change_horizon':4096,'report_prefixes':[1024,4096,16384],
                'fixed_rule_to_continue_full':'Both N0/N2 16384 qualified latch<=.50 AND V4 1024 first detection>=.10; otherwise stop G2 as a failed admission structure.',
                'not_statistical_promotion':'This screen only governs computation; it is not independent validation or a promotion gate.',
            },
            'if_screen_passes':{
                'synthetic':'Original512 fixed indices and required cases, no new seeds',
                'RP_candidates':['G2_EVENT','G2_FULL'],
                'G2_EVENT':'.9 BL80+.1 qualified-ever score',
                'G2_FULL':'Same preregistered M_FULL offset-logistic penalty/optimizer, replace only event/age/direction coordinates with qualified-memory equivalents.',
                'nested':'Reuse exact original inner BL80 OOF; H-only signal needs no population-label fit; train correction using only outerTRAIN inner-OOF.',
                'all5_frozen_before_G2_outer_score_release':True,
            },
            'stop_rules':['Never tune the .95 percentile, bootstrap segment length, grid, weights or qualification strength after G2 results.',
                'If synthetic improves but RP adds no gain, close this memory-admission representation.',
                'If admission becomes too slow or support collapses, diagnose that failure; do not silently fall back to an unqualified latch.'],
            'budget':{'screen_slots_reserved':1,'full_slots_reserved':2,'learned_meta_recipes_reserved':1,
                'total_full_reserved_after_G2':14,'remaining_full_slots':2,'total_learned_meta_recipes':4,
                'major_families_after_G2':5},
            'G1_sources_and_results_immutable':True,
            'causal_contract_tests_required_before_screen':True,
        }
        save('G2_PROTOCOL_LOCK.json',policy)
        usage=read_json(BASE/'runtime/CANDIDATE_USAGE.json')
        usage.update({'screen_reserved':7,'full_reserved':14,'learned_meta_reserved':4,
            'major_families_reserved':5,'G2_screen_completed':False,'updated_utc':utc_now()})
        save('runtime/CANDIDATE_USAGE.json',usage,False)
        save('research_notes/NULL_GROWTH_RATIONALE.md', '''# Why permanent memory requires a separate admission contract

G1 improved the fixed V4 synthetic detection rate but lost RP ranking performance. Exact pair accounting assigns -0.01199901087930746 of mean fold TS-AUC to pairs where only the negative observation has a latch, compared with +0.010647958159426887 where only the positive has a latch. These are algebraic partitions of fixed predictions, not causal treatment effects.

A stationary detector can have acceptable short-run alarm frequency and still acquire a false permanent latch over repeated opportunities. Under an idealized stationary null with independent opportunities and a fixed positive trigger probability q, the chance of at least one trigger over K opportunities is 1-(1-q)^K. This elementary calculation motivates the observed-block-count correction; it is not an independence assumption about the overlapping G1 windows or a formal statement about its floating-point infinite-horizon behavior.

The e-detector literature controls average run length under stated model conditions. Average run length is a different criterion from the probability that an irreversible false event has ever occurred. We do not transfer its guarantees to this state engine. [Shin, Ramdas and Rinaldo, E-detectors](https://arxiv.org/abs/2203.03532v4).

Confidence-sequence change-detection constructions also distinguish their sequential estimation assumptions from their resulting alarm-frequency guarantees. The cited reduction establishes an average-run-length guarantee; it does not establish a guarantee for our H-calibrated latch. [Shekhar and Ramdas, 2024](https://proceedings.mlr.press/v235/shekhar24a.html).

Reusing a finite calibration set can make otherwise marginally valid conformal p-values dependent. Conditional calibration requires extra analysis. Our dependent time-series H blocks do not automatically satisfy an independent-sample conformal model. [Bates et al., Testing for Outliers with Conformal p-values](https://arxiv.org/abs/2104.08279v3).

G2 therefore makes a limited empirical proposal: use an H-only moving-block bootstrap of maximum fresh evidence to qualify memory admission. The current detector is unchanged. H-prefix shortening, short historical support, bootstrap joins, and repeated threshold comparisons can all cause miscalibration. No probability calibration or anytime false-alarm theorem is claimed. The same frozen synthetic and RP gates must decide whether the proposal is useful.

The qualifying threshold depends on blocks actually observed. It never receives the future series length. Beyond the preregistered support of1024 observed blocks, new memory admission stops explicitly; existing memory and current-state inference continue. This must be reported as a late-change detection limitation, not hidden as a guarantee of safety.

Retrieved primary sources on2026-09-13. No external competition data, hidden tests, leaderboard or submission was used.
''')
        save('research_notes/CANDIDATE_BUDGET_ACCOUNTING.md', '''# NEXT7 budget accounting

The immutable initial protocol reserves at most16 complete scientific candidates and5 learned meta recipes. G1 uses12 full candidates, including3 standalone diagnostics, and3 learned recipes. Its5-member conditional uncertainty comparison set is frozen and all G1 recipes remain unchanged. Those failed comparisons do not become deployment roles.

G2 reserves2 further scientific candidates and1 additional learned recipe. Only the fixed screen can authorize their computation. The totals become14 scientific slots and4 learned recipes, with2 scientific slots and1 learned recipe available. The final retained candidate set remains at most5 and deployment/research roles at most4. Original rejected artifacts remain in the workspace for audit; the final handoff will distinguish selected roles from research history.

This accounting does not authorize extra thresholds, extra weight menus or score-driven reinterpretation of a bug. Unused budget is not a requirement to fit more models.
''')
    note('G2 H-only admission protocol frozen','Implement and test before the fixed64-seed screen','F4_NULL_GROWTH_CORRECTION')


if __name__=='__main__':
    main()
