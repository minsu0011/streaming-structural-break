"""Freeze finite structural menus and all gates before NEXT7 score release."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, digest, csv_write, utc_now, require_guard, activity, note


def main():
    guard = require_guard()
    assert not (BASE / 'NEXT7_PROTOCOL_LOCK.json').exists()
    with activity('preregister NEXT7 structural budget and contracts', 'PREREGISTRATION'):
        inherited = read_json(ROOT / 'artifacts/next6/NEXT6_PROTOCOL_LOCK.json')
        split = read_json(ROOT / 'NEXT5_SPLIT_LOCK.json')
        hard4_calibration = ROOT / 'artifacts/next6_extension/calibration/fresh/DONE.json'
        hard4_cut = read_json(hard4_calibration)['thresholds']['FRESH']
        protocol = {
            'status': 'FROZEN', 'created_utc': utc_now(), 'version': 'NEXT7_G1_V1',
            'before_any_NEXT7_scientific_score': True,
            'split_sha256': guard.split.digest, 'outer_split': 'PRIMARY',
            'inner_mapping': inherited['inner_mapping'],
            'inner_mapping_authority_sha256': digest(ROOT / 'artifacts/next6/NEXT6_PROTOCOL_LOCK.json'),
            'new_outer_or_inner_split': False,
            'target': '1[t >= tau] for tau>=0; otherwise zero; tau only offline target/forensic',
            'forbidden': ['Shadow', 'Seal A/B/C/D', 'reduced', 'public', 'cloud', 'future suffix',
                          'future total length', 'dataset_id feature', 'tau inference', 'graph performance search'],
            'frozen_primary': 'BL80; matched RP = .8 * J102 + .2 * C205',
            'matched_baseline_authority': 'artifacts/next6/MODEL_COMPARISON_NEXT6.csv',
            'frozen_hard4': {'source': 'src.next6_extension.fresh_runtime.SafeState',
                'threshold': hard4_cut,
                'threshold_authority_sha256': digest(hard4_calibration),
                'block_size': 16, 'confirmation_blocks': 4, 'never_modified': True},
            'G1_primary_scale': 'S1_BLOCK_LOG_MAD', 'G1_primary_confirmation': 'C1_SOFT_3OF4',
            'scale_menu': {
                'S1_BLOCK_LOG_MAD': {
                    'question': 'Does translation-invariant robust scale of distinct blocks retain contraction evidence?',
                    'statistic': 'ell(B)=log(max(median(abs(B-median(B))), eps)); exactly 16 current values',
                    'eps': 'max(1e-12 * robust MAD of H first60%, 1e-30)',
                    'H_split': 'fit=floor(.6*len(H)); disjoint 16-point blocks separately in each part; trailing remainder unused for scale calibration',
                    'center': '(n_cal*median(ell_cal)+32*median(ell_fit))/(n_cal+32)',
                    'dispersion': 'sqrt((n_cal*s_cal^2+32*s_fit^2)/(n_cal+32)), s=max(IQR(ell)/1.349,.15)',
                    'evidence': 'clip((ell-current_H_center)/dispersion,-4,4)',
                    'min_support': 'H>=512, >=16 fit blocks and >=8 calibration blocks; finite nonconstant H',
                    'fallback': 'If n_cal<16 use fit-block median/dispersion for both center and scale; if numeric reference invalid output INSUFFICIENT_REFERENCE and no confirmation',
                    'no_AR_ARCH_duplication': 'Raw block MAD intentionally isolates marginal scale; existing BL80 whitening remains unchanged',
                    'difference_from_old': 'B2/R2 used individual residual energy + EWMA/CUSUM; this uses block-centered MAD with exactly finite 16-value input',
                },
                'S2_BLOCK_SCALE_PIT': {
                    'parent': 'S1_BLOCK_LOG_MAD', 'only_change': 'H-relative mapping of the same block scale',
                    'u': '(count_cal(<ell)+.5*count_cal(=ell)+32*F_fit_midrank(ell)+.5)/(n_cal+33)',
                    'F_fit_midrank': '(count_fit(<ell)+.5*count_fit(=ell)+.5)/(n_fit+1)',
                    'evidence': 'clip(log(u/(1-u))/(pi/sqrt(3)),-4,4)',
                    'fallback': 'If n_cal<16, replace calibration sample with fit sample; both are H-only',
                    'difference_from_old_R2': 'PIT of fresh centered 16-point block MAD, finite four-block aggregation, no per-point energy PIT EWMA',
                },
            },
            'confirmation_menu': {
                'C0_HARD4': {'formula': 'max over channel/direction of min(abs(z_1..z_4)) when all four share sign',
                    'distinct_blocks': 4, 'parent': 'NEXT6 exact frozen baseline; also applied to signed scale as isolated ablation'},
                'C1_SOFT_3OF4': {'formula': 'max over direction d in {-1,+1}: max(0,sum(clip(d*z_k,-4,4))/4) if at least3 of4 d*z_k>0, else0',
                    'distinct_blocks': 4, 'minimum_same_direction': 3,
                    'stop_rule': 'No threshold/count/window edits after calibration or evaluation'},
                'C2_ACCUMULATED_4BLOCK': {'parent': 'C1_SOFT_3OF4',
                    'only_change': 'minimum same-direction blocks =2 (preregistered structural comparator, never a post-result relaxation)',
                    'formula': 'same signed bounded four-block mean, require2 same-direction blocks',
                    'stop_rule': 'Reject if single-shock pairing shows excess latch or long-null gate fails; do not micro-tune'},
            },
            'calibration': {
                'purpose': 'fixed synthetic null reference conditional on observed H length; no validation labels',
                'partition': 30, 'indices': [0, 255], 'nulls': ['N0', 'N1', 'N2', 'N3'],
                'H_lengths': [1000, 1500, 2048, 3000, 5000], 'horizon': 1024,
                'alpha_path': .05,
                'cut': 'for each H length/rule/scale, max over four backgrounds of order statistic ceil(.95*(256+1)), capped at256; strict > alarm',
                'H_routing': 'largest listed H length <=actual H; endpoints clamped; H<1000 uses H1000 cutoff plus explicit short-H status',
                'simultaneous_new_scale_union': 'No claim of familywise false-alarm control across separate candidate columns',
                'NEXT6_core_cut': 'Reuse frozen exact cut; union alarm has no new 5% guarantee',
                'H_pseudo_online': 'Each H supplies scale distribution; no online values update H calibration',
                'threshold_table_frozen_before_evaluation': True,
                'not_anytime_probability_guarantee': True,
            },
            'memory': {
                'update': 'only on a completed block with fresh confirmation strictly above its frozen cut',
                'strength': '1-exp(-confirmation_strength/cut), clipped to[0,1]; no update on raw shock alone',
                'ever': 'max(previous_ever, current_confirmed_strength)',
                'episode': 'false-to-true confirmation transition; capped episode count7',
                'direction': 'sign of triggering scale channel; core direction from its dominant channel',
                'ages': 'observed online counter minus first/last confirmed time, log1p capped atlog1p(16384) for model features',
                'recovery': 'Current state recovers independently; event memory is never reset by recovery',
                'current_finite_memory': 'New scale block +4 completed blocks +2 recovery blocks; event variables explicitly excluded from coupling proof',
                'forensic': ['firstShockTime', 'firstConfirmationTime', 'firstLatchTime',
                    'confirmationChannel', 'confirmationBlocks', 'confirmationStrength', 'eventDirection', 'eventAge'],
            },
            'current_scale_score': '1-exp(-abs(latest_signed_scale)); held constant between completed blocks; zero during warmup',
            'current_state_score': '1-exp(-max(core_persistent/core_cut, scale_confirmation/scale_cut)); bounded ranking score, not probability',
            'integration_menu': {
                'M0': 'unchanged BL80',
                'M_CURRENT': '.9*BL80+.1*CURRENT_STATE',
                'M_EVENT': '.9*BL80+.1*EVER_EVENT',
                'M_SCALE': '.9*BL80+.1*CURRENT_SCALE',
                'M_EVENT_SCALE': '.9*BL80+.05*EVER_EVENT+.05*CURRENT_SCALE',
                'M_FULL_FIXED': '.9*BL80+(.1/3)*(CURRENT_STATE+EVER_EVENT+CURRENT_SCALE)',
                'I3_CURRENT_EVENT': 'small regularized offset-logistic correction using current/event coordinates',
                'M_FULL': 'same offset-logistic learner, append signed scale coordinates only',
                'I4_EVENT_SCALE_TREE': 'small depth2 residual correction tree with event and signed scale coordinates; preregister exact implementation before fitting',
                'AMPLITUDE_CONTROL': '.9*BL80; both original-cut and independently calibrated-cut alarm reporting',
            },
            'offset_logistic': {
                'offset': 'logit(clip(BL80,1e-6,1-1e-6)), fixed coefficient1',
                'features': 'current score, event score, episode count/7, clipped log ages, recovery flag, confirmation direction; FULL additionally signed scale, abs scale and scale confirmation score',
                'standardization': 'mean/std from outer TRAIN OOF rows only, std floor1e-4; clip standardized values at8',
                'objective': 'mean W3-weighted logistic loss + .01/2 * sum(beta^2); W3 normalized to mean1',
                'intercept': False, 'optimizer': 'L-BFGS-B, maxiter300, ftol1e-12, gtol1e-8, zero initialization',
                'no_hyperparameter_selection': True,
            },
            'nested': {
                'base': 'For each outer fold, new J102/C205 inner fits on outerTRAIN minus held inner groups, exact frozen original recipe',
                'label_free_branch': 'H-only calibration and externally frozen synthetic null thresholds need no label fit; OOF identity is still checked',
                'head': 'Train only on inner OOF base/state/event/scale rows; outer validation labels unavailable',
                'outer_base': 'Reuse hash-validated existing NEXT5 matched RP outer fits',
                'release': 'All five outer inner/head contracts and prediction artifacts frozen before any NEXT7 outer score release',
                'source_change': 'New prediction math is new candidate; real software repairs get separate provenance and scope audits',
            },
            'budgets': {'major_families': 6, 'screen_candidates': 30, 'full_5fold_candidates': 16, 'meta_finalists': 5,
                'confirmation_rules': 4, 'scale_representations': 3},
            'promotion': {'mean_delta_min': .0012, 'paired_CI95_lower_strictly_above': 0,
                'P_delta_positive_min': .95, 'matched_positive_folds_min': 4,
                'worst_fold_delta_min': -.001, 'latency_ratio_max': 3,
                'leakage': 'PASS', 'major_robustness_collapse': False},
            'generation': {'mean_delta_min': .0025, 'CI_lower_positive': True, 'ALT_consistently_positive': True,
                'scale_down_or_shock_improves': True, 'not_BL80_redundancy': True},
            'robustness_role': {'mean_delta_min': -.0005, 'worst_fold_delta_min': 0,
                'single_and_repeated_shock_false_latch_absolute_reduction_min': .10,
                'V3_or_V4_first_detection_absolute_improvement_min': .05, 'paired_CI_support_required': True},
            'major_collapse_definition': {
                'null': 'Any background/horizon false latch > BL80 matched-calibration alarm by>.05 and paired95%CI lower>0',
                'shock': 'Single or repeated shock excess latch > frozen NEXT6 hard4 by>.10 and paired95%CI lower>0',
                'scale': 'Any V1/V2/V3/V4 first detection <BL80 by>.10 and paired95%CI upper<0',
                'hard_long_null': 'Any16384 null ever-confirmed rate>.50 blocks a promoted event-memory role',
            },
            'uncertainty': {'whole_series_paired_draws': 2000, 'strong_candidate_draws': 5000,
                'strong_rule': 'Choose5000 before drawing only if mean_delta>=.0025, otherwise2000; never extend draws to flip gate',
                'interpretation': 'conditional on fitted OOF; does not adjust all adaptive DEV reuse'},
            'time_fracture': [[0,9],[10,24],[25,49],[50,99],[100,199],[200,399],[400,None]],
            'break_age': [[0,4],[5,9],[10,19],[20,49],[50,None]],
            'H_bins': [[512,1249],[1250,1749],[1750,2499],[2500,3999],[4000,None]],
            'evaluation': {'partition': 31, 'indices': [0,511], 'null_horizons': [1024,4096,16384],
                'changed_horizons': [1024,4096], 'cases': ['N0','N1','N2','N3','V1','V2','V3','V4','O1','OB16',
                    'REPEATED_SHOCK','TEMP_SCALE_PULSE','RETURN_NORMAL','D1','D2','M1','M2','A1','A2','T1','T2','LATE_PERSISTENT','SCALE_PLUS_SHOCK'],
                'H_transport_partition': 33, 'H_transport_lengths': [1000,1500,3000,5000],
                'existing_synthetic_seeds_reused': True, 'adaptive_DEVELOPMENT_reuse_disclosed': True},
            'ALT': 'Only strong finalists; reuse already frozen ALT_A/ALT_B mappings and H regime artifacts; no new seeds',
            'stop_search': 'All structural hypotheses delta<=.0005 -> stop micro-search and pursue failure forensic/runtime/reproduction',
            'minimum_productive_seconds': 36000, 'audit_after_productive_minimum': True,
        }
        save('NEXT7_PROTOCOL_LOCK.json', protocol)
        save('NEXT7_CANDIDATE_BUDGET.json', {'status':'FROZEN', 'created_utc':utc_now(), **protocol['budgets'],
            'families': ['F1_SIGNED_SCALE','F2_FRESH_CONFIRMATION','F3_CONFIRMED_EVENT_MEMORY',
                         'F4_NULL_GROWTH_CORRECTION','F5_CAUSAL_NESTED_INTEGRATION','F6_SUPPORT_OR_TIMING_STRUCTURAL_REPAIR'],
            'bug_reruns_count_separately_with_receipt': True, 'unused_slots_are_not_obligations': True})
        save('NEXT7_RANDOMNESS_LOCK.json', {'status':'FROZEN', 'created_utc':utc_now(),
            'synthetic_authority_sha256':digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
            'synthetic_seed':read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')['seed'],
            'calibration_partition':30, 'evaluation_partition':31, 'H_transport_partition':33,
            'bootstrap_seed':20260913, 'base_learner_seed':20260908,
            'new_fold_seed':False, 'inner_mapping_sha256':digest(ROOT/'artifacts/next6/NEXT6_PROTOCOL_LOCK.json'),
            'existing_screen_ids':split.get('screen_ids'), 'no_seed_shopping':True})
        board = []
        for i, (name, evidence, test, stop) in enumerate([
            ('scale contraction','NEXT6 negative tail support and hard4 confirmation loss','Signed fresh-block MAD; direction-wise V1..V4 and RP gain','Synthetic-only benefit without RP increment: stop representation'),
            ('single/repeated shock latch','NEXT6 current recovery succeeded; BL80 lifetime max contaminated','Matched shock/control event latch and finite-current coupling','Shock-driven memory violates contract: repair before scoring'),
            ('current vs ever target','Official labels stay1 after change; current state can recover','RETURN_NORMAL target alignment and event persistence','Memory only late gain: timing forensic'),
            ('long-null accumulation','Max-like historical memory can eventually latch','1024/4096/16384 first latch and final memory distribution','If growth fails gate, preregister H-only null-growth candidate'),
            ('H calibration','NEXT6 short-H alarms increased','Frozen H buckets, H-only support shrinkage and transport','No retrospective bin/threshold tuning'),
            ('BL80 redundancy','Existing strong features already summarize scale and persistence','Pearson/Spearman/within-time and nested incremental gain','corr>.995 and delta<.0005: no independent-signal claim'),
            ('adaptive DEV reuse','RP repeatedly used NEXT2..NEXT6','Finite candidate budget and outer-blind fit/release receipts','All structural gains<=.0005: diagnostic phase'),
        ], 1):
            board.append({'id':f'B{i}','name':name,'evidence':evidence,'severity':'HIGH' if i<=5 else 'MEDIUM',
                'expected_TS_AUC_impact':'UNKNOWN_BEFORE_NEXT7','confidence':'mechanism motivated, empirical gain unknown',
                'overfit_risk':'HIGH_ADAPTIVE_DEV','engineering_cost':'MODERATE','next_test':test,'stop_rule':stop,'status':'OPEN'})
        save('BOTTLENECK_BOARD_INITIAL_NEXT7.json', {'created_utc':utc_now(),'items':board})
        save('runtime/BOTTLENECK_BOARD_CURRENT.json', {'created_utc':utc_now(),'items':board}, False)
        csv_write('FAILURE_DEDUPLICATION_NEXT7.csv', [
            {'old':'B2 lower/upper log semitail','source':'src/next4/scale_asymmetry.py','new':'S1_BLOCK_LOG_MAD','difference':'block-centered translation-invariant scale, finite16raw scope, no CUSUM/EWMA'},
            {'old':'R2 point energy PIT','source':'artifacts/next5/registrations/R2_CONDITIONAL_SCALE_PIT.json','new':'S2_BLOCK_SCALE_PIT','difference':'distribution of disjoint block MAD; H fit/cal support shrinkage; no per-point energy PIT'},
            {'old':'two-pole temporal state','source':'src/next4/temporal_state.py','new':'F3_CONFIRMED_EVENT_MEMORY','difference':'latched event updates only after fresh multi-block confirmation; separate recoverable state'},
            {'old':'lifetime max','source':'src/next/channel_evidence.py','new':'F3_CONFIRMED_EVENT_MEMORY','difference':'no update from instantaneous raw shock; complete block provenance required'},
            {'old':'NEXT6 M2 edge search','source':'src/next6/graph.py','new':'NONE','difference':'closed; graphs explanatory only'},
        ])
        for filename, text in {
            'CURRENT_STATE_CONTRACT.md': '# NEXT7 current state\n\nThe frozen NEXT6 hard4 engine is retained exactly. New scale uses disjoint current blocks of16. Its scale buffer, four-block confirmation and two-block recovery memory are finite. Event variables are separate. NEXT6 dependence covariates have a4-point lag footprint: disjoint current timestamps do not imply completely disjoint raw dependence inputs or independent blocks. No graph selection affects NEXT7 performance.\n',
            'CONFIRMATION_ENGINE_CONTRACT.md': '# Fresh confirmation\n\nC0 hard4, C1 three-of-four signed bounded mean, and C2 two-of-four signed bounded mean are registered before any NEXT7 score. Exactly four distinct completed blocks are examined. No incomplete block counts. Cutoffs use the frozen null-calibration procedure and are frozen before evaluation. The NEXT6 hard4 threshold remains unchanged. No post-result threshold, count, block-size or weight edits.\n',
            'EVENT_MEMORY_CONTRACT.md': '# Confirmed event memory\n\nRaw shock alone never updates event memory. Only a completed block meeting a frozen confirmation rule can update the running maximum of confirmed strength. Current state can recover while historical event memory remains. Memory is an evidence/ranking score, not a calibrated break probability. Inference accepts H and one current scalar; no tau, dataset_id, suffix or future length. All latch episodes receive causal timing, direction and block provenance; tau is attached only offline.\n',
        }.items():
            save(filename, text)
    note('NEXT7 protocol frozen', 'Finite scale/confirmation/blend menu; nested OOF and promotion gates fixed', 'F1_SIGNED_SCALE')


if __name__ == '__main__':
    main()
