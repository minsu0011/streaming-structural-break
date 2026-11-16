"""Finite hypothesis queue, frozen before the first NEXT supervised score."""
from dataclasses import asdict
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.engine import EngineConfig, FEATURE_NAMES, FEATURE_GROUPS, feature_hash
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    path = ROOT/'configs/next_initial_candidates.json'
    if path.exists():
        print('Initial candidate queue already frozen')
        return
    candidates = []
    core = ['I','S1','S2','S3','S4']
    def add(label, hypothesis, change, *, parent='PRIMARY', config=None, groups=None, base='none', learner='M4_lightgbm', weighting='W3', risks='DEV selection bias; correlated evidence; possible latency cost'):
        eid = f'N{len(candidates)+1:03d}_{label}'
        candidates.append({'experiment_id': eid, 'parent_id': parent, 'generation': 1,
            'hypothesis': hypothesis, 'change': change, 'expected_effect': 'Improve causal same-time ranking or reveal a matched negative result',
            'risks': risks, 'engine_config': asdict(config or EngineConfig()),
            'groups': core if groups is None else groups, 'base': base, 'learner': learner,
            'weighting': weighting, 'params': {}, 'phase': label.split('_')[0]})
        return eid
    anchor = add('SEQ_AR8_SD_CORE', 'H-whitened sequential mean/scale/dependence/distribution evidence may rank breaks across heterogeneous series.',
                 'New bounded sequential statistic engine on AR8 SD innovations')
    for order in [1,2,4,12]:
        add(f'WHITEN_AR{order}', 'A different fixed AR horizon may better remove stationary historical dependence.',
            f'Only AR order changes from 8 to {order}', parent=anchor, config=EngineConfig(order=order))
    for normalization in ['raw','mad','arch1','arch2','ewma','uniform','gaussian']:
        add(f'NORM_{normalization.upper()}', 'Innovation normalization may reduce cross-series nuisance scale, volatility or tail heterogeneity.',
            f'Only innovation normalization changes from SD to {normalization}', parent=anchor, config=EngineConfig(normalization=normalization))
    arch_anchor = add('ARCH_REFERENCE', 'The new matrix/training adapter must reproduce the unchanged strongest-mean binary reference.',
        'No model/statistic change; engineering and screening reference', groups=[], base='arch8')
    add('SEQ_HISTGB', 'A second small tree learner may capture nonlinear sequential evidence differently.',
        'Only learner changes from LightGBM to HistGB', parent=anchor, learner='M3_histgb')
    for family in ['S1','S2','S3','S4']:
        add(f'ABLATE_REMOVE_{family}', 'A matched removal measures actual marginal utility of the evidence family.',
            f'Remove only {family} from the core', parent=anchor, groups=[g for g in core if g != family])
    for family in ['S1','S2','S3','S4']:
        add(f'STANDALONE_{family}', 'A small standalone family establishes signal strength before combination.',
            f'Keep instantaneous I and only {family}', parent=anchor, groups=['I',family])
    add('STANDALONE_I', 'Instantaneous innovation amplitude is a cheap reference for sequential value.',
        'Keep only instantaneous features', parent=anchor, groups=['I'])
    for family in ['G','B','M']:
        add(f'ADD_{family}', 'Fixed-age likelihood or causal evidence memory can recover persistent weak changes.',
            f'Append only {family} to core', parent=anchor, groups=core+[family])
    for family in ['G','B']:
        add(f'STANDALONE_{family}', 'A compact standalone likelihood detector may carry distinct break evidence.',
            f'Instantaneous I plus only {family}', parent=anchor, groups=['I',family])
    for family in ['S1','S2','S3','S4','G','B','M']:
        add(f'ARCH_ADD_{family}', 'New evidence may complement the strong ARCH8 representation despite weak standalone ranking.',
            f'Append only the {family} block to the unchanged ARCH8 model inputs', parent=arch_anchor, base='arch8', groups=[family])
    add('ARCH_ADD_CORE', 'A jointly trained small tree may exploit complementary core innovation statistics.',
        'Append the already ablated core engine block to ARCH8', parent=arch_anchor, base='arch8', groups=core)
    for weighting in ['W0','W1','W2']:
        add(f'WEIGHT_ARCH_{weighting}', 'Train-only alternative weighting tests the relation between binary loss and pair-weighted TS-AUC.',
            f'Only weighting changes from W3 to {weighting}', parent=arch_anchor, base='arch8', groups=[], weighting=weighting)
    policy = {'status': 'LOCKED', 'created_utc': utc_now(),
        'robustness_policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'fitting_code_sha256': fitting_hash(), 'predictor_code_sha256': predictor_hash(),
        'engine_feature_sha256_default': feature_hash(EngineConfig()),
        'feature_names': list(FEATURE_NAMES), 'feature_groups': list(FEATURE_GROUPS),
        'screen_reference': 'Fresh refit of frozen PRIMARY recipe on identical fixed 3000-ID subset; validation original folds 0 and 4',
        'screen_admission': 'candidate two-fold mean minus freshly refit PRIMARY screen mean >= -0.002',
        'screen_bootstrap_replicates': 500, 'full_cv_authority': 'original five DEV folds only',
        'mean_glr': 'known H-standardized zero mean and unit variance; ages 1,2,4,8,16,32,64,128',
        'variance_glr': 'known mean zero, variance MLE with second-moment floor 1e-8',
        'joint_glr': 'mean and variance MLE only at ages >=4, variance floor 1e-8; avoids one-observation singularity',
        'bayes_lite': {'mean_prior_variances': [.25,1.], 'mean_amplitudes': [-.25,.25,-.5,.5,-1.,1.],
                       'variance_ratios': [.5,.75,1.5,2.], 'age_prior_cell_widths': [1,1,2,4,8,16,32,64]},
        'uncalibrated_statistics_note': 'H-fit residuals need not be independent Gaussian. GLR corrections and Bayes-like scores are ranking features, not certified p-values, posterior probabilities or e-values.',
        'small_grid_controls': 'No hyperparameter/seed/gate sweep. Later hypotheses require a new prospective queue with explicit parent and one-factor change.',
        'complementarity_before_late_ensemble': 'within-time rank correlation <0.98 and opposite-sign subgroup deltas of magnitude >=0.002 in supported mechanisms or regimes; score scale alignment must be learned from training only',
        'candidates': candidates}
    write_json(path, policy)
    print('FROZEN', len(candidates), 'candidates', flush=True)


if __name__ == '__main__':
    main()
