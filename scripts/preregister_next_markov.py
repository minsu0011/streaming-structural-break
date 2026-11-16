"""Freeze conditional transition evidence and its small matched ablations."""
from pathlib import Path
from dataclasses import asdict
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.engine import EngineConfig
from src.next.markov_evidence import MarkovConfig
from src.next.extensions import extension_hash
from src.next.registered_extensions import register_extension, registration_hash
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.next.ranking_objectives import implementation_hash as ranking_hash
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    path = ROOT / 'configs/next_markov_candidates.json'
    if path.exists():
        print('Markov queue already frozen')
        return
    candidates = []
    registration = {'kind': 'markov_transition', 'module': 'markov_evidence'}
    def add(eid, parent, change, base='arch8', config=None, groups=None, ranking=False):
        extension = {'kind': registration['kind'], 'settings': asdict(config or MarkovConfig())}
        register_extension(extension, registration)
        item = {'experiment_id': eid, 'parent_id': parent, 'generation': 4, 'phase': 'TRANSITION',
            'hypothesis': 'Conditional transition probabilities can change while marginal moments stay stable; H-posterior predictive evidence may capture this missed dependence.',
            'change': change, 'expected_effect': 'Detect nonlinear transition-pattern changes with a fixed H-only reference and bounded rolling state.',
            'risks': 'Quartile quantization loses amplitude information; overlapping Markov transitions are dependent; Bayes factors are features and are not calibrated posterior probabilities or anytime-valid tests.',
            'engine_config': asdict(EngineConfig()), 'groups': groups or ['TRANSITION', 'GLR', 'BAYES'],
            'base': base, 'learner': 'M4_lightgbm', 'weighting': 'W3', 'params': {},
            'extension': extension, 'extension_frozen_hash': extension_hash(extension),
            'extension_registration': registration, 'registration_source_sha256': registration_hash()}
        if base == 'arch8':
            item['base_family_selection'] = 'ABCD'
        if ranking:
            item.update(learner='histogram_ranknet', ranking_objective={'kind': 'histogram_ranknet', 'bins': 128}, ranking_source_sha256=ranking_hash())
        candidates.append(item)
    add('T001_MARKOV_STANDALONE', 'N021_STANDALONE_S3', 'Replace innovation-product summaries by raw-input quartile transition evidence.', base='none')
    add('T002_MARKOV_ARCH', 'R013_ARCH_REFERENCE', 'Append 40 conditional-transition statistics to unchanged ARCH8 ABCD.')
    add('T003_MARKOV_ARCH_GLR', 'T002_MARKOV_ARCH', 'Remove Bayes-factor and transition-surprise families; retain conditional GLR only.', groups=['GLR'])
    add('T004_MARKOV_ARCH_BAYES', 'T002_MARKOV_ARCH', 'Remove GLR and transition-surprise families; retain integrated Bayes-factor evidence only.', groups=['BAYES'])
    add('T005_MARKOV_ARCH_LAG1', 'T002_MARKOV_ARCH', 'Only remove lags 2, 4 and 8.', config=MarkovConfig(lags=(1,)))
    add('T006_MARKOV_ARCH_PRIOR4', 'T002_MARKOV_ARCH', 'Only decrease alternative Dirichlet row concentration from 16 to 4.', config=MarkovConfig(prior_strength=4.))
    add('T007_MARKOV_ARCH_AGE128', 'T002_MARKOV_ARCH', 'Only remove age hypotheses 256 and 512.', config=MarkovConfig(max_age=128))
    add('T008_MARKOV_ARCH_RANK', 'T002_MARKOV_ARCH', 'Only replace W3 binary loss by the already audited 128-cell same-time ranking surrogate.', ranking=True)
    write_json(path, {'status': 'LOCKED', 'created_utc': utc_now(), 'generation': 4,
        'fitting_code_sha256': fitting_hash(), 'predictor_code_sha256': predictor_hash(),
        'robustness_policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'thresholds_splits_seeds_unchanged': True,
        'categorical_policy': 'Four raw-value bins from H-only 25/50/75% quantiles; right insertion at ties; lags 1/2/4/8; H transition counts plus 0.5 per cell.',
        'null_model': 'Integrate per-row transition probabilities under Dirichlet(H_counts + 0.5).',
        'alternative_model': 'Integrate under Dirichlet(strength * null_posterior_row_mean), strength 16 primary or 4 ablation.',
        'age_policy': 'Powers of two 1..512 primary, 1..128 ablation; mixture weights are successive age widths. Online windows contain online transition endpoints, with past-H lagged origins permitted.',
        'streaming_policy': 'Incremental table-count, log-gamma-ratio lookup and n*log(n) updates; fixed arrays; O(lags * ages) per point. No online model refit.',
        'mathematical_tests': 'Direct full Dirichlet marginal likelihood and conditional saturated GLR comparisons pass through ring wraparound; unchanged-marginal AR-sign change responds; exact prefix and batch/stream parity pass.',
        'candidates': candidates})
    print('FROZEN MARKOV', len(candidates), 'candidates')


if __name__ == '__main__':
    main()
