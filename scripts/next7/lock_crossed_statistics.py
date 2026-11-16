"""Freeze crossed-grid uncertainty definitions before any completed grid summary."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, digest, utc_now, note


def main():
    assert not list((BASE / 'conditional_null/grid').glob('*/H_*/DONE.json'))
    save('CROSSED_STATISTICS_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'diagnostic_protocol_sha256': digest(BASE / 'CONDITIONAL_NULL_FORENSIC_LOCK.json'),
        'count_file': 'synthetic_stats/COUNTS_2000x512.npy',
        'count_file_sha256': digest(BASE / 'synthetic_stats/COUNTS_2000x512.npy'),
        'draws': 2000, 'new_random_draws': 0,
        'primary_seed_pair_reweighting': 'For each existing count vector C_b, compute sum_ij C_bi*C_bj*M_ij /512^2. The same count weights the historical and innovation member of an original seed unit.',
        'diagonal': 'Retain all cells, including original diagonal and its squared duplicate multiplicities. This is the complete empirical product-grid functional, not a U-statistic with removed diagonal.',
        'conditional_H_intervals': 'For each fixed H_i, reweight only its512 innovation outcomes with the same count vectors; pointwise Monte Carlo resampling intervals.',
        'population_summary': 'Mean of all512x512 cells with seed-pair product percentile intervals; not262144 independent observations.',
        'H_distribution': 'Report quantiles[0,.1,.25,.5,.75,.9,.95,1] of per-H conditional rates, and fractions above.05,.1,.5. These are fixed diagnostic summaries, not new cutoffs.',
        'decomposition': 'Qualified ever latch = fixed-horizon raw maximum exceedance + retained earlier crossing without a terminal-cut exceedance; verify cellwise disjoint identity.',
        'descriptive_ANOVA': 'For the complete balanced matrix, report H row-effect, innovation column-effect and interaction mean-square fractions. Do not interpret these as causal variance components.',
        'uncertainty_limit': 'These are approximate pointwise resampling diagnostics conditional on the fixed generator and state algorithms. No exact finite-sample coverage, conditional false-alarm guarantee or anytime coverage is asserted.',
        'method_distinction': 'This preregistered seed-pair product scheme uses the same counts on both axes. It is NOT the independently reweighted-factor pigeonhole bootstrap from Owen2007 or OwenEckles2012; their consistency/conservatism results are not imported.',
        'relevant_primary_sources': ['https://arxiv.org/abs/0712.1111', 'https://arxiv.org/abs/1106.2125'],
        'reference_use': 'Primary sources motivate respecting crossed row/column dependence, not a theorem for this exact same-count procedure.',
        'no_tuning_or_promotion': True, 'before_any_complete_crossed_grid': True,
        'statistics_source_sha256': digest(ROOT / 'src/next7/crossed_statistics.py')})
    note('Crossed-grid statistics frozen', 'Respect512 seed units, distinguish conditional-H and product-grid uncertainty, no probability overclaim', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()
