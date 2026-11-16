"""Register an exact calibration-support forensic, not another threshold."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, digest, utc_now, note


def main():
    assert not list((BASE / 'conditional_null/grid').glob('*/H_*/DONE.json'))
    save('BOOTSTRAP_SUPPORT_FORENSIC_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'hypothesis': 'Empirical four-block resampling has a finite H-conditional support envelope. More draws or a longer bootstrapped path cannot create evidence outside that support.',
        'scope': 'The same4 null backgrounds,512 H references and H_lengths1000/2048/5000 as the already fixed conditional grid.',
        'H_only_input': 'Reconstruct exactly the original G2 H-prefix reference and pseudo-online nine-channel completed blocks. Verify frozen calibration extrema/cuts.',
        'support_enumeration': 'Enumerate every original four-block chunk and every tail-k/head-(4-k) splice of two possible chunks, k=1,2,3. Take the supremum joint_ratio and the existing floor1.',
        'proof': 'A sliding window of four blocks in a concatenation of four-block chunks is internal to one chunk or intersects exactly two adjacent chunks. Enumerating these possibilities covers every bootstrap path, for any draws or length.',
        'not_a_new_cutoff': 'The envelope is a diagnostic bound only. Do not use it for admission, model scores, selection, calibration or promotion.',
        'future_comparison_after_complete_grid': 'Split qualified long-null events into future maximum beyond the empirical support envelope; terminal-cut exceedance within the envelope; and earlier retained crossing without terminal-cut exceedance.',
        'interpretation': 'Beyond-envelope events cannot be eliminated merely by increasing repetitions/horizon of this unchanged empirical bootstrap and its existing order-statistic cutoff. This does not show that all possible H-only calibration methods fail.',
        'uncertainty': 'Reuse the exact existing2000 seed-pair product count draws; same pointwise diagnostic limitations as CROSSED_STATISTICS_LOCK.',
        'new_random_draws': 0, 'new_model_fits': 0, 'new_candidates': 0,
        'no_future_DGP_parameters_in_detector': True, 'before_any_complete_crossed_cohort': True,
        'G2_implementation_sha256': digest(BASE / 'G2_IMPLEMENTATION_LOCK.json'),
        'conditional_grid_lock_sha256': digest(BASE / 'CONDITIONAL_NULL_FORENSIC_LOCK.json'),
        'source_sha256': digest(ROOT / 'src/next7/bootstrap_support.py')})
    note('Exact bootstrap-support forensic frozen', 'Separate unseen empirical tails from retained earlier crossings without changing any threshold', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()
