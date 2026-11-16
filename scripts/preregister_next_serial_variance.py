"""Two fixed-horizon tests of H-only serial correction in GLR/Bayes evidence."""
from pathlib import Path
import copy
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.registered_extensions import register_extension, registration_hash
from src.next.extensions import extension_hash
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.utils.artifacts import write_json, sha256, utc_now

if __name__ == '__main__':
    out = ROOT/'configs/next_serial_variance_candidates.json'
    if out.exists():
        raise RuntimeError('Serial-correction queue is already frozen')
    candidates, parents = [], {}
    for eid, parent_id in [('J101_SERIAL_VARIANCE_128', 'V007_VARIANCE_FORECAST'), ('J102_SERIAL_VARIANCE_512', 'Y001_LONG_VARIANCE_ARCH1')]:
        path = ROOT/'artifacts/next/experiments'/parent_id/'RESULTS.json'
        parents[parent_id] = sha256(path)
        candidate = copy.deepcopy(json.loads(path.read_text(encoding='utf-8'))['candidate'])
        candidate.update(experiment_id=eid, parent_id=parent_id, generation=18, phase='H_SERIAL_GLR_BAYES_CORRECTION',
            hypothesis='The forecast channel is serially correlated even under a stationary null. An independent-observation sum variance can overstate long-age GLR and mean-prior Bayes evidence.',
            change='Only replace age n by the H-fitted AR(1) variance of each channel sum in GLR and analytic mean-prior Bayes summaries; derived memory follows corrected GLR. Keep instantaneous, EWMA, CUSUM, H AR/ARCH fitting and old 51 features unchanged.',
            expected_effect='Improve comparability across H persistence regimes and reduce spurious long-age evidence.',
            risks='AR(1) is an approximation to nonlinear, clipped energy channels. Estimated H autocorrelation is noisy, and removing persistence can weaken a genuine variance shift. This is not a valid e-process or guaranteed false-alarm likelihood.',
            extension_registration={'kind': 'serial_variance', 'module': 'serial_variance_evidence'},
            registration_source_sha256=registration_hash())
        candidate['extension']['kind'] = 'serial_variance'
        candidate['keep_names'] = [name.replace('conditional_variance__', 'serial_variance__') for name in candidate['keep_names']]
        register_extension(candidate['extension'], candidate['extension_registration'])
        candidate['extension_frozen_hash'] = extension_hash(candidate['extension'])
        candidates.append(candidate)
    write_json(out, {'status': 'LOCKED', 'created_utc': utc_now(), 'generation': 18,
        'fitting_code_sha256': fitting_hash(), 'predictor_code_sha256': predictor_hash(),
        'robustness_policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'thresholds_splits_seeds_unchanged': True, 'parent_results_sha256': parents,
        'historical_estimator': 'Pearson lag-1 correlation of each H-calibrated channel after the unchanged H burn-in and [-12,12] clipping; clamp rho to [-.95,.95], use zero for a constant H channel.',
        'sum_variance': 'v(n)=n+2*sum_{k=1}^{n-1}(n-k)*rho^k; strictly positive for the stationary AR(1) approximation.',
        'bayes_summary': 'For S~N(n*mu,v(n)) and mu~N(0,s2), log B=-.5*log(1+s2*n*n/v)+.5*s2*(n/v)^2*S*S/(1+s2*n*n/v), using the unchanged s2=.25/1 tau mixtures.',
        'numerical_anchor': 'Synthetic rho=0 is bitwise equal to every original channel statistic over 1100 points with age grid through 512; H channel calibration and all local statistics also match the parent exactly.',
        'maximum_new_candidates': 2, 'candidates': candidates})
    print('FROZEN SERIAL-VARIANCE CORRECTION', len(candidates), sha256(out), flush=True)
