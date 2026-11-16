"""Exact background moment calculation, without simulation or model selection."""
from pathlib import Path
from fractions import Fraction
import sys
import json
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    lock = json.loads((ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json').read_text())
    rows = []
    for s in lock['scenarios']:
        if s['id'] not in ('N0', 'N1', 'N2', 'N3'): continue
        alpha = Fraction(str(s['arch_before']))
        phi = Fraction(str(s['phi_before']))
        df = s['t_df_before']
        assert df == (5 if s['id'] == 'N3' else None)
        kappa = Fraction(3) if df is None else Fraction(3*(df-2), df-4)
        v = 1-phi*phi; coefficient = kappa*alpha*alpha
        finite = coefficient < 1
        rows.append({'background': s['id'], 'locked_scenario': s, 'alpha': float(alpha),
            'standardized_noise_fourth_moment': float(kappa),
            'fourth_moment_recursion_coefficient_exact': str(coefficient),
            'fourth_moment_recursion_coefficient': float(coefficient),
            'limiting_innovation_variance': float(v), 'finite_limiting_fourth_moment': finite,
            'limiting_innovation_fourth_moment': float(kappa*v*v*(1-alpha*alpha)/(1-coefficient)) if finite else None})
    assert len(rows) == 4
    assert [r['finite_limiting_fourth_moment'] for r in rows] == [True, True, False, False]
    assert rows[2]['fourth_moment_recursion_coefficient_exact'] == '507/400'
    assert rows[3]['fourth_moment_recursion_coefficient_exact'] == '36/25'
    write_json(ROOT/'artifacts/next2/synthetic_moments/RESULTS.json', {
        'status': 'PASS', 'created_utc': utc_now(), 'rows': rows, 'script_sha256': sha256(Path(__file__)),
        'generator_sha256': sha256(ROOT/'src/next2/counterexamples.py'),
        'lock_sha256': sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
        'interpretation_sha256': sha256(ROOT/'docs/NEXT2_SYNTHETIC_MOMENT_INTERPRETATION.md'),
        'source_url': 'https://link.springer.com/article/10.1007/s00184-020-00779-x',
        'new_fits': 0, 'observations_accessed': 0, 'simulations': 0,
        'scope': 'Limiting stationary latent innovation law. Finite burn-in samples and clipped detector features are not claimed to have infinite sample moments.'})
    print('NEXT2 SYNTHETIC MOMENTS PASS', flush=True)


if __name__ == '__main__': main()
