"""Retain all three matched raw-law control results without choosing a winner."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity
from src.next8.raw_engine_gate_control import BANKS


def main():
    lock = read_json(BASE / 'RAW_ENGINE_GATE_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    with activity('Interpret all raw-engine null mean and scale selector controls', 'RESULT_INTERPRETATION', 'B1_CONDITIONAL_DIRECTION'):
        summaries = []
        for bank in BANKS:
            name = 'RAW_ENGINE_GATE_CONTROL_' + bank + '_NEXT8.json'
            r = read_json(BASE / name)
            assert r['status'] == 'COMPLETE' and r['bank'] == bank
            summaries.append({'bank': bank, 'result_sha256': digest(BASE / name),
                'initial_counts': r['initial_classification_counts'], 'final_counts': r['final_classification_counts'],
                'selected_stable': r['final_selection']['stable_features'], 'selected_variable': r['final_selection']['variable_features'],
                'statistical_fixture_fits': r['statistical_fixture_fits'], 'H_map_quantile_fits_separate': r['H_map_quantile_fits_separate']})
        atomic(BASE / 'RAW_ENGINE_GATE_CONTROLS_NEXT8.json', {'status': 'COMPLETE', 'timestamp': utc_now(),
            'banks': summaries, 'unique_innovation_groups': 5120, 'banks_not_independent_confirmations': True,
            'statistical_fixture_fits': sum(r['statistical_fixture_fits'] for r in summaries),
            'actual_RP_fit_prediction_score_bootstrap': 0,
            'limits': ['One seed and Gaussian background; these counts are not general power or familywise error rate estimates.',
                      'NULL labels index a synthetic latent regime with no observation change. Positive laws are actual mean+2 or stdx2 aftertau.',
                      'Known raw signal does not predetermine which individual feature or state is valid; retain every unexpected classification.',
                      'Full H engine and both TRAIN selection stages are exercised; no synthetic outer B/JOINT certification or final-model promotion.',
                      'Original RP selectors, normalizers, predictions, gates and canonical roles are unchanged.']})


if __name__ == '__main__':
    main()
