"""Run one fixed statistical positive/null control, without RP access."""
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, activity, note
from src.next8.gate_reachability import cases, controlled_cache, single_control_coordinate, summarize_case


def main(index):
    policy = read_json(BASE / 'GATE_REACHABILITY_CONTROL_POLICY_NEXT8.json')
    assert policy['registered_after_outer_score_release'] and policy['no_outer_feedback_into_predictor_settings']
    lock = read_json(BASE / 'GATE_REACHABILITY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    directory = BASE / 'gate_reachability' / ('case_' + format(index, '02d'))
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise RuntimeError('Controlled diagnostic partial results preserved; no silent rerun')
    world, rho, seed = cases()[index]
    with activity('Core gate control ' + str(index) + ' ' + world, 'SYNTHETIC_GATE_CONTROL', 'B1_CONDITIONAL_DIRECTION'):
        fixture = controlled_cache(world, rho, seed)
        panels, corrections = [], []
        with single_control_coordinate(world) as screening:
            for k, scope in enumerate(fixture.scopes):
                panel, correction = screening.screen_split(fixture.cache, scope, directory / ('inner' + str(k)))
                panels.append(panel)
                corrections.append(correction)
        result = summarize_case(index, fixture, panels, corrections, directory)
    note('Controlled gate diagnostic complete', 'case' + str(index) + '; ' + result['selection']['classifications'][0]['classification'],
         'B1_CONDITIONAL_DIRECTION')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, choices=range(16), required=True)
    main(parser.parse_args().case)
