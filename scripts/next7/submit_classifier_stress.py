"""Queue fixed classifier diagnostics alongside the continuing conditional grid."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, digest, utc_now, note
from src.next6_extension.synthetic import CASES


def main():
    assert read_json(BASE / 'CLASSIFIER_STRESS_PREFLIGHT_GATE.json')['status'] == 'PASS'
    path = BASE / 'runtime/TASK_QUEUE.json'
    previous = read_json(path)
    before_sha = digest(path)
    script = 'scripts/next7/run_classifier_stress.py'
    common = {'status': 'READY', 'script': script, 'source_sha256': digest(ROOT / script),
        'hypothesis': 'F5_CAUSAL_NESTED_INTEGRATION'}
    gates = ['PREFLIGHT_NEXT7.json', 'CLASSIFIER_STRESS_PREFLIGHT_GATE.json']
    calibration = {**common, 'id': 'classifier_calibration', 'arguments': ['--mode', 'calibrate'], 'gates': gates}
    evaluation = [{**common, 'id': 'classifier_evaluation_' + case,
        'arguments': ['--mode', 'evaluate', '--case', case],
        'gates': gates + ['CLASSIFIER_CALIBRATION_FROZEN.json']} for case in CASES]
    assert not ({i['id'] for i in previous} & {i['id'] for i in [calibration, *evaluation]})
    started, pending = [], []
    for item in previous:
        parent = BASE / 'runtime/queue' / item['id']
        (started if parent.exists() and list(parent.glob('attempt_*/START.json')) else pending).append(item)
    # Preserve every started job and scientific definition. Interleaving is only
    # resource scheduling; it does not depend on any classifier stress outcome.
    ordered = started + [calibration]
    for index in range(max(len(pending), (len(evaluation) + 1) // 2)):
        ordered += evaluation[2 * index:2 * index + 2]
        ordered += pending[index:index + 1]
    assert len(ordered) == len(previous) + 24 and len({i['id'] for i in ordered}) == len(ordered)
    save('CLASSIFIER_STRESS_QUEUE_SUBMISSION.json', {'status': 'QUEUED', 'created_utc': utc_now(),
        'old_queue_sha256': before_sha, 'old_queue': previous, 'new_queue_order': [i['id'] for i in ordered],
        'new_tasks': [calibration, *evaluation], 'maximum_owned_workers': 2,
        'same_fitted_artifacts_no_new_candidate': True, 'source_sha256': digest(__file__)})
    save('runtime/TASK_QUEUE.json', ordered, False)
    note('Classifier stress queued with calibration gate', '24 diagnostic tasks interleaved with the fixed conditional grid; two-worker limit retained', 'F5_CAUSAL_NESTED_INTEGRATION')


if __name__ == '__main__':
    main()
