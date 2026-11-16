"""Add bounded verification and isolated timing jobs for four nominated models."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


if __name__ == '__main__':
    path = ROOT/'configs/next_research_schedule.json'
    schedule = json.loads(path.read_text(encoding='utf-8'))
    ids = {job['id'] for job in schedule['jobs']}
    added = []
    def add(eid, suffix, script, args, completion, status='PASS'):
        key = eid.lower()+'_'+suffix
        if key in ids:
            return
        job = {'id': key, 'script': 'scripts/'+script, 'args': args, 'completion_artifact': completion}
        if status != 'PASS':
            job['completion_status'] = status
        added.append(job)
        ids.add(key)
    eid = 'F101_VARIANCE_TREES50'
    add(eid, 'fast_parity_v1', 'verify_next_fast_variance.py', ['--candidate', eid],
        f'artifacts/next/runtime_parity/{eid}/pruned_variance_fast_h/RESULTS.json')
    add(eid, 'contract_fixture_v1', 'build_next_contract_artifact.py', ['--candidate', eid, '--tag', 'v1'],
        f'artifacts/next/deployment/contracts/{eid}/v1/model.json', 'CV_FOLD_0_CONTRACT_FIXTURE_NOT_FINAL_SELECTION')
    add(eid, 'contract_v1', 'verify_next_inference_contract.py', ['--candidate', eid, '--tag', 'v1'],
        f'artifacts/next/inference_contract/{eid}/v1.json')
    add(eid, 'api_oof_v1', 'verify_next_inference_oof.py', ['--candidate', eid], f'artifacts/next/inference_oof/{eid}/RESULTS.json')
    candidates = ['V007_VARIANCE_FORECAST', 'Y001_LONG_VARIANCE_ARCH1', 'Q005_VARIANCE_NO_MEMORY', 'F101_VARIANCE_TREES50']
    for eid in candidates:
        add(eid, 'all_dev_artifact_contract_v1', 'verify_next_all_dev_artifact.py', ['--candidate', eid, '--tag', 'research_v1'],
            f'artifacts/next/all_dev_artifact_contract/{eid}/research_v1/RESULTS.json')
        add(eid, 'all_dev_official_wire_v1', 'verify_next_official_runner.py',
            ['--candidate', eid, '--model-directory', f'artifacts/next/refits/{eid}/cache_research_v1', '--tag', 'all_dev_v1'],
            f'artifacts/next/official_runner/{eid}/all_dev_v1/RESULTS.json')
    for eid in candidates[1:]:
        add(eid, 'cv_latency_fast_h_v1', 'benchmark_next_candidate.py', ['--candidate', eid, '--runtime', 'fast_h'],
            f'artifacts/next/latency/research_v1/{eid}_fast_h.json')
    for eid in ['PRIMARY', *candidates]:
        add(eid, 'all_dev_latency_v1', 'benchmark_next_all_dev.py', ['--candidate', eid], f'artifacts/next/latency/all_dev_v1/{eid}.json')
    # Place these after existing Y/Q OOF engineering and before later research.
    anchor = next(i for i, job in enumerate(schedule['jobs']) if job['id'] == 'q005_variance_no_memory_api_oof_v1')+1
    schedule['jobs'][anchor:anchor] = added
    write_json(path, schedule)
    receipt = ROOT/'artifacts/next/preregistration/ACTUAL_ARTIFACT_ENGINEERING_QUEUE.json'
    if receipt.exists():
        raise RuntimeError('Queue extension already recorded')
    write_json(receipt, {'status': 'LOCKED', 'created_utc': utc_now(), 'jobs': added,
        'schedule_sha256': sha256(path), 'all_dev_latency_policy_sha256': sha256(ROOT/'configs/next_all_dev_latency_policy.json'),
        'nomination': 'Original-CV mean leader Y001, fully reviewed V007 reference, no-memory complementary Q005, and 50-tree F101 speed candidate. No role assigned by nomination.'})
    print('ADDED ARTIFACT ENGINEERING JOBS', len(added), 'TOTAL', len(schedule['jobs']), flush=True)
