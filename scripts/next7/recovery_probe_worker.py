"""Harmless queue recovery probe, isolated from every scientific artifact."""
from pathlib import Path
import argparse
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import psutil
from src.next7.common import BASE, read_json, save, digest, utc_now


def main(probe_id, behavior):
    assert probe_id.startswith('recovery_probe_') and all(c.isalnum() or c == '_' for c in probe_id)
    relative = f'audit/recovery_probes/{probe_id}'
    directory = BASE / relative
    checkpoint = directory / 'CHECKPOINT.json'
    resumed = checkpoint.exists()
    if not resumed:
        save(relative + '/CHECKPOINT.json', {'status': 'COMPLETE', 'part': 1,
            'sum_of_squares': sum(i * i for i in range(1000)), 'created_utc': utc_now()})
    else:
        assert read_json(checkpoint)['sum_of_squares'] == sum(i * i for i in range(1000))
    starts = list(directory.glob('CHILD_READY_*.json'))
    number = len(starts) + 1
    save(relative + f'/CHILD_READY_{number}.json', {'status': 'READY', 'pid': os.getpid(),
        'process_created': psutil.Process().create_time(), 'resumed_checkpoint': resumed,
        'checkpoint_sha256': digest(checkpoint), 'created_utc': utc_now()})
    if resumed and behavior == 'resume_after_one':
        save(relative + '/RESULT.json', {'status': 'COMPLETE', 'checkpoint_sha256': digest(checkpoint),
            'total': read_json(checkpoint)['sum_of_squares'] + sum(i * i for i in range(1000, 2000)),
            'parts': 2, 'checkpoint_reused_without_overwrite': True})
        return
    # A bounded readiness barrier lets the controller interrupt this owned
    # process. It is a process-lifecycle test, never research-time padding.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        time.sleep(.02)
    raise TimeoutError('Probe controller did not interrupt its ready child within30 seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-id', required=True)
    parser.add_argument('--behavior', choices=['resume_after_one', 'always_wait'], required=True)
    args = parser.parse_args()
    main(args.probe_id, args.behavior)
