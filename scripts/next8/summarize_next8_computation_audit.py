"""Final audit aggregation only after actual full replay and complete tests."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, atomic, read_json, digest, activity, utc_now, note
from src.next8.runtime_accounting import audit_admitted
from src.next8.computation_audit import summarize_full_replay, summarize_execution, summarize_computation


def main():
    admission = read_json(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')
    hb = read_json(BASE / 'runtime/HEARTBEAT.json')
    assert audit_admitted(admission, hb['productive_seconds'], digest(BASE / 'SESSION_NEXT8.json'), 36000.)
    lock = read_json(BASE / 'COMPUTATION_AUDIT_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    out = BASE / 'computation_audit'
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Existing audit aggregation evidence retained')
    with activity('Aggregate full RP computation and leakage evidence', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
        full = summarize_full_replay()
        execution = summarize_execution()
        deterministic, leakage, audit = summarize_computation(full, execution)
        for name, result in (('FULL_RP_REPLAY_NEXT8.json', full), ('SCALAR_EXECUTION_NEXT8.json', execution),
                             ('DETERMINISM_COMPUTATION_NEXT8.json', deterministic), ('LEAKAGE_AUDIT_NEXT8.json', leakage),
                             ('COMPUTATION_AUDIT_NEXT8.json', audit)):
            atomic(out / name, {'timestamp': utc_now(), **result})
        atomic(out / 'DONE.json', {'status': 'COMPLETE', 'timestamp': utc_now(),
            'files_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()}})
    note('Full computation audit PASS', 'Final archive replay and measured audit/handoff interval remain separate',
         'B7_NORMALIZATION_AND_LINEAGE', phase='FINAL_AUDIT')


if __name__ == '__main__':
    main()
