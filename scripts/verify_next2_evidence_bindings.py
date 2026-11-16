"""Verify that selected exactness proofs still bind the actual saved OOF inputs."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    selection = json.loads((ROOT/'DEV_CANDIDATES_NEXT2.json').read_text())
    checked = {}
    def check(relative, expected):
        path = ROOT/relative; assert sha256(path) == expected, str(relative)
        checked[path.relative_to(ROOT).as_posix()] = {'sha256': expected, 'bytes': path.stat().st_size}
    for path, digest in selection['evidence_sha256'].items(): check(path, digest)
    for stage in ('all_dev', 'oof'):
        proof_path = ROOT/'artifacts/next2/shared_blend_runtime'/stage/'RESULTS.json'
        proof = json.loads(proof_path.read_text()); assert proof['status'] == 'PASS'
        assert proof['prediction_max_abs_diff'] == proof['selected_feature_max_abs_diff'] == 0
        for key, value in proof['identity']['sources'].items():
            if stage == 'all_dev': check(key, value)
            else:
                folder = ROOT/'artifacts'/('next' if key == 'J102_SERIAL_VARIANCE_512' else 'next2')/'experiments'/key/'full'
                for name, digest in value.items():
                    name = 'POLICY.json' if name == 'policy_sha256' else name
                    check((folder/name).relative_to(ROOT), digest)
    for folder in ('shared_blend_runtime/alternate', 'alternate_runtime_v2/DISTILLED'):
        proof_path = ROOT/'artifacts/next2'/folder/'RESULTS.json'; r = json.loads(proof_path.read_text())
        assert r['status'] == 'PASS' and r['points'] == 16057620 and r['prediction_max_abs_diff'] == 0
        for path, digest in r['receipts_sha256'].items(): check(path, digest)
    for variant in ('J_ORIGINAL', 'J_USED', 'BL80_COMPACT', 'BL80_SHARED_V1'):
        folder = ROOT/'artifacts/next2/latency'/variant; r = json.loads((folder/'RESULTS.json').read_text()); assert r['status'] == 'PASS'
        for name, digest in r['files_sha256'].items(): check((folder/name).relative_to(ROOT), digest)
    write_json(ROOT/'artifacts/next2/final_audit/EVIDENCE_BINDINGS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'script_sha256': sha256(Path(__file__)), 'files_checked': len(checked), 'files': checked,
        'total_bytes_hashed': sum(v['bytes'] for v in checked.values()), 'new_fits': 0, 'scores_reselected': False,
        'scope': 'Selected proofs, original saved component/blend OOF models and prediction coordinates, every alternate/H child proof, and measured timing files. Byte verification only; no seal observations.'})
    print('NEXT2 SELECTED EVIDENCE INPUT BINDINGS PASS', len(checked), flush=True)


if __name__ == '__main__': main()
