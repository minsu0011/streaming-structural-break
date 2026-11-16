"""The public adapter must suppress disk-cache initialization before imports."""
import json
from pathlib import Path
import subprocess
import sys


def test_verifier_changes_only_entry_import_and_owned_output_directory():
    root = Path(__file__).resolve().parents[1]
    original = (root / 'scripts/verify_next5_streaming.py').read_text(encoding='utf-8')
    repaired = (root / 'scripts/verify_next5_streaming_readonly_core.py').read_text(encoding='utf-8')
    expected = original.replace('from src.next5.streaming import infer,load_prepared',
        'from src.next5.readonly_streaming import infer,load_prepared').replace(
        "out=BASE/'runtime_validation/api'/cid", "out=BASE/'runtime_validation/api_readonly_v1'/cid")
    assert repaired == expected


def test_fresh_entry_import_is_readonly_and_matches_original_batch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'cold.json'
    result = subprocess.run([sys.executable, '-B', str(root / 'scripts/probe_next5_readonly_entry.py'),
        '--output', str(output)], cwd=root, capture_output=True, text=True, timeout=180,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(output.read_text(encoding='utf-8'))
    assert receipt['status'] == 'PASS' and receipt['cold_audit_enabled_before_entry_import']
    assert receipt['persistent_filesystem_write_attempts'] == receipt['new_fits'] == 0
    assert receipt['suppressed_project_cache_requests'] > 0
    assert receipt['synthetic_fixture_points'] == 192
    assert receipt['batch_probability_max_abs_difference'] <= 1e-15
