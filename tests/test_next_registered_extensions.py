import json
import subprocess
import sys
import pytest
from src.next.registered_extensions import register_extension


def test_registration_rejects_overwrite_and_invalid_local_modules():
    with pytest.raises(RuntimeError):
        register_extension({'kind': 'ar_score'}, {'kind': 'ar_score', 'module': 'markov_evidence'})
    with pytest.raises(ValueError):
        register_extension({'kind': 'unsafe'}, {'kind': 'unsafe', 'module': '../markov_evidence'})
    with pytest.raises(ValueError):
        register_extension({'kind': 'missing'}, {'kind': 'missing', 'module': 'module_that_is_missing'})


def test_registered_bank_resolves_after_a_fresh_interpreter():
    script = """
import json
from src.next.extensions import EXTENSION_MODULES, names_and_groups
from src.next.registered_extensions import register_extension
assert 'markov_transition' not in EXTENSION_MODULES
register_extension({'kind':'markov_transition','settings':{}}, {'kind':'markov_transition','module':'markov_evidence'})
print(json.dumps(names_and_groups({'kind':'markov_transition','settings':{}})[0]))
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True)
    names = json.loads(result.stdout)
    assert len(names) == 40 and names[0].startswith('markov_transition__')
