"""Freeze tested G2 code and common bootstrap draws before screen outcomes."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, digest, utc_now, require_guard
from src.next7.null_growth import shared_uniforms
import numpy as np


def main():
    require_guard()
    xml_path = BASE / 'audit/G2_CONTRACT_TESTS_001.xml'
    suites = list(ET.parse(xml_path).getroot().iter('testsuite'))
    assert sum(int(s.attrib.get('tests', 0)) for s in suites) == 11
    assert all(int(s.attrib.get('failures', 0)) == 0 and int(s.attrib.get('errors', 0)) == 0 for s in suites)
    sources = ['src/next7/null_growth.py', 'src/next7/tests/test_null_growth.py',
        'scripts/next7/evaluate_g2_synthetic.py', 'scripts/next7/lock_g2_implementation.py',
        'src/next7/event.py', 'src/next7/scale.py', 'src/next7/calibration.py',
        'src/next6_extension/fresh_runtime.py', 'src/next6_extension/fresh_state.py',
        'src/next6_extension/synthetic.py']
    hashes = {p: digest(ROOT / p) for p in sources}
    save('audit/G2_CONTRACT_TEST_GATE.json', {'status': 'PASS', 'created_utc': utc_now(),
        'tests': 11, 'xml_sha256': digest(xml_path), 'source_sha256': hashes})
    path = BASE / 'calibration/g2/SHARED_UNIFORMS.npy'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.save(f, shared_uniforms(), allow_pickle=False)
    save('G2_IMPLEMENTATION_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'source_sha256': hashes, 'protocol_sha256': digest(BASE / 'G2_PROTOCOL_LOCK.json'),
        'uniforms_sha256': digest(path), 'before_G2_screen': True, 'before_G2_RP_scores': True,
        'G1_source_unchanged': True})


if __name__ == '__main__':
    main()
