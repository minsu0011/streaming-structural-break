"""Compile fixed scientific results after complete evidence synthesis."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, activity, note
from src.next8.research_reports import build_scientific_reports


def main():
    lock = read_json(BASE / 'RESEARCH_REPORTS_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    with activity('Assemble actual scientific tables and Korean feature cards', 'SCIENTIFIC_REPORTING', 'B4_MOVEMENT_VS_DISCRIMINATION'):
        build_scientific_reports()
    note('Scientific tables and Korean cards complete', 'Final audit and archive delivery remain separate', 'B4_MOVEMENT_VS_DISCRIMINATION')


if __name__ == '__main__':
    main()
