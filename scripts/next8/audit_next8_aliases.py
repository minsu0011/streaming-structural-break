"""A semantic model-alias gate before releasing heldout scores."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,digest,atomic,utc_now,activity,note
from src.next8.alias_audit import audit_all_aliases


def main():
    if (BASE/'PREDICTION_FREEZE_NEXT8.json').exists():raise RuntimeError('Alias gate must precede score release')
    lock=read_json(BASE/'ALIAS_CONTRACT_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    with activity('Complete heldout model alias semantic audit','ALIAS_CONTRACT','B7_NORMALIZATION_AND_LINEAGE'):
        result=audit_all_aliases()
        atomic(BASE/'MODEL_ALIAS_CONTRACT_GATE_NEXT8.json',{'timestamp':utc_now(),**result})
    note('Complete semantic alias contract','All heldout retained inputs checked; no target scores read','B7_NORMALIZATION_AND_LINEAGE')


if __name__=='__main__':main()
