"""Run the original finite negative-control recipes once."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,digest,atomic,utc_now,activity,note
from src.next8.cache import immutable_npz
from src.next8.negative_controls import fixed_controls


def main():
    lock=read_json(BASE/'NEGATIVE_CONTROL_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    directory=BASE/'negative_controls';directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise RuntimeError('Existing negative-control results preserved')
    with activity('Execute fixed confounding and unobservable-state controls','NEGATIVE_CONTROLS','B1_CONDITIONAL_DIRECTION'):
        result,arrays=fixed_controls();immutable_npz(directory/'ILLUSTRATIONS.npz',**arrays)
        atomic(directory/'DONE.json',{'timestamp':utc_now(),**result,'illustration_sha256':digest(directory/'ILLUSTRATIONS.npz'),
            'implementation_lock_sha256':digest(BASE/'NEGATIVE_CONTROL_IMPLEMENTATION_LOCK_NEXT8.json')})
    note('Complete fixed negative controls','Time pooling, fixed U, survivor composition, same-raw recurrence and forbidden schemas','B1_CONDITIONAL_DIRECTION')


if __name__=='__main__':main()
