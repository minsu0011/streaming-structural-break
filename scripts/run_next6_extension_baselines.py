"""Simple resumable stage loop; checkpoints and ledger use the existing scheduler."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6_extension.common import *
import calibrate_next6_extension as calibration
import evaluate_next6_extension_synthetic as evaluation

if __name__=='__main__':
    source={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'scripts/calibrate_next6_extension.py',ROOT/'scripts/evaluate_next6_extension_synthetic.py',ROOT/'src/next6_extension/synthetic.py')}
    lock=BASE/'EXTENDED_EVALUATION_IMPLEMENTATION_LOCK.json'
    if not lock.exists():save(lock.name,{'created_utc':utc_now(),'sources':source,'evaluation_partition31_not_opened':True})
    assert read_json(lock)['sources']==source
    for fold in range(5):sys.argv=['calibrate_next6_extension.py','--fold',str(fold)];calibration.main()
    save('BASELINE_CALIBRATION_FROZEN.json',{'created_utc':utc_now(),'all_five_folds_before_evaluation':True,'receipts':{str(f):digest(BASE/f'calibration/fold_{f}/DONE.json') for f in range(5)}})
    for fold in range(5):evaluation.run(fold)
