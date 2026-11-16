"""Allocate a train-only null calibration split before any threshold result."""
from pathlib import Path
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.guard import SealGuard
from src.next.diagnostic_training import ranked_training_groups,training_ids_for_groups,implementation_hash
from src.utils.artifacts import write_json,sha256,utc_now


if __name__=='__main__':
    path = ROOT/'configs/next_alarm_calibration_policy.json'
    if path.exists():
        print('Alarm calibration policy already frozen')
    else:
        guard = SealGuard(ROOT)
        seed,folds = 202609098,[]
        for fold in range(5):
            order = ranked_training_groups(guard,fold,seed,'null-alarm-calibration')
            cut = math.ceil(.2*len(order))
            folds.append({'outer_fold':fold,
                'training_ids':training_ids_for_groups(guard,fold,order[cut:]),
                'calibration_ids':training_ids_for_groups(guard,fold,order[:cut]),
                'validation_ids':sorted(sid for sid in guard.dev_ids if guard.split.assignment(sid)[1]==fold)})
        write_json(path,{'status':'LOCKED','created_utc':utc_now(),'seed':seed,'alphas':[.01,.05],
            'candidates':['PRIMARY','V007_VARIANCE_FORECAST'],'folds':folds,
            'split_sha256':guard.split.digest,'seal_lock_sha256':guard.lock_sha,
            'training_adapter_sha256':implementation_hash(),'alarm_code_sha256':sha256(ROOT/'src/next/alarm_calibration.py'),
            'allocation':'20% of outer-training H groups in fixed SHA order form a calibration-only set; the other 80% train the model. Full original outer validation remains held out.',
            'threshold':'No-break calibration-series maxima only; one-based ceil((n+1)*(1-alpha)) order statistic; strict exceedance. If rank exceeds n, emit no alarms.',
            'refit':'No refit after calibration; thresholds and outer predictions come from exactly the same smaller training model.',
            'early_diagnostics':'First crossing stops the detector. An alarm before tau is premature, not counted as a later successful detection. Delay summaries condition on detected series; report miss and premature fractions alongside them.',
            'window_policy':'Within-horizon clean detection rates use break series with at least that many observed post-break points; horizons fixed at 5,10,20,50.',
            'authority':'Secondary alarm/delay diagnostic, never a TS-AUC promotion criterion or tuned production threshold.',
            'limitations':'Nominal alpha describes the calibration rule, not a distribution-free guarantee for heterogeneous competition sequences. Variable observed lengths affect whole-series false alarms. Outer data already participated in adaptive research.'})
        print('FROZEN NESTED ALARM DIAGNOSTIC',10,'fits')
