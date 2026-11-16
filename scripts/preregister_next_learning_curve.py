"""Freeze nested training-size diagnostics without changing validation folds."""
from pathlib import Path
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.guard import SealGuard
from src.next.diagnostic_training import ranked_training_groups,training_ids_for_groups,implementation_hash
from src.utils.artifacts import write_json,sha256,utc_now


if __name__=='__main__':
    path = ROOT/'configs/next_learning_curve_policy.json'
    if path.exists():
        print('Learning-curve policy already frozen')
    else:
        guard = SealGuard(ROOT)
        seed,fractions = 202609097,[.25,.5,.75]
        folds = []
        for fold in range(5):
            order = ranked_training_groups(guard,fold,seed,'learning-curve')
            subsets = [{'fraction':f,'training_ids':training_ids_for_groups(guard,fold,order[:math.ceil(f*len(order))])} for f in fractions]
            folds.append({'outer_fold':fold,'subsets':subsets,'validation_ids':sorted(sid for sid in guard.dev_ids if guard.split.assignment(sid)[1]==fold)})
        write_json(path,{'status':'LOCKED','created_utc':utc_now(),'seed':seed,'fractions':fractions,
            'candidates':['PRIMARY','HIGHEST_MEAN','V007_VARIANCE_FORECAST'],'folds':folds,
            'split_sha256':guard.split.digest,'seal_lock_sha256':guard.lock_sha,'training_adapter_sha256':implementation_hash(),
            'motivation':'The fixed 3000-series screen rejected the restored ARCH reference despite a stronger full result. Separate training-size sensitivity from changes in validation composition.',
            'allocation':'Within each outer training partition, rank H-group IDs by SHA256(next-learning-curve:seed:outer_fold:group). Select nested fractions, without reading targets, lengths or scores.',
            'validation':'All original held-out series remain fixed at every training size. Full-size points reuse immutable preexisting primary OOF, never refit unchanged models for curve padding.',
            'train_order':'Original fold, then original dataset ID and online time.',
            'authority':'Secondary diagnostic only. No screen threshold, primary gate, candidate ranking or split changes.',
            'interpretation':'Each outer validation score has already participated in adaptive research. These curves diagnose sample sensitivity; they are not independent confirmation.'})
        print('FROZEN LEARNING CURVE',len(folds)*len(fractions)*3,'new partial-training fits')
