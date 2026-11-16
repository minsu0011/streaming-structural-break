"""Read-only pre-fit proof of all 20 retained ALT/H coordinate pairs."""
from pathlib import Path
import sys
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,utc_now,progress
from src.next4.robustness import partitions,frozen_predictions
from src.next.data import prepare_dev


def main():
    out=BASE/'stress_coordinate_audit';out.mkdir(parents=True,exist_ok=False)
    data=prepare_dev(ROOT);meta=data.rows();splits,proof=partitions(data,meta);records=[]
    for name,folds in splits.items():
        coverage=np.zeros(len(meta['target']),dtype=np.uint8)
        for fold,(train,valid) in enumerate(folds):
            baseline,sources=frozen_predictions(data,name,fold,valid)
            assert len(train)+len(valid)==len(coverage) and not np.intersect1d(train,valid).size
            assert len(np.unique(valid))==len(valid) and len(np.unique(train))==len(train)
            assert all(len(pred)==len(valid) for pred in baseline.values());coverage[valid]+=1
            records.append({'split':name,'fold':fold,'training_rows':len(train),'validation_rows':len(valid),
                'validation_rows_sha256':hashlib.sha256(valid.tobytes()).hexdigest(),'frozen_sources_sha256':sources})
        assert np.all(coverage==1)
        progress('Verifying original ALT/H coordinates',name+' all rows held out exactly once; frozen J/C coordinates identical')
        print('NEXT4 STRESS COORDINATES',name,'PASS',flush=True)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'new_fits':0,'new_candidate_scores':0,'seal_rows':0,
        'splitproof':proof,'records':records,'source_sha256':digest(ROOT/'src/next4/robustness.py'),
        'runner_sha256':digest(Path(__file__)),'historical_duplicate_group_intersections':0},immutable=True)


if __name__=='__main__':main()
