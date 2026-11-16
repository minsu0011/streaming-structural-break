"""Check exact original ResearchPool training memberships of every M0 component."""
from pathlib import Path
import sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.next6.common import BASE,read_json,save,digest,activity
from src.next6.data import source

def main():
    with activity('matched_reference_training_membership','FINAL_AUDIT'):
        data=source();proof=[]
        for fold in range(5):
            train=np.sort(data.ids[data.series['fold']!=fold]);sha=hashlib.sha256(train.tobytes()).hexdigest()
            for kind in ('J102','C205','B2_augmented_J','D2_expert'):
                directory=ROOT/f'artifacts/next5/references/fits/PRIMARY/full/{kind}/fold_{fold}';r=read_json(directory/'RESULTS.json')
                assert sha==r['training_ids_sha256'] and len(train)==r['training_series']
                assert r['research_split_sha256']==data.guard.split.digest and r['shadow_lock_sha256']==data.guard.shadow_lock_sha
                assert digest(directory/'model.joblib')==r['model_sha256']
                proof.append({'kind':kind,'fold':fold,'exact_training_series':len(train),'ids_numpy_dtype':str(train.dtype),'training_ids_sha256':sha,'model_sha256':r['model_sha256']})
        save('MATCHED_REFERENCE_AUDIT_NEXT6.json',{'status':'PASS','reference_models':proof,'same_original_grouped_fold_training_sets':True,
            'historical_full_DEV_score_not_used_as_gate_reference':True,'new_reference_fits':0})

if __name__=='__main__':main()
