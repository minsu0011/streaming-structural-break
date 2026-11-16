"""Independently bind all 56 old inner fits without launching any learner."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,utc_now,progress
from src.next4.nested_reuse import load_outer,source_hash
from src.next.data import prepare_dev


def main():
    out=BASE/'nested_reuse_audit';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'timestamp':utc_now(),'source_sha256':source_hash(),'new_fits':0},immutable=True)
    data=prepare_dev(ROOT);records=[]
    for scope,folds in [('screen',[0,4]),('full',list(range(5)))]:
        for fold in folds:
            progress('Auditing retained honest inner OOF',scope+' outer '+str(fold)+'; coordinate, group, target, model and prediction hashes')
            *_,proof=load_outer(data,scope,fold)
            atomic_json(out/(scope+'_outer_'+str(fold)+'.json'),proof,immutable=True);records.append(proof)
            print('NEXT4 NESTED REUSE AUDIT',scope,fold,proof['inner_base_fits_reused'],flush=True)
    assert sum(r['inner_base_fits_reused'] for r in records)==56
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'inner_base_fits_available_for_reuse':56,
        'new_fits':0,'seal_rows':0,'outer_validation_in_meta_training':False,
        'source_sha256':source_hash(),'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    data.guard.check_state();progress('Honest nested OOF reuse audit passed','56 old base fits validated; NEXT4 may fit only small new meta learners on these exact inner OOF rows')


if __name__=='__main__':main()
