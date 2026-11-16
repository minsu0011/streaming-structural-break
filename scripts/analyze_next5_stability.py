"""Conditional ALT/H uncertainty using original PRIMARY-stratified draws."""
from pathlib import Path
import argparse
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from numba import set_num_threads
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.data import prepare_research,admitted_npz
from src.next5.stability import assignments,heldout_rows,frozen_primary_counts
from src.next5.bootstrap import PreparedAUC,paired_summary
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True)
    parser.add_argument('--split',choices=('ALT_A','ALT_B','H_EXCLUSION'),required=True);args=parser.parse_args();cid=args.candidate
    policy=read_json(BASE/'stability/POLICY_LOCK.json')
    for name,sha in policy['execution_sources_sha256'].items():assert digest(ROOT/name)==sha
    assert cid in policy['diagnostic_nominees'];source=BASE/'stability/candidates'/cid/args.split
    result=read_json(source/'RESULTS.json');assert result['status']=='COMPLETE'
    out=source/'BOOTSTRAP';out.mkdir(exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'candidate':cid,'split':args.split,'new_random_draws':False,
        'result_sha256':digest(source/'RESULTS.json'),'policy_sha256':digest(BASE/'stability/POLICY_LOCK.json')},immutable=True)
    set_num_threads(2);data=prepare_research();metadata=data.rows();mapping=assignments(args.split,data.guard)
    all_ids,all_counts,count_proof=frozen_primary_counts(data.guard);old_folds=[];new_folds=[];fold_reports=[];files={}
    for fold in range(5):
        _,valid=heldout_rows(metadata['dataset_id'],mapping,fold);directory=source/('fold_'+str(fold))
        receipt=read_json(directory/'RESULTS.json');path=directory/'predictions.npz';assert digest(path)==receipt['prediction_file_sha256']
        values=admitted_npz(path,data.guard,'Fixed held-out candidate paired uncertainty')
        np.testing.assert_array_equal(values['global_row'],valid);np.testing.assert_array_equal(values['dataset_id'],metadata['dataset_id'][valid])
        ids=np.unique(metadata['dataset_id'][valid]);columns=np.searchsorted(all_ids,ids);np.testing.assert_array_equal(all_ids[columns],ids)
        counts=all_counts[:,columns];code=np.searchsorted(ids,metadata['dataset_id'][valid]).astype(np.int32)
        y=metadata['target'][valid];times=metadata['time_online'][valid];ref=values['baseline'];new=values['prediction']
        refdir=BASE/'stability/reference_bootstrap'/args.split;refdir.mkdir(parents=True,exist_ok=True)
        path=refdir/('fold_'+str(fold)+'.npy');record_path=refdir/('fold_'+str(fold)+'.json')
        identity={'original_PRIMARY_count_hashes':count_proof,'heldout_ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
            'heldout_rows_sha256':hashlib.sha256(valid.tobytes()).hexdigest(),'baseline_prediction_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),
            'policy_sha256':digest(BASE/'stability/POLICY_LOCK.json')}
        if record_path.exists():
            previous=read_json(record_path);assert previous['identity']==identity and digest(path)==previous['sha256'];old=np.load(path,allow_pickle=False)
        else:
            if path.exists():raise RuntimeError('Preserve partial reference uncertainty for recovery review')
            old=PreparedAUC.create(y,ref,times,code,len(ids)).evaluate(counts);np.save(path,old,allow_pickle=False)
            atomic_json(record_path,{'created_utc':utc_now(),'identity':identity,'sha256':digest(path)},immutable=True)
        progress('Fixed held-out paired uncertainty',cid+' '+args.split+' fold '+str(fold)+'; existing multiplicities only')
        candidate=PreparedAUC.create(y,new,times,code,len(ids)).evaluate(counts)
        assert abs(old[0]-receipt['BL80_TS_AUC'])<1e-14 and abs(candidate[0]-receipt['TS_AUC'])<1e-14
        np.save(out/('fold_'+str(fold)+'.npy'),candidate,allow_pickle=False);files['fold_'+str(fold)+'.npy']=digest(out/('fold_'+str(fold)+'.npy'))
        positives=len(np.unique(metadata['dataset_id'][valid][y==1]));negative_only=len(ids)-positives
        fold_reports.append({'fold':fold,'series':len(ids),'positive_series':positives,'no_positive_series':negative_only,
            'supported':positives>=100 and negative_only>=100,'candidate_TS_AUC':float(candidate[0]),'BL80_TS_AUC':float(old[0]),
            **paired_summary(old[:,None],candidate[:,None])})
        old_folds.append(old);new_folds.append(candidate);print('NEXT5 STABILITY BOOTSTRAP',cid,args.split,fold,flush=True)
    old=np.asarray(old_folds).T;new=np.asarray(new_folds).T;np.savez_compressed(out/'AGGREGATE_DRAWS.npz',candidate_fold_scores=new,reference_fold_scores=old)
    files['AGGREGATE_DRAWS.npz']=digest(out/'AGGREGATE_DRAWS.npz')
    atomic_json(out/'RESULTS.json',{'created_utc':utc_now(),'status':'PASS','candidate':cid,'split':args.split,
        'paired_all_time':paired_summary(old,new),'fold_reports':fold_reports,'files_sha256':files,'new_fits':0,'new_random_draws':False,
        'original_PRIMARY_counts_sha256':count_proof,'bootstrap_design':'Original PRIMARY-fold-stratified whole-series resampling, restricted by ID to each new held-out fold without normalization. ALT/H fold totals vary.',
        'conditional_fixed_OOF_not_selection_adjusted':True,'diagnostic_refits_not_new_finalists':True,'shadow_rows':0,'seal_rows':0},immutable=True)
    data.guard.check_state();progress('Held-out paired uncertainty complete',cid+' '+args.split+'; primary failure status unchanged')


if __name__=='__main__':main()
