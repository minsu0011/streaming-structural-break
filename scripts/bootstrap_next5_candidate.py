"""Fixed paired whole-series bootstrap, exact original float64 prediction ties."""
from pathlib import Path
import argparse
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from numba import set_num_threads
from src.next5.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next5.data import prepare_research,admitted_npz
from src.next5.bootstrap import PreparedAUC,paired_summary
from src.next5.references import primary
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args();cid=args.candidate
    source=BASE/'experiments'/cid/'full';out=BASE/'bootstrap'/cid
    if (source/'SKIPPED.json').exists():
        atomic_json(out/'SKIPPED.json',{'created_utc':utc_now(),'candidate':cid,'reason':'No full candidate after failed screen','new_fits':0},immutable=True)
        return
    result=read_json(source/'RESULTS.json');assert result['status']=='COMPLETE' and result['scope']=='full'
    policy=read_json(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json');lock=read_json(BASE/'bootstrap/COUNTS_LOCK.json')
    for name,sha in lock['source_sha256'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError('Frozen paired bootstrap source changed')
    if out.exists():raise RuntimeError('Never rerun real bootstrap draws after inspecting uncertainty')
    out.mkdir(parents=True);atomic_json(out/'START.json',{'created_utc':utc_now(),'candidate':cid,
        'full_result_sha256':digest(source/'RESULTS.json'),'counts_lock_sha256':digest(BASE/'bootstrap/COUNTS_LOCK.json'),
        'source_sha256':digest(Path(__file__)),'new_fits':0},immutable=True)
    set_num_threads(2);data=prepare_research();metadata=data.rows();baseline,_,references=primary('full','PRIMARY',metadata,data.guard)
    candidate_draws=[];reference_draws=[];files={}
    for fold in range(5):
        directory=source/('fold_'+str(fold));receipt=read_json(directory/'RESULTS.json')
        path=directory/'predictions.npz'
        if digest(path)!=receipt['predictions_file_sha256']:raise RuntimeError('Candidate OOF bytes changed')
        prediction=admitted_npz(path,data.guard,'Paired ResearchPool OOF bootstrap');rows=prediction['global_row']
        np.testing.assert_array_equal(rows,np.flatnonzero(metadata['fold']==fold))
        np.testing.assert_array_equal(prediction['dataset_id'],metadata['dataset_id'][rows])
        np.testing.assert_array_equal(prediction['baseline'],baseline[rows])
        count_receipt=lock['folds'][fold];count_path=ROOT/count_receipt['path']
        if digest(count_path)!=count_receipt['sha256']:raise RuntimeError('Frozen real bootstrap counts changed')
        draws=admitted_npz(count_path,data.guard,'Frozen paired bootstrap counts')
        ids=metadata['dataset_id'][rows];code=np.searchsorted(draws['dataset_id'],ids).astype(np.int32)
        np.testing.assert_array_equal(draws['dataset_id'][code],ids);counts=draws['counts']
        target=metadata['target'][rows];times=metadata['time_online'][rows]
        refdir=BASE/'bootstrap/ResearchPool_BL80';refdir.mkdir(parents=True,exist_ok=True)
        refpath=refdir/('fold_'+str(fold)+'.npy');refreceipt=refdir/('fold_'+str(fold)+'.json')
        identity={'counts_sha256':count_receipt['sha256'],'reference_sources':references,
            'row_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),'prediction_dtype':'float64','fold':fold}
        if refpath.exists():
            prior=read_json(refreceipt)
            if prior['identity']!=identity or digest(refpath)!=prior['sha256']:raise RuntimeError('Reference paired draws identity mismatch')
            old=np.load(refpath,allow_pickle=False)
        else:
            progress('Matched BL80 paired bootstrap','ResearchPool fold '+str(fold)+'; frozen 2000 draws shared by all candidates')
            old=PreparedAUC.create(target,baseline[rows],times,code,counts.shape[1]).evaluate(counts)
            assert abs(old[0]-ts_auc(target,baseline[rows],times))<1e-14
            np.save(refpath,old,allow_pickle=False)
            atomic_json(refreceipt,{'created_utc':utc_now(),'identity':identity,'sha256':digest(refpath),'new_fits':0},immutable=True)
        progress('Candidate paired bootstrap',cid+' fold '+str(fold)+'; no new counts or post-score reseeding')
        new=PreparedAUC.create(target,prediction['prediction'],times,code,counts.shape[1]).evaluate(counts)
        assert abs(new[0]-receipt['TS_AUC'])<1e-14
        candidate_draws.append(new);reference_draws.append(old)
        path=out/('fold_'+str(fold)+'.npy');np.save(path,new,allow_pickle=False);files[path.name]=digest(path)
        print('NEXT5 BOOTSTRAP',cid,fold,flush=True)
    candidate=np.asarray(candidate_draws).T;reference=np.asarray(reference_draws).T;summary=paired_summary(reference,candidate)
    np.savez_compressed(out/'AGGREGATE_DRAWS.npz',candidate_fold_scores=candidate,reference_fold_scores=reference)
    files['AGGREGATE_DRAWS.npz']=digest(out/'AGGREGATE_DRAWS.npz');gate=policy['performance_gate']
    numeric=bool(result['mean_delta']>=gate['mean_delta_min'] and result['median_delta']>=gate['median_delta_min'] and result['worst_delta']>=gate['worst_delta_min'])
    admitted=bool(numeric and summary['ci_2_5']>gate['ci_lower_strict_min'] and summary['probability_delta_positive']>=gate['bootstrap_probability_min'])
    report={'created_utc':utc_now(),'status':'PASS','candidate':cid,'family':result['family'],'paired_all_time':summary,
        'performance_numeric_admission':numeric,'performance_primary_and_bootstrap_admission':admitted,
        'remaining_required':['ALT_A/ALT_B','H-regime exclusion','supported mechanism collapse','causality/prefix/determinism','runtime <=3x'] if admitted else [],
        'draws':2000,'folds':5,'new_fits':0,'shadow_rows':0,'seal_rows':0,'files_sha256':files,
        'full_result_sha256':digest(source/'RESULTS.json'),'counts_lock_sha256':digest(BASE/'bootstrap/COUNTS_LOCK.json'),
        'conditional_fixed_OOF_not_selection_adjusted':True,'promotion':False}
    atomic_json(out/'RESULTS.json',report,immutable=True);data.guard.check_state()
    progress('ResearchPool paired uncertainty complete',cid+' primary/uncertainty admission '+str(admitted))
    print('NEXT5 PAIRED RESULT',cid,summary['estimate_mean_delta'],summary['ci_2_5'],summary['ci_97_5'],summary['probability_delta_positive'],admitted,flush=True)


if __name__=='__main__':main()
