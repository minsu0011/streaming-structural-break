"""Freshly recompute the original fixed paired statistics without extra draws."""
import argparse
import hashlib
from pathlib import Path
import re
import sys
import numpy as np
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.frozen import fold_predictions
from src.next.data import prepare_dev
from src.next.bootstrap import PreparedAUC,paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from src.scoring.ts_auc import ts_auc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--candidates',nargs='+',default=['B2_LOG_ENERGY_SEMITAIL']);args=parser.parse_args()
    if any(not re.fullmatch('[A-Za-z0-9_]+',s) for s in [args.tag,*args.candidates]):raise ValueError('Safe existing identifiers required')
    session=read_json(BASE/'SESSION.json');assert session.get('audit_started_utc')
    pp=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(pp)
    out=BASE/'final_audit'/args.tag/'primary_statistics';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'candidates':args.candidates,'policy_sha256':digest(pp),'source_sha256':digest(Path(__file__))},immutable=True)
    set_num_threads(2);data=prepare_dev(ROOT);meta=data.rows();evidence={};reference_values=[]
    candidate_values={cid:[] for cid in args.candidates};oracle_differences=[];count_proofs=[]
    for fold in range(5):
        rows=np.flatnonzero(meta['fold']==fold);ids,y,t=(meta[k][rows] for k in ('dataset_id','target','time_online'))
        code,counts=draw_group_multiplicities(ids,policy['bootstrap']['draws'],policy['bootstrap']['seed']+fold)
        old_identity_path=BASE/'bootstrap/frozen_BL80'/f'fold_{fold}.json';old_identity=read_json(old_identity_path)
        assert old_identity['identity']['counts_sha256']==hashlib.sha256(counts.tobytes()).hexdigest()
        assert old_identity['identity']['rows_sha256']==hashlib.sha256(rows.tobytes()).hexdigest()
        assert old_identity['identity']['policy_sha256']==digest(pp)
        baseline,proof=fold_predictions(data,fold,'full',rows)
        progress('Final audit original paired reference','Recompute fold '+str(fold)+' with exactly the existing 2000 multiplicities',phase='FINAL_AUDIT',counts_as_research=False)
        values=PreparedAUC.create(y,baseline['BL80'],t,code,counts.shape[1]).evaluate(counts)
        old_path=BASE/'bootstrap/frozen_BL80'/f'fold_{fold}.npy'
        assert digest(old_path)==old_identity['sha256']
        np.testing.assert_array_equal(values,np.load(old_path,allow_pickle=False))
        score=ts_auc(y,baseline['BL80'],t);error=abs(float(values[0])-score);assert error<=1e-14
        oracle_differences.append(error);reference_values.append(values)
        evidence[old_identity_path.relative_to(ROOT).as_posix()]=digest(old_identity_path)
        count_proofs.append({'fold':fold,'whole_series_groups':counts.shape[1],
            'replicates':len(counts)-1,'counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),
            'reference_arrays_bitwise_equal':True,'frozen_OOF_sources':proof})
        for cid in args.candidates:
            source=BASE/'experiments'/cid/'full';receipt_path=source/f'fold_{fold}.json';receipt=read_json(receipt_path)
            for name in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy'):
                assert digest(source/name)==receipt['files_sha256'][name]
            np.testing.assert_array_equal(np.load(source/f'fold_{fold}_global_rows.npy',allow_pickle=False),rows)
            prediction=np.load(source/f'fold_{fold}.npy',allow_pickle=False)
            values=PreparedAUC.create(y,prediction,t,code,counts.shape[1]).evaluate(counts)
            prior_path=BASE/'bootstrap'/cid/f'fold_{fold}.npy'
            b=read_json(BASE/'bootstrap'/cid/'RESULTS.json');assert digest(prior_path)==b['files_sha256'][prior_path.name]
            np.testing.assert_array_equal(values,np.load(prior_path,allow_pickle=False))
            score=ts_auc(y,prediction,t);error=abs(float(values[0])-score);assert error<=1e-14
            assert abs(score-receipt['ts_auc'])<=1e-14
            oracle_differences.append(error);candidate_values[cid].append(values)
            evidence[receipt_path.relative_to(ROOT).as_posix()]=digest(receipt_path)
            print('NEXT4 AUDIT ORIGINAL PAIRED DRAW PARITY',cid,fold,flush=True)
    reference=np.asarray(reference_values).T;summaries={}
    for cid,values in candidate_values.items():
        candidate=np.asarray(values).T;summary=paired_summary(reference,candidate)
        path=BASE/'bootstrap'/cid/'RESULTS.json';previous=read_json(path)
        assert summary==previous['paired_all_time']
        evidence[path.relative_to(ROOT).as_posix()]=digest(path);summaries[cid]=summary
        np.savez_compressed(out/(cid+'_PAIRED_DRAWS.npz'),reference=reference,candidate=candidate)
    data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),
        'audit_started_utc':session['audit_started_utc'],'candidate_summaries':summaries,
        'all_original_reference_and_candidate_draw_arrays_bitwise_equal':True,
        'original_counts_regenerated_and_hash_matched':True,'new_or_additional_bootstrap_draws':0,
        'seed_changed':False,'original_replicates_per_fold':2000,'fold_count':5,
        'maximum_official_scorer_difference':max(oracle_differences),
        'new_fits':0,'seal_rows':0,'count_proofs':count_proofs,
        'source_sha256':digest(Path(__file__)),'evidence_sha256':evidence,
        'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    print('NEXT4 FINAL ORIGINAL PRIMARY STATISTICS AUDIT PASS',flush=True)


if __name__=='__main__':main()
