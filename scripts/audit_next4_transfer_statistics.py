"""Recheck existing ALT/H overall and SCALE statistics with original counts."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.robustness import partitions,frozen_predictions
from src.next.data import prepare_dev
from src.next.bootstrap import PreparedAUC,paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from src.scoring.ts_auc import ts_auc

CANDIDATES=('B1_SPLIT_ENERGY_BRANCHES','B2_LOG_ENERGY_SEMITAIL','D2_DISCRETE_SCALE_MIXTURE')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    assert args.tag.replace('_','').isalnum()
    session=read_json(BASE/'SESSION.json');assert session.get('audit_started_utc')
    out=BASE/'final_audit'/args.tag/'transfer_statistics';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'candidates':CANDIDATES,'scopes':['all_time','SCALE'],'source_sha256':digest(Path(__file__))},immutable=True)
    set_num_threads(2);data=prepare_dev(ROOT);metadata=data.rows();splits,split_proof=partitions(data,metadata)
    path=ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    assert digest(path)==read_json(path.parent/'REGIME_ASSIGNMENT_LOCK.json')['files_sha256'][path.name]
    series=pd.read_parquet(path).set_index('dataset_id');data.guard.admit(series.index.to_numpy(),purpose='Final fixed ALT/H statistics audit')
    evidence={};summaries=[];count_proofs=[];maximum_error=0.
    for split,folds in splits.items():
        reference={scope:[] for scope in ('all_time','SCALE')}
        values={cid:{scope:[] for scope in reference} for cid in CANDIDATES}
        for fold,(_,rows) in enumerate(folds):
            baseline,proof=frozen_predictions(data,split,fold,rows);evidence.update(proof)
            ids,y,t=(metadata[k][rows] for k in ('dataset_id','target','time_online'))
            code,counts=draw_group_multiplicities(ids,2000,20260908+fold)
            frame=series.reindex(ids);mask=(frame.signature.to_numpy()=='SCALE_DOMINANT')|(frame.tau.to_numpy()<0)
            refdir=BASE/'robustness_forensic/frozen_BL80'/split;refpath=refdir/f'fold_{fold}_SCALE.npy'
            refreceipt=read_json(refpath.with_name(refpath.name+'.json'))
            counts_sha=hashlib.sha256(counts.tobytes()).hexdigest()
            assert refreceipt['identity']['counts_sha256']==counts_sha
            assert refreceipt['identity']['rows_sha256']==hashlib.sha256(rows[mask].tobytes()).hexdigest()
            assert digest(refpath)==refreceipt['sha256']
            count_proofs.append({'split':split,'fold':fold,'counts_sha256':counts_sha})
            a=PreparedAUC.create(y,baseline['BL80'],t,code,counts.shape[1]).evaluate(counts)
            sa=PreparedAUC.create(y[mask],baseline['BL80'][mask],t[mask],code[mask],counts.shape[1]).evaluate(counts)
            np.testing.assert_array_equal(sa,np.load(refpath,allow_pickle=False))
            reference['all_time'].append(a);reference['SCALE'].append(sa)
            for cid in CANDIDATES:
                directory=BASE/'robustness'/cid/split;rp=directory/f'fold_{fold}.json';receipt=read_json(rp)
                evidence[rp.relative_to(ROOT).as_posix()]=digest(rp)
                for suffix in ('.npy','_global_rows.npy','_bootstrap.npy','_reference_bootstrap.npy'):
                    name=f'fold_{fold}'+suffix;assert digest(directory/name)==receipt['files_sha256'][name]
                np.testing.assert_array_equal(np.load(directory/f'fold_{fold}_global_rows.npy',allow_pickle=False),rows)
                np.testing.assert_array_equal(a,np.load(directory/f'fold_{fold}_reference_bootstrap.npy',allow_pickle=False))
                prediction=np.load(directory/f'fold_{fold}.npy',allow_pickle=False)
                b=PreparedAUC.create(y,prediction,t,code,counts.shape[1]).evaluate(counts)
                sb=PreparedAUC.create(y[mask],prediction[mask],t[mask],code[mask],counts.shape[1]).evaluate(counts)
                np.testing.assert_array_equal(b,np.load(directory/f'fold_{fold}_bootstrap.npy',allow_pickle=False))
                sf=BASE/'robustness_forensic'/cid;name=f'{split}_fold_{fold}_SCALE.npy';sr=read_json(sf/'RESULTS.json')
                assert digest(sf/name)==sr['files_sha256'][name]
                np.testing.assert_array_equal(sb,np.load(sf/name,allow_pickle=False))
                for score,want in ((ts_auc(y,prediction,t),receipt['ts_auc']),(ts_auc(y[mask],prediction[mask],t[mask]),sb[0])):
                    error=abs(score-want);assert error<=1e-14;maximum_error=max(maximum_error,error)
                values[cid]['all_time'].append(b);values[cid]['SCALE'].append(sb)
                evidence[(sf/'RESULTS.json').relative_to(ROOT).as_posix()]=digest(sf/'RESULTS.json')
            progress('Final audit original ALT/H statistics',split+' fold '+str(fold)+': three existing candidates, original counts and overall/SCALE arrays exact',phase='FINAL_AUDIT',counts_as_research=False)
            print('NEXT4 AUDIT ALT H',split,fold,'EXACT',flush=True)
        for cid in CANDIDATES:
            old=read_json(BASE/'robustness'/cid/split/'RESULTS.json')
            old_scale=next(r for r in read_json(BASE/'robustness_forensic'/cid/'RESULTS.json')['summaries'] if r['split']==split and r['scope']=='SCALE')
            for scope in reference:
                result=paired_summary(np.asarray(reference[scope]).T,np.asarray(values[cid][scope]).T)
                expected=old['paired_all_time'] if scope=='all_time' else {k:old_scale[k] for k in result}
                assert result==expected;summaries.append({'candidate':cid,'split':split,'scope':scope,**result})
    data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'candidates':CANDIDATES,'partitions':list(splits),'folds_per_partition':5,'scopes':['all_time','SCALE'],
        'all_original_arrays_bitwise_equal':True,'all_original_summary_fields_exact':True,
        'partition_rows_independently_reconstructed':True,'maximum_official_scorer_difference':maximum_error,
        'new_random_counts_or_seeds':0,'new_fits':0,'seal_rows':0,'split_proof':split_proof,
        'original_count_proofs':count_proofs,'summaries':summaries,'evidence_sha256':evidence,
        'source_sha256':digest(Path(__file__))},immutable=True)
    print('NEXT4 FINAL ALT H STATISTICS AUDIT PASS',flush=True)


if __name__=='__main__':main()
