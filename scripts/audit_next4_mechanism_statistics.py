"""Recompute fixed supported-mechanism selection evidence with original counts."""
import argparse
import hashlib
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.frozen import fold_predictions
from src.next4.diagnostics import evaluation_masks,scale_directions
from src.next.data import prepare_dev
from src.next.bootstrap import PreparedAUC,paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from src.scoring.ts_auc import ts_auc

CANDIDATES=('A2_MAGNITUDE_PORTMANTEAU','B1_SPLIT_ENERGY_BRANCHES','B2_LOG_ENERGY_SEMITAIL',
            'C3_SHAPE_RATIO_DISTRIBUTION','D2_DISCRETE_SCALE_MIXTURE','D3_HAZARD_MIXTURE')
SCOPES=tuple('mechanism_'+s for s in ('SCALE_DOMINANT','DEPENDENCE_DOMINANT','MIXED','WEAK_OR_AMBIGUOUS','observed_scale_increase'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args();assert re.fullmatch('[A-Za-z0-9_]+',args.tag)
    session=read_json(BASE/'SESSION.json');assert session.get('audit_started_utc')
    out=BASE/'final_audit'/args.tag/'mechanism_statistics';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'candidate_ids':CANDIDATES,'scopes':SCOPES,'source_sha256':digest(Path(__file__))},immutable=True)
    set_num_threads(2);data=prepare_dev(ROOT);metadata=data.rows()
    path=ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    assert digest(path)==read_json(path.parent/'REGIME_ASSIGNMENT_LOCK.json')['files_sha256'][path.name]
    series=pd.read_parquet(path).set_index('dataset_id');data.guard.admit(series.index.to_numpy(),purpose='Final DEV-only mechanism audit')
    directions,direction_report=scale_directions(data,series);references={s:[] for s in SCOPES}
    candidates={cid:{s:[] for s in SCOPES} for cid in CANDIDATES};count_proofs=[];maximum_error=0.;evidence={}
    for fold in range(5):
        rows=np.flatnonzero(metadata['fold']==fold);ids,y,t=(metadata[k][rows] for k in ('dataset_id','target','time_online'))
        code,counts=draw_group_multiplicities(ids,2000,20260908+fold)
        assert hashlib.sha256(counts.tobytes()).hexdigest()==read_json(BASE/'bootstrap/frozen_BL80'/f'fold_{fold}.json')['identity']['counts_sha256']
        baseline,_=fold_predictions(data,fold,'full',rows);masks=evaluation_masks(series.reindex(ids),y,t,directions)
        count_proofs.append({'fold':fold,'multiplicities_sha256':hashlib.sha256(counts.tobytes()).hexdigest()})
        for scope in SCOPES:
            mask=masks[scope];filename=f'fold_{fold}_{scope}.npy';prior=BASE/'forensic/frozen_BL80'/filename
            receipt=read_json(prior.with_name(filename+'.json'));assert digest(prior)==receipt['sha256']
            assert receipt['identity']['rows_sha256']==hashlib.sha256(rows[mask].tobytes()).hexdigest()
            values=PreparedAUC.create(y[mask],baseline['BL80'][mask],t[mask],code[mask],counts.shape[1]).evaluate(counts)
            np.testing.assert_array_equal(values,np.load(prior,allow_pickle=False));references[scope].append(values)
            for cid in CANDIDATES:
                source=BASE/'experiments'/cid/'full';r=read_json(source/f'fold_{fold}.json');pp=source/f'fold_{fold}.npy'
                assert digest(pp)==r['files_sha256'][pp.name]
                coord=source/f'fold_{fold}_global_rows.npy';assert digest(coord)==r['files_sha256'][coord.name]
                np.testing.assert_array_equal(np.load(coord,allow_pickle=False),rows)
                prediction=np.load(pp,allow_pickle=False);values=PreparedAUC.create(y[mask],prediction[mask],t[mask],code[mask],counts.shape[1]).evaluate(counts)
                oldpath=BASE/'forensic'/cid/filename;old=read_json(oldpath.parent/'RESULTS.json')
                assert digest(oldpath)==old['files_sha256'][filename]
                np.testing.assert_array_equal(values,np.load(oldpath,allow_pickle=False));candidates[cid][scope].append(values)
                error=abs(float(values[0])-ts_auc(y[mask],prediction[mask],t[mask]));assert error<=1e-14;maximum_error=max(maximum_error,error)
                evidence[(oldpath.parent/'RESULTS.json').relative_to(ROOT).as_posix()]=digest(oldpath.parent/'RESULTS.json')
            progress('Final audit supported mechanism statistics',f'Fold {fold} {scope}: unchanged masks and original counts, six original candidates exact',phase='FINAL_AUDIT',counts_as_research=False)
            print('NEXT4 AUDIT MECHANISM',fold,scope,'EXACT',flush=True)
    summaries={}
    for cid in CANDIDATES:
        summaries[cid]={};frame=pd.read_csv(BASE/'forensic'/cid/'PAIRED_FRACTURES.csv')
        for scope in SCOPES:
            summary=paired_summary(np.asarray(references[scope]).T,np.asarray(candidates[cid][scope]).T)
            if scope=='mechanism_SCALE_DOMINANT':
                old=read_json(BASE/'forensic'/cid/'RESULTS.json')['paired_scale']
                assert summary=={k:old[k] for k in summary}
            previous=frame[frame.scope==scope].iloc[0]
            for key,value in summary.items():assert abs(float(value)-float(previous[key]))<=1e-14
            summaries[cid][scope]=summary
    data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'candidate_ids':CANDIDATES,'scopes':SCOPES,'fold_count':5,'original_replicates':2000,
        'all_original_candidate_and_reference_arrays_bitwise_equal':True,'all_original_SCALE_summary_fields_exact':True,
        'official_scorer_max_abs_difference':maximum_error,'direction_reconstruction':direction_report,
        'candidate_summaries':summaries,'original_multiplicity_proofs':count_proofs,
        'new_fits':0,'new_random_counts_or_seeds':0,'seal_rows':0,'evidence_sha256':evidence,
        'source_sha256':digest(Path(__file__))},immutable=True)
    print('NEXT4 FINAL MECHANISM AUDIT COMPLETE',flush=True)


if __name__=='__main__':main()
