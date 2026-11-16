"""Recombine matching independently fitted stress components; no meta refits."""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.blend import combine,source_hash
from src.scoring.ts_auc import ts_auc
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    args=parser.parse_args()
    path=ROOT/f'artifacts/next2/experiments/{args.candidate}/RESULTS.json'
    original=json.loads(path.read_text(encoding='utf-8'))
    if original['status']!='FINALIST_DIAGNOSTICS_PENDING':
        raise RuntimeError('Fixed full-numeric finalist admission required')
    blend=original['candidate']['blend']
    if blend['implementation_sha256']!=source_hash():
        raise RuntimeError('Blend implementation changed')
    compdir=ROOT/'artifacts/next2/robustness'/blend['complement']
    compreceipt=json.loads((compdir/'RESULTS.json').read_text(encoding='utf-8'))
    assert compreceipt['status']=='PASS' and compreceipt['completed_refits']==20
    data=prepare_dev(ROOT); metadata=data.rows()
    outroot=ROOT/'artifacts/next2/robustness'/args.candidate
    outroot.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for split in ('ALT_1','ALT_2','ALT_3','LEAVE_ONE_H_REGIME_OUT'):
        out=outroot/split
        out.mkdir(exist_ok=True)
        scores,references=[],[]
        sources={}
        for fold in range(5):
            dirs=[ROOT/'artifacts/next/robustness/candidates'/blend['primary']/split,compdir/split]
            values=[]; rows=None
            for directory in dirs:
                r=json.loads((directory/f'fold_{fold}.json').read_text(encoding='utf-8'))
                for name,digest in r['files_sha256'].items():
                    assert sha256(directory/name)==digest
                coordinate=np.load(directory/f'fold_{fold}_global_rows.npy',allow_pickle=False)
                if rows is None:
                    rows=coordinate
                else:
                    np.testing.assert_array_equal(rows,coordinate)
                data.guard.admit(np.unique(metadata['dataset_id'][rows]),purpose='fixed blend stress OOF')
                values.append(np.load(directory/f'fold_{fold}.npy',allow_pickle=False))
                sources[str(directory.relative_to(ROOT)/f'fold_{fold}.json')]=sha256(directory/f'fold_{fold}.json')
            p=combine(values[0],values[1],blend['primary_weight'])
            score=ts_auc(metadata['target'][rows],p,metadata['time_online'][rows])
            ref=ts_auc(metadata['target'][rows],values[0],metadata['time_online'][rows])
            np.save(out/f'fold_{fold}.npy',p,allow_pickle=False)
            np.save(out/f'fold_{fold}_global_rows.npy',rows,allow_pickle=False)
            files=[f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy']
            write_json(out/f'fold_{fold}.json',{'status':'PASS','candidate':args.candidate,
                'split_name':split,'fold':fold,'ts_auc':score,'reference_ts_auc':ref,'delta':score-ref,
                'component_sources_sha256':sources.copy(),'new_fits':0,'seal_rows':0,
                'files_sha256':{name:sha256(out/name) for name in files}})
            scores.append(score); references.append(ref)
        summary={'status':'PASS','split_name':split,'fold_scores':scores,'reference_fold_scores':references,
            'mean':float(np.mean(scores)),'median':float(np.median(scores)),'worst':min(scores),
            'mean_delta':float(np.mean(scores)-np.mean(references)),
            'worst_delta':min(scores)-min(references),
            'minimum_matched_fold_delta':min(a-b for a,b in zip(scores,references))}
        write_json(out/'RESULTS.json',summary)
        summaries.append(summary)
        print('NEXT2 BLEND STRESS',args.candidate,split,summary,flush=True)
    write_json(outroot/'RESULTS.json',{'status':'PASS','candidate':args.candidate,'created_utc':utc_now(),
        'summaries':summaries,'completed_evaluations':20,'new_fits':0,'component_fit_requirement':20,
        'nomination_sha256':sha256(path),'combination_sha256':source_hash(),'seal_rows':0})
    checkpoint('Blend alternate/regime validation '+args.candidate,['Engineering and synthetic behavior','Apply fixed gates'],
        artifacts=[(outroot/'RESULTS.json').relative_to(ROOT).as_posix()])


if __name__=='__main__':
    main()
