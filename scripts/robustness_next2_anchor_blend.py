"""Reuse matched anchor/complement stress models; compare every result to J102."""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next2.blend import combine, source_hash
from src.scoring.ts_auc import ts_auc
from src.next2.session import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now

J = 'J102_SERIAL_VARIANCE_512'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    path = ROOT/f'artifacts/next2/experiments/{args.candidate}/RESULTS.json'
    original = json.loads(path.read_text())
    assert original['status'] == 'FINALIST_DIAGNOSTICS_PENDING'
    blend = original['candidate']['blend']
    assert blend['implementation_sha256'] == source_hash()
    assert original['candidate']['quality_reference'] == J
    data = prepare_dev(ROOT)
    metadata = data.rows()
    outroot = ROOT/'artifacts/next2/robustness'/args.candidate
    summaries = []
    for split in ('ALT_1','ALT_2','ALT_3','LEAVE_ONE_H_REGIME_OUT'):
        out = outroot/split
        out.mkdir(parents=True,exist_ok=True)
        scores, references, anchors = [], [], []
        for fold in range(5):
            directories = [ROOT/'artifacts/next/robustness/candidates'/blend['primary']/split,
                ROOT/'artifacts/next2/robustness'/blend['complement']/split,
                ROOT/'artifacts/next/robustness/candidates'/J/split]
            arrays, receipts, coordinates = [], {}, None
            for directory in directories:
                receipt = directory/f'fold_{fold}.json'
                r = json.loads(receipt.read_text())
                assert r['status'] == 'PASS'
                for name,digest in r['files_sha256'].items():
                    assert sha256(directory/name) == digest
                rows = np.load(directory/f'fold_{fold}_global_rows.npy',allow_pickle=False)
                if coordinates is None:
                    coordinates = rows
                else:
                    np.testing.assert_array_equal(coordinates, rows)
                arrays.append(np.load(directory/f'fold_{fold}.npy',allow_pickle=False))
                receipts[receipt.relative_to(ROOT).as_posix()] = sha256(receipt)
            data.guard.admit(np.unique(metadata['dataset_id'][coordinates]),purpose='Frozen robust-anchor stress comparison')
            p = combine(arrays[0],arrays[1],blend['primary_weight'])
            y,t = metadata['target'][coordinates],metadata['time_online'][coordinates]
            value, reference, anchor = [ts_auc(y,v,t) for v in (p,arrays[2],arrays[0])]
            np.save(out/f'fold_{fold}.npy',p,allow_pickle=False)
            np.save(out/f'fold_{fold}_global_rows.npy',coordinates,allow_pickle=False)
            write_json(out/f'fold_{fold}.json',{'status':'PASS','candidate':args.candidate,'split_name':split,
                'fold':fold,'ts_auc':value,'reference_ts_auc':reference,'reference':J,'anchor_ts_auc':anchor,
                'delta':value-reference,'delta_vs_anchor':value-anchor,'new_fits':0,'seal_rows':0,
                'component_sources_sha256':receipts,'files_sha256':{name:sha256(out/name) for name in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy')}})
            scores.append(value); references.append(reference); anchors.append(anchor)
        summary = {'status':'PASS','split_name':split,'fold_scores':scores,'reference_fold_scores':references,
            'reference':J,'anchor_fold_scores':anchors,'mean':float(np.mean(scores)),'median':float(np.median(scores)),
            'worst':min(scores),'mean_delta':float(np.mean(scores)-np.mean(references)),
            'worst_delta':min(scores)-min(references),'minimum_matched_fold_delta':min(a-b for a,b in zip(scores,references)),
            'mean_delta_vs_anchor':float(np.mean(scores)-np.mean(anchors))}
        write_json(out/'RESULTS.json',summary)
        summaries.append(summary)
        print('NEXT2 ANCHOR STRESS',args.candidate,split,summary['mean_delta'],flush=True)
    write_json(outroot/'RESULTS.json',{'status':'PASS','candidate':args.candidate,'created_utc':utc_now(),
        'summaries':summaries,'completed_evaluations':20,'new_fits':0,'reference':J,'seal_rows':0,
        'nomination_sha256':sha256(path),'combination_sha256':source_hash(),'script_sha256':sha256(Path(__file__))})
    checkpoint('Robust-anchor matched ALT/H '+args.candidate,['Consolidate fixed performance gates','Retain synthetic tradeoffs'])


if __name__=='__main__':
    main()
