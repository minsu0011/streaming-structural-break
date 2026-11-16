"""Copy a CV fold artifact to a separate, hash-pinned inference contract fixture."""
from pathlib import Path
import argparse
import json
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next.candidate_io import load_full_oof
from src.next.registered_inference import implementation_hash
from src.next.log_energy_runtime import source_hash
from src.utils.artifacts import write_json,sha256,utc_now


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',default='Z001_LOG_ENERGY_CURRENT')
    parser.add_argument('--tag',default='v1')
    args = parser.parse_args()
    if not args.tag.replace('_','').isalnum():
        raise ValueError('Invalid fixture tag')
    data = prepare_dev(ROOT)
    candidate,_,_,_,sources = load_full_oof(data,args.candidate)
    out = ROOT/'artifacts/next/deployment/contracts'/args.candidate/args.tag
    out.mkdir(parents=True,exist_ok=True)
    model_path = ROOT/'artifacts/next/experiments'/args.candidate/'full/fold_0.joblib'
    descriptor = {'kind':'NEXT_REGISTERED_BINARY','runtime':'log_energy_fast_h','candidate':candidate,
        'model_sha256':sha256(model_path),'runtime_sha256':source_hash(),'inference_sha256':implementation_hash(),
        'status':'CV_FOLD_0_CONTRACT_FIXTURE_NOT_FINAL_SELECTION','oof_sources':sources,
        'seal_rows_fitted':0,'reduced_new_usage':0}
    if (out/'model.json').exists():
        old = json.loads((out/'model.json').read_text(encoding='utf-8'))
        if old!=descriptor or sha256(out/'model.joblib')!=descriptor['model_sha256']:
            raise RuntimeError('Contract fixture changed; retain it and use an explicit new revision')
    else:
        shutil.copy2(model_path,out/'model.joblib')
        write_json(out/'model.json',descriptor)
    print('NEXT CONTRACT FIXTURE',out,flush=True)
