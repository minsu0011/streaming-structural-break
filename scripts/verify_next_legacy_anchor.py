"""Verify the new legacy-bank adapter against all fixed-screen ARCH8 features."""
from pathlib import Path
import sys
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next.extensions import build_extension_cache,extension_hash
from src.next.cache import load_design
from src.next.design_selection import select_reference_families
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    data = prepare_dev(ROOT)
    extension = {'kind':'legacy_bank','settings':{'order':8,'normalization':'arch1'}}
    path = ROOT/'artifacts/next/LEGACY_BANK_ANCHOR_PARITY.json'
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['extension_hash']!=extension_hash(extension):
            raise RuntimeError('Legacy adapter source changed')
        for name,digest in result['files_sha256'].items():
            if sha256(ROOT/name)!=digest:
                raise RuntimeError('Legacy anchor cache changed')
        print('LEGACY ANCHOR PARITY retained PASS')
        return
    directory = build_extension_cache(data,extension,'screen')
    actual = np.load(directory/'features.npy',mmap_mode='r',allow_pickle=False)
    rows = np.load(directory/'global_row_index.npy',allow_pickle=False)
    expected,metadata,names,sources,base_rows = load_design(data,scope='screen',groups=(),base='arch8')
    expected,names = select_reference_families(expected,names,'ABCD')
    np.testing.assert_array_equal(rows,base_rows)
    for start in range(0,len(rows),100000):
        np.testing.assert_array_equal(actual[start:start+100000],expected[start:start+100000])
    files = [directory/'features.npy',directory/'global_row_index.npy',directory/'MANIFEST.json']
    write_json(path,{'status':'PASS','created_utc':utc_now(),'extension_hash':extension_hash(extension),
        'rows':len(rows),'features':len(names),'series':len(np.unique(metadata['dataset_id'])),
        'float32_features_exact':True,'additional_model_fits':0,'score_evaluations':0,
        'interpretation':'Engineering-only adapter parity. Existing R013 scores and E001 full-OOF exact refit remain the reference; no duplicate candidate selection.',
        'files_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in files},'reference_sources':sources,
        'seal_rows':0,'reduced_new_usage':0})
    print('LEGACY ANCHOR PARITY PASS',len(rows),'rows',len(names),'features')


if __name__=='__main__':
    main()
