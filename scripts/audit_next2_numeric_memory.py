"""Retained numerical allocation ownership, and real-prefix state growth probes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.io import model_class
from src.next2.canonical_model import load
from src.next2.pruned_runtime import PreparedCompactPredictor
from src.next2.used_feature_runtime import PreparedUsedPredictor
from src.next2.runtime import prepare
from src.next.serial_variance_runtime import PreparedSerialVariancePredictor
from src.next.fast_variance_initialization import PreparedFastVariancePredictor
from src.next2.numeric_memory import accounting
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    data=prepare_dev(ROOT)
    lengths=np.diff(data.series['historical_offsets'])
    ordered=np.argsort(lengths,kind='stable')
    indices=[int(ordered[k]) for k in (0,len(ordered)//2,len(ordered)-1)]
    selected=[int(data.ids[k]) for k in indices]
    data.guard.admit(selected,purpose='Numerical state accounting: shortest/median/longest DEV H')
    protocol=json.loads((ROOT/'configs/next2/RUNTIME_MEASUREMENT_LOCK.json').read_text())
    out=ROOT/'artifacts/next2/numeric_memory';out.mkdir(parents=True,exist_ok=True)
    identity={'script_sha256':sha256(Path(__file__)),'accounting_source_sha256':sha256(ROOT/'src/next2/numeric_memory.py'),
        'model_menu_sha256':sha256(ROOT/'configs/next2/RUNTIME_MEASUREMENT_LOCK.json'),'series_ids':selected,'growth_probe_points':16384}
    if (out/'POLICY.json').exists():assert json.loads((out/'POLICY.json').read_text())==identity
    else:write_json(out/'POLICY.json',identity)
    constructors={'original_serial':PreparedSerialVariancePredictor,'original_variance':PreparedFastVariancePredictor,
        'compact':PreparedCompactPredictor,'tree_used':PreparedUsedPredictor,'prepared':prepare}
    rows=[];allocations=[]
    for variant,recipe in protocol['variants'].items():
        path=ROOT/recipe['model_path'];assert sha256(path)==recipe['model_sha256']
        model=load(path) if recipe['model_format']=='NEXT2_CANONICAL_V1' else model_class(recipe['candidate']).load(path)
        engine=constructors[recipe['runtime']](model)
        for sid in selected:
            h,o,_=data.get(sid);state=engine.new_state(h)
            before,owners,private=accounting(state)
            snapshots={key:r['owner'].copy() for key,r in owners.items()}
            for point in np.resize(o,16384):state.predict_one(point)
            after,after_owners,after_private=accounting(state)
            assert before==after and set(owners)==set(after_owners) and private==after_private
            changed={key for key,r in owners.items() if not np.array_equal(snapshots[key],r['owner'],equal_nan=True)}
            rows.append({'variant':variant,'candidate':recipe['candidate_id'],'dataset_id':sid,'historical_points':len(h),
                **before,'legacy_state_array_bytes_property':state.state_array_bytes,
                'observed_changed_private_allocation_bytes':sum(owners[k]['bytes'] for k in private&changed),
                'private_unchanged_in_this_probe_bytes':sum(owners[k]['bytes'] for k in private-changed),
                'growth_probe_points':16384,'retained_numeric_growth_bytes':0,'model_file_bytes':path.stat().st_size})
            for key,r in owners.items():
                allocations.append({'variant':variant,'dataset_id':sid,'bytes':r['bytes'],'dtype':r['dtype'],
                    'shape':str(r['shape']),'paths':';'.join(r['paths']),'private_per_series':key in private,
                    'changed_during_probe':key in changed,'numpy_owns_data':r['numpy_owns_data']})
        print('NEXT2 NUMERIC MEMORY',variant,before['private_per_series_numeric_bytes'],flush=True)
    pd.DataFrame(rows).to_csv(out/'MEMORY_BY_H.csv',index=False)
    pd.DataFrame(allocations).to_csv(out/'ALLOCATION_INVENTORY.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'identity':identity,'variants':len(protocol['variants']),
        'historical_size_probes':len(selected),'seal_rows':0,'new_fits':0,
        'files_sha256':{name:sha256(out/name) for name in ('MEMORY_BY_H.csv','ALLOCATION_INVENTORY.csv')},
        'boundary':'Retained NumPy numerical allocations, deduplicated by ultimate NumPy owner address and size; includes Numba-held shared buffers, excludes Python object headers, code/JIT and allocator overhead. Process RSS belongs to isolated latency results.',
        'mutation_classification':'Observed changed allocation over a 16,384-point probe; unchanged does not prove a semantic constant. Private vs model-shared uses allocation ownership, not observed mutation.',
        'legacy_property':'Older state_array_bytes included shared selector arrays and may count multiple retained NumPy views by object identity; reported alongside owner accounting.'})
    checkpoint('Audited private and shared numerical state ownership',['Isolated direct 10M growth check','Finalize state reduction accounting'],
        artifacts=['artifacts/next2/numeric_memory/RESULTS.json'])


if __name__=='__main__':
    main()
