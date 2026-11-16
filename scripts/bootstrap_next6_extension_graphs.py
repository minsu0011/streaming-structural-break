from pathlib import Path
import sys,os,argparse,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.bootstrap_graph import prepare,discovery
from src.next6.data import arrays
from src.next6.graph import pack_sample
from src.next6.common import ids_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args()
    lock=read_json(BASE/'EXTENSION_PROTOCOL_LOCK.json');replicates=lock['workstreams'][0]['training_graph_resamples']
    source_sha=digest(ROOT/'src/next6_extension/bootstrap_graph.py')
    frozen=BASE/'BOOTSTRAP_IMPLEMENTATION_LOCK.json'
    if not frozen.exists():save(frozen.name,{'source_sha256':source_sha,'created_utc':utc_now(),'internal_halves':'original fixed hash halves; independent group bootstrap within each half','replicates':replicates,'pvalues':'Not recomputed: diagnostic p-values did not select the original graph.'})
    assert read_json(frozen)['source_sha256']==source_sha
    data,meta,v=arrays()
    for fold in ([a.fold] if a.fold is not None else range(5)):
        directory=BASE/f'graph_bootstrap/fold_{fold}';directory.mkdir(parents=True,exist_ok=True)
        g=joblib.load(ROOT/f'artifacts/next6/fold_{fold}/graphs/outer_full/graph.joblib');tr=g.provenance['training_ids'];data.guard.admit(tr,purpose='NEXT6 extension training graph bootstrap only')
        with activity(f'bootstrap_training_graph_fold{fold}','RELATION_RESAMPLING'):
            sample=pack_sample(data,meta,v,tr);packed,halves,groups=prepare(sample,data.guard.split.groups);del sample
            if not (directory/'PARITY.json').exists():
                edges,detail=discovery(packed,np.ones(len(groups),dtype=int))
                assert [(e['source'],e['target'],e['lag'],e['selection_phase']) for e in edges]==[(e['source'],e['target'],e['lag'],e['selection_phase']) for e in g.selected]
                error=max(abs(a['selection_gain']-b['selection_gain']) for a,b in zip(edges,g.selected))
                assert error<1e-10,error
                save(str((directory/'PARITY.json').relative_to(BASE)),{'status':'PASS','max_gain_error':error,'training_ids_sha256':ids_hash(tr),'training_groups':len(groups),'original_graph_sha256':digest(ROOT/f'artifacts/next6/fold_{fold}/graphs/outer_full/graph.joblib')})
            for b in range(replicates):
                dest=directory/f'replicate_{b:03}.json'
                if dest.exists():continue
                rng=np.random.default_rng(np.random.SeedSequence([2026091301,fold,b]));m=np.zeros(len(groups),dtype=int)
                for half in (0,1):
                    ix=np.flatnonzero(halves==half);m[ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)))
                start=time.perf_counter();edges,detail=discovery(packed,m)
                save(str(dest.relative_to(BASE)),{'replicate':b,'fold':fold,'selected':edges,**detail,'unique_resampled_groups':int(np.count_nonzero(m)),'wall_seconds':time.perf_counter()-start,'training_only':True,'multiplicity_sha256':__import__('hashlib').sha256(m.tobytes()).hexdigest()})
                checkpoint('graph resampling',f'fold {fold}, {b+1}/{replicates}, edges {len(edges)}, seconds {time.perf_counter()-start:.1f}')
            save(str((directory/'DONE.json').relative_to(BASE)),{'status':'COMPLETE','replicates':replicates,'training_only':True,'created_utc':utc_now()})
    data.guard.check_state()
if __name__=='__main__':main()
