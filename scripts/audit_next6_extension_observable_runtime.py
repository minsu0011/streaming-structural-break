"""Audit the exported 27-column network and fused adapter without refitting."""
from pathlib import Path
import sys,os,pickle,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.observer import load,ObservableModel
from src.next6_extension.runtime import Prepared
from src.next6_extension.fresh_state import categorize
from src.next6.data import source

def main():
    lock={'created_utc':utc_now(),'adapter_sha256':digest(ROOT/'src/next6_extension/runtime.py'),'audit_sha256':digest(Path(__file__)),
        'purpose':'Fused scalar implementation of the already exported network; no fitted quantities or decisions changed.'}
    save('FUSED_OBSERVABLE_RUNTIME_LOCK.json',lock)
    model=load(BASE/'observable_models/fold_0/model.joblib');prepared=Prepared(model);rng=np.random.default_rng(991);h=rng.normal(size=2048);o=rng.normal(size=4096)
    prepared.new_state(h).predict_one(o[0]);measurements=[]
    for repeat in range(7):
        state=prepared.new_state(h);start=time.perf_counter()
        for x in o:state.predict_one(x)
        measurements.append((time.perf_counter()-start)*1e6/len(o))
    save('FUSED_OBSERVER_LATENCY_PILOT.json',{'created_utc':utc_now(),'median_microseconds_per_scalar':float(np.median(measurements)),'repeats':measurements,'points_per_repeat':len(o),'bounded_mutable_state_bytes':state.bounded_mutable_state_bytes,'concurrent_research_jobs_active':True,'final_clean_benchmark_required':True})
    print('Fused scalar pilot us:',np.median(measurements),flush=True)
    data=source();oldscore=np.load(BASE/'rp_replay/evidence.npy',mmap_mode='r');oldstates=np.load(BASE/'rp_replay/states.npy',mmap_mode='r');maxerror=0.;scalarpoints=0;series=0
    with activity('exported_observable_network_all_RP_audit','RUNTIME_DEBUG_AUDIT'):
        for fold in range(5):
            model=load(BASE/f'observable_models/fold_{fold}/model.joblib');prepared=Prepared(model);ids=data.ids[data.series['fold']==fold]
            chosen=set(sorted(map(int,ids),key=lambda s:hashlib.sha256(('NEXT6_OBSERVABLE_AUDIT:'+str(s)).encode()).hexdigest())[:20])
            for sid in ids:
                h,o,_=data.get(int(sid));index=data.id_to_index[int(sid)];a,b=data.series['online_offsets'][index:index+2]
                out=prepared.new_state(h).replay(o);reference=model.new_state(h).replay(o);error=float(np.max(np.abs(out-reference),initial=0));maxerror=max(maxerror,error);np.testing.assert_allclose(out,reference,rtol=1e-12,atol=1e-12)
                np.testing.assert_array_equal(out[:,1],oldscore[a:b]);np.testing.assert_array_equal(out[:,0],oldstates[a:b]);assert np.isfinite(out).all()
                previous=out[:-1,0].astype(int);current=out[1:,0].astype(int)
                np.testing.assert_array_equal(out[1:,20],model.macro_probabilities[0,previous,current]);np.testing.assert_array_equal(out[1:,21],model.macro_probabilities[1,previous,current]);assert out[0,24]==0 and np.all(out[1:,24]==1)
                if int(sid) in chosen:
                    state=prepared.new_state(h);other=prepared.new_state(h[::-1]);actual=[];size=state.bounded_mutable_state_bytes
                    for x in o:actual.append(state.predict_one(x));other.predict_one(-x)
                    np.testing.assert_array_equal(actual,out);assert state.bounded_mutable_state_bytes==size
                    at=min(117,len(o)//2);prefix=prepared.new_state(h);np.testing.assert_array_equal(prefix.replay(o[:at]),out[:at]);restored=pickle.loads(pickle.dumps(prefix));np.testing.assert_array_equal(restored.replay(o[at:]),out[at:])
                    altered=o.copy();altered[at:]=-altered[at:]*3;np.testing.assert_array_equal(prepared.new_state(h).replay(altered)[:at],out[:at]);scalarpoints+=len(o)
                series+=1
            checkpoint('fused observable network verified',f'fold {fold}, all1280 series,20 scalar/prefix/serialization audits')
    with activity('coherent_nonempty_network_manual_probability_audit','RUNTIME_DEBUG_AUDIT'):
        graph=joblib.load(BASE/'coherent_power/KNOWN_POSITIVE_GRAPH.joblib');assert graph.selected;graph.validate()
        positive=ObservableModel(model.threshold,graph,model.macro_probabilities.copy(),model.macro_support.copy(),{'synthetic_diagnostic_only':True},model.source_sha256.copy())
        prepared=Prepared(positive);rng=np.random.default_rng(931);h=rng.normal(size=2048);o=rng.normal(size=2048);state=prepared.new_state(h);previous=state.features.H_previous_state.copy();out=state.replay(o)
        for end in np.flatnonzero(out[:,4]):
            current=categorize(out[end,12:]);lf=ln=used=0
            for j,i in enumerate(graph.parents):
                if i<0:continue
                own,p,y=int(previous[j]),int(previous[i]),int(current[j]);lf+=np.log(graph.full[1,j,own,p,y]/graph.full[0,j,own,p,y]);ln+=np.log(graph.null[1,j,own,p,y]/graph.null[0,j,own,p,y]);used+=1
            np.testing.assert_allclose(out[end,7:10],[lf/used,ln/used,(lf-ln)/used],rtol=1e-12,atol=1e-12);previous=current
        state=prepared.new_state(h);np.testing.assert_array_equal([state.predict_one(x) for x in o],out);np.testing.assert_allclose(positive.new_state(h).replay(o),out,rtol=1e-12,atol=1e-12)
    data.guard.check_state()
    save('FUSED_OBSERVABLE_RUNTIME_PARITY.json',{'status':'PASS','created_utc':utc_now(),'all_RP_series':series,'all_RP_points':len(oldscore),'max_readable_reference_error':maxerror,'macrostate_and_evidence_unchanged':True,'both_macro_models_manual_lookup':'PASS','extra_scalar_points':scalarpoints,'extra_scalar_series':100,'prefix_suffix_serialization_interleaving':'PASS','known_nonempty_coherent_graph_manual_lookup':'PASS','fused_state_bytes':state.bounded_mutable_state_bytes,'runtime_sha256':lock['adapter_sha256']})
if __name__=='__main__':main()
