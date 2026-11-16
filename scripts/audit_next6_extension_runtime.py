from pathlib import Path
import sys,os,pickle,warnings,subprocess,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.fresh_runtime import SafeState
from src.next6_extension.fresh_state import State,categorize
from src.next6.data import source

def main():
    data=source();cut=read_json(BASE/'calibration/fresh/DONE.json')['thresholds']['FRESH'];oldscore=np.load(BASE/'rp_replay/evidence.npy',mmap_mode='r');oldstates=np.load(BASE/'rp_replay/states.npy',mmap_mode='r');maxerror=0.;nscalar=0;proof=[]
    assert digest(ROOT/'src/next6_extension/fresh_runtime.py')==read_json(BASE/'SAFE_H_RUNTIME_REPAIR.json')['adapter_sha256']
    with activity('safe_H_adapter_all_RP_parity','RUNTIME_DEBUG_AUDIT'):
        for fold in range(5):
            graph=joblib.load(BASE/f'fresh_graphs/fold_{fold}/graph.joblib');ids=data.ids[data.series['fold']==fold]
            selected=set(sorted(map(int,ids),key=lambda sid:__import__('hashlib').sha256(('NEXT6_EXTENSION_RUNTIME:'+str(sid)).encode()).hexdigest())[:20])
            for sid in ids:
                i=data.id_to_index[int(sid)];h,o,tau=data.get(int(sid));a,b=data.series['online_offsets'][i:i+2];state=SafeState(h,threshold=cut,graph=graph);out=state.replay(o)
                error=float(np.abs(out[:,1]-oldscore[a:b]).max(initial=0));maxerror=max(maxerror,error);assert error==0;np.testing.assert_array_equal(out[:,0],oldstates[a:b])
                if int(sid) in selected:
                    original=State(h,threshold=cut,graph=graph).replay(o);np.testing.assert_array_equal(out,original)
                    scalar=SafeState(h,threshold=cut,graph=graph);other=SafeState(h[::-1],threshold=cut,graph=graph);size=scalar.bounded_mutable_state_bytes;actual=[]
                    for x in o:actual.append(scalar.predict_one(x));other.predict_one(-x)
                    np.testing.assert_array_equal(out,actual);assert size==scalar.bounded_mutable_state_bytes
                    at=min(117,len(o)//2);prefix=SafeState(h,threshold=cut,graph=graph);head=prefix.replay(o[:at]);restored=pickle.loads(pickle.dumps(prefix));np.testing.assert_array_equal(head,out[:at]);np.testing.assert_array_equal(restored.replay(o[at:]),out[at:])
                    changed=o.copy();changed[at:]=-changed[at:]*3.;np.testing.assert_array_equal(SafeState(h,threshold=cut,graph=graph).replay(changed)[:at],out[:at]);np.testing.assert_array_equal(SafeState(h,threshold=cut,graph=graph).replay(o[:at]),out[:at])
                    nscalar+=len(o)
            checkpoint('safe adapter verified RP fold',f'{fold}:1280 replay,20 extra streaming audits')
            proof.append({'fold':fold,'series':len(ids),'extra_scalar_prefix_serialization_series':20,'graph_sha256':digest(BASE/f'fresh_graphs/fold_{fold}/graph.joblib')})
    # Nonempty known-positive topology exercises actual conditional table reads,
    # including both phase models and sparse-context backoff.
    with activity('nonempty_graph_scalar_probability_audit','RUNTIME_DEBUG_AUDIT'):
        graph=joblib.load(BASE/'interaction_power/KNOWN_POSITIVE_GRAPH.joblib');rng=np.random.default_rng(930);h=rng.normal(size=2048);o=rng.normal(size=1024);s=SafeState(h,threshold=cut,graph=graph);previous=s.H_previous_state.copy();out=s.replay(o);manual=[]
        for end in np.flatnonzero(out[:,4]):
            current=categorize(out[end,12:]);lf=ln=used=0
            for j,i in enumerate(graph.parents):
                if i<0:continue
                own,p,y=int(previous[j]),int(previous[i]),int(current[j]);lf+=np.log(graph.full[1,j,own,p,y]/graph.full[0,j,own,p,y]);ln+=np.log(graph.null[1,j,own,p,y]/graph.null[0,j,own,p,y]);used+=1
            np.testing.assert_allclose(out[end,7:10],[lf/max(used,1),ln/max(used,1),(lf-ln)/max(used,1)],rtol=1e-12,atol=1e-12);previous=current
        scalar=SafeState(h,threshold=cut,graph=graph);np.testing.assert_array_equal(np.array([scalar.predict_one(x) for x in o]),out)
    save('SAFE_RUNTIME_PARITY.json',{'status':'PASS','created_utc':utc_now(),'all_RP_series':6400,'all_RP_evidence_max_error':maxerror,'all_RP_macrostate_exact':True,'extra_scalar_points':nscalar,'folds':proof,'nonempty_known_positive_graph_manual_table_parity':True,'prefix_suffix_length_serialization_interleaving':'PASS','H_overflow_adapter_no_ordinary_RP_output_change':True})
    dest=BASE/'audit';dest.mkdir(exist_ok=True);xml=dest/'EXTENSION_TESTS.xml'
    with activity('extension_affected_tests','RUNTIME_DEBUG_AUDIT'):
        tests=sorted(str(p) for p in (ROOT/'tests').glob('test_next6_extension_*.py'));cmd=[sys.executable,'-B','-m','pytest',*tests,'-q','-p','no:cacheprovider','--junitxml',str(xml)];r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8');(dest/'EXTENSION_TESTS.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');assert r.returncode==0,r.stdout+r.stderr
        suites=ET.parse(xml).getroot();test_count=sum(int(e.attrib.get('tests',0)) for e in suites.iter('testsuite'));fail=sum(int(e.attrib.get('failures',0))+int(e.attrib.get('errors',0)) for e in suites.iter('testsuite'))
        save('audit/EXTENSION_TEST_RECEIPT.json',{'status':'PASS','created_utc':utc_now(),'tests':test_count,'failures_and_errors':fail,'command':cmd,'test_source_sha256':{str(Path(p).relative_to(ROOT)):digest(p) for p in tests},'synthetic_only_tests_no_raw_competition_scanner':True})
    data.guard.check_state()
if __name__=='__main__':main()
