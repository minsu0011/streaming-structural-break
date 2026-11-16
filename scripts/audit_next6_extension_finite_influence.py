"""Boundary-complete coupling audit of the exported network, not a score study."""
from pathlib import Path
import sys,os,pickle
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.observer import load,ObservableModel
from src.next6_extension.runtime import Prepared
from src.next6_extension.fresh_state import FLOAT32_MAX
from src.next2.counterexamples import generate

def main():
    backgrounds=('N0','N1','N2','N3','D2','V2');durations=(1,4,16,17,64,256);patterns=('positive_extreme','negative_extreme','alternating_extreme','zero','eight_sigma');seeds=8
    save('FINITE_INFLUENCE_AUDIT_LOCK.json',{'created_utc':utc_now(),'source_sha256':digest(Path(__file__)),'backgrounds':backgrounds,'seeds_per_background':seeds,'all_block_start_offsets':list(range(16)),'durations':durations,'patterns':patterns,'graphs':['RP_fold0_export','coherent_known_positive'],
        'endpoint':'All27 outputs bitwise equal to paired original suffix after contamination end-exclusive +128. Same H and aligned timestamps. Check mutable state too.',
        'scope':'Temporary observation contamination; not an intervention into a latent AR/ARCH process. No new score benchmark, model, threshold, or graph fit.'})
    model=load(BASE/'observable_models/fold_0/model.joblib');graph=joblib.load(BASE/'coherent_power/KNOWN_POSITIVE_GRAPH.joblib');positive=ObservableModel(model.threshold,graph,model.macro_probabilities.copy(),model.macro_support.copy(),{'diagnostic_only':True},model.source_sha256.copy());original=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json');rows=[];worst=0;maxafter=0
    for graphname,m in [('RP_fold0_export',model),('coherent_known_positive',positive)]:
        prepared=Prepared(m)
        for case in backgrounds:
            with activity('finite_influence_'+graphname+'_'+case,'RUNTIME_INVARIANT_AUDIT'):
                for seed in range(seeds):
                    h,o,_=generate(case,seed,partition=36,horizon=1024,control=case.startswith('N'),lock=original);template=pickle.dumps(prepared.new_state(h));base=pickle.loads(template);expected=base.replay(o)
                    for offset in range(16):
                        at=256+offset
                        for duration in durations:
                            end=at+duration;bound=end+128
                            for pattern in patterns:
                                changed=o.copy()
                                if pattern=='positive_extreme':changed[at:end]=FLOAT32_MAX
                                elif pattern=='negative_extreme':changed[at:end]=-FLOAT32_MAX
                                elif pattern=='alternating_extreme':changed[at:end]=np.where(np.arange(duration)%2==0,FLOAT32_MAX,-FLOAT32_MAX)
                                elif pattern=='zero':changed[at:end]=0
                                else:changed[at:end]=np.median(h)+8*np.std(h.astype(float))
                                state=pickle.loads(template);actual=state.replay(changed);assert np.isfinite(actual).all();np.testing.assert_array_equal(actual[:at],expected[:at]);np.testing.assert_array_equal(actual[bound:],expected[bound:]);difference=np.flatnonzero(np.any(actual!=expected,axis=1));last=int(difference[-1]) if len(difference) else -1;lag=max(0,last-end+1);worst=max(worst,lag)
                                for j in (3,4,5,6,7,8,14):np.testing.assert_array_equal(state.features.args[j],base.features.args[j])
                                np.testing.assert_array_equal(state.args[3],base.args[3]);np.testing.assert_array_equal(state.args[4],base.args[4])
                                rows.append({'graph':graphname,'background':case,'seed':seed,'block_start_offset':offset,'duration':duration,'pattern':pattern,'last_affected_offset_after_end_exclusive':lag,'all27_coupled_after128':True,'final_mutable_state_equal':True})
                checkpoint('finite influence completed',graphname+' '+case)
    csv_write('FINITE_INFLUENCE_STRESS_CASES.csv',rows);save('FINITE_INFLUENCE_AUDIT_DONE.json',{'status':'PASS','created_utc':utc_now(),'cases':len(rows),'all27_columns_bitwise_coupled':True,'all_mutable_state_bitwise_coupled':True,'maximum_observed_remaining_affected_points_after_end':worst,'conservative_bound':128,'source_sha256':digest(Path(__file__))})
if __name__=='__main__':main()
