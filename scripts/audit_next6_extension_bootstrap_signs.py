"""Recover selected coefficient signs from saved resamples, without rediscovery."""
from pathlib import Path
import sys,os,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from threadpoolctl import threadpool_limits
from src.next6_extension.common import *
from src.next6_extension.bootstrap_graph import prepare
from src.next6.graph import pack_sample,owncols,ridge
from src.next6.data import arrays

def signs(packed,multiplicity,selected):
    values=[]
    with threadpool_limits(limits=2):
        for phase in (0,1):
            chosen=[e for e in selected if e['selection_phase']==phase]
            if not chosen:continue
            coefficients={}
            for validation_half in (0,1):
                tr=packed[2*phase+1-validation_half];tm=multiplicity[tr['unique_group']];w=tm[tr['code']]/tr['count'][tr['code']];w=w/w.sum();x=tr['x'];y=tr['y'];g=x.T@(x*w[:,None]);xy=x.T@(y*w[:,None])
                for e in chosen:
                    j=e['target'];c=e['source']+16*e['lag_index'];beta=ridge(g,xy[:,j],owncols(j)+[c]);coefficients.setdefault((e['source'],j,e['lag']),[]).append(float(beta[-1]))
            for e in chosen:
                coef=coefficients[e['source'],e['target'],e['lag']];assert coef[0]*coef[1]>0
                values.append({'source':e['source'],'target':e['target'],'lag':e['lag'],'selection_phase':phase,'coefficient_internal_train_half1':coef[0],'coefficient_internal_train_half0':coef[1],'sign':1 if coef[0]>0 else -1,'both_internal_halves_same_sign':True})
    return values

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);a=p.parse_args();fold=a.fold;done=BASE/f'graph_bootstrap/fold_{fold}/DONE.json';assert done.exists(),'Finish this fold resampling first'
    lock=BASE/'BOOTSTRAP_SIGN_RECOVERY_LOCK.json'
    if not lock.exists():save(lock.name,{'created_utc':utc_now(),'source_sha256':digest(Path(__file__)),'purpose':'Complete preregistered coefficient sign report from saved selected edges. Reconstruct exact multiplicities and weighted training regressions only; do not reselect edges.'})
    assert read_json(lock)['source_sha256']==digest(Path(__file__))
    dest=BASE/f'bootstrap_signs/fold_{fold}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'DONE.json').exists():return
    data,meta,v=arrays();original=joblib.load(ROOT/f'artifacts/next6/fold_{fold}/graphs/outer_full/graph.joblib');sample=pack_sample(data,meta,v,original.provenance['training_ids']);packed,halves,groups=prepare(sample,data.guard.split.groups);del sample
    with activity(f'selected_bootstrap_coefficient_signs_fold{fold}','RELATION_RESAMPLING_AUDIT'):
        if not (dest/'ORIGINAL_SIGNS.json').exists():save(str((dest/'ORIGINAL_SIGNS.json').relative_to(BASE)),{'edges':signs(packed,np.ones(len(groups),dtype=int),original.selected)})
        for replicate in range(200):
            out=dest/f'replicate_{replicate:03}.json'
            if out.exists():continue
            rec=read_json(BASE/f'graph_bootstrap/fold_{fold}/replicate_{replicate:03}.json');rng=np.random.default_rng(np.random.SeedSequence([2026091301,fold,replicate]));m=np.zeros(len(groups),dtype=int)
            for half in (0,1):
                ix=np.flatnonzero(halves==half);m[ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)))
            assert hashlib.sha256(m.tobytes()).hexdigest()==rec['multiplicity_sha256'];selected=signs(packed,m,rec['selected']);assert len(selected)==len(rec['selected'])
            save(str(out.relative_to(BASE)),{'fold':fold,'replicate':replicate,'exact_multiplicity_sha256':rec['multiplicity_sha256'],'edges':selected,'no_rediscovery':True})
            if (replicate+1)%25==0:checkpoint('selected coefficient signs',f'fold {fold}: {replicate+1}/200')
        save(str((dest/'DONE.json').relative_to(BASE)),{'status':'PASS','created_utc':utc_now(),'replicates':200,'multiplicities_exact':True,'original_selection_sign_gate_verified':True})
    data.guard.check_state()
if __name__=='__main__':main()
