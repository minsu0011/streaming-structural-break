"""Freeze diagnostic details, generate actual raw controls through identical engines."""
from pathlib import Path
import sys,os,copy,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np
from src.next6.common import BASE,read_json,save,digest,activity,checkpoint,utc_now
from src.next6.nodes import NodeState,SET_C
from src.next5.reference_features import ReferenceState
from src.next5.synthetic import cohort_series
from src.next6.evaluation import csv_write
from src.next6.graph import discover,design

def diagnostic_lock():
    path=BASE/'DIAGNOSTIC_LOCK.json'
    if path.exists():return read_json(path)
    assert not (BASE/'OUTER_SCORE_RELEASE.json').exists()
    menu=[]
    for kind,horizon,cases,paired in [('calibration',1024,['N0','N1','N2','N3'],False),
        ('counterexample',1024,['N0','N1','N2','N3','V1','V2','V3','V4','D1','D2','O1','OB16','LATE_V2'],True),
        ('long_null',4096,['N0','N1','N2','N3'],False)]:
        for case in cases:menu.append(dict(kind=kind,scenario=case,horizon=horizon,seeds=64,paired=paired))
    lock={'created_utc':utc_now(),'menu':menu,'threshold':'per fold and method, pooled 256 independent stationary calibration maxima at 1024; finite-sample rank ceil((n+1)*.95), strict >',
        'threshold_alpha':.05,'recovery':'first 16 consecutive points within absolute .01 of paired-control score, search from impulse/burst end; censored otherwise',
        'case_method':'M3 fixed before outer score, no winning-model case selection','case_order':'sha256 NEXT6_CASE:dataset_id, fold0 only',
        'case_categories':{'success':'first alarm after tau with delay <=31 and no prealarm','delayed':'first alarm after tau with delay >=32 and no prealarm','false_alarm':'no break and any alarm','miss':'break and no alarm'},
        'raw_null_graph':'64 stationary calibration series per background; diagnostic artificial phase split at t=256, never treated as real change; actual same lag discovery engine',
        'boundary_trap':'compare genuine per-series lag1 at first online point against intentionally invalid previous-series last online; no invalid features used by models',
        'late_break_tau':512,'model_scores':'all five fitted outer models; evaluation never alters calibration or model',
        'latency':'fold0 same H, actual scalar points, 5 interleaved blocks of 2048 points after JIT warmup; batch timing separate',
        'prefix_audit':'20 hash-ordered heldout series per fold; all fitted methods full-series fused versus cached OOF, 3 per fold additional altered suffix/reset/serialization/scalar checks',
        'H_calibration_diagnostic':'first64 hash-ordered outer0 training H only, pseudo-online versus same-H fitted engine state proportions; no recalibration'}
    save('DIAGNOSTIC_LOCK.json',lock);return lock

def generated(spec,index,control,base_lock):
    lock=copy.deepcopy(base_lock);scenario=spec['scenario']
    if scenario=='LATE_V2':scenario='V2';lock['tau_cycle']=[512]
    return cohort_series(spec['kind'],scenario,index,spec['horizon'],control,lock)

def key(spec):return spec['kind']+'__'+spec['scenario']

def prepare(spec,lock):
    directory=BASE/'synthetic_cache'/key(spec);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():return
    n=spec['seeds']*(2 if spec['paired'] else 1);horizon=spec['horizon']
    with activity('raw_controls_'+key(spec),'CONTROL_RESEARCH'):
        tab=np.lib.format.open_memmap(directory/'tabular.npy',mode='w+',dtype=np.float32,shape=(n,horizon,245))
        states=np.lib.format.open_memmap(directory/'states.npy',mode='w+',dtype=np.int8,shape=(n,horizon,16))
        initial=np.empty((n,16,16),dtype=np.int8);rows=[];p=0
        for seed in range(spec['seeds']):
            for control in ((False,True) if spec['paired'] else (True,)):
                h,o,tau=generated(spec,seed,control,lock)
                j=ReferenceState(h,{'version':'next5_reference_features_v1','kind':'J102'});c=ReferenceState(h,SET_C);ns=NodeState(h)
                initial[p]=ns.pseudo_s[-16:];x,s,temporal=ns.replay(o)
                tab[p,:,:77]=j.replay(o);tab[p,:,77:101]=c.replay(o);tab[p,:,101:]=temporal;states[p]=s
                rows.append({'seed':seed,'control':control,'tau':int(tau),'scheduled_tau':512 if spec['scenario']=='LATE_V2' else int(lock['tau_cycle'][seed%len(lock['tau_cycle'])])});p+=1
            if seed%16==15:checkpoint('control feature generation',key(spec)+' '+str(seed+1)+'/64')
        tab.flush();states.flush();np.save(directory/'H_states.npy',initial,allow_pickle=False)
        save(str((directory/'DONE.json').relative_to(BASE)),{'spec':spec,'rows':rows,'node_source_sha256':digest(ROOT/'src/next6/nodes.py'),
            'hashes':{name:digest(directory/name) for name in ('tabular.npy','states.npy','H_states.npy')},'status':'COMPLETE'})

def null_graphs():
    records=[];traps=[]
    for case in ('N0','N1','N2','N3'):
        d=BASE/'synthetic_cache'/('calibration__'+case)
        tab=np.load(d/'tabular.npy',mmap_mode='r');states=np.load(d/'states.npy',mmap_mode='r');initial=np.load(d/'H_states.npy')
        ids=np.arange(64)+1000000;take=np.arange(0,1024,8);temporal=np.asarray(tab[:,take,101:]).reshape(-1,144)
        times=np.tile(take,64);sid=np.repeat(ids,len(take));sample={'X':design(temporal,times),'Y':temporal[:,:16].astype(float),'phase':(times>=256).astype(np.int8),'ids':sid}
        with activity('mechanical_raw_null_graph_'+case,'CONTROL_RESEARCH'):edges,rows=discover(sample,{int(i):int(i) for i in ids})
        for e in edges:records.append({'background':case,'actual_change':False,'phase_labels':'artificial fixed t256 null split',**e})
        csv_write('raw_null_graph/'+case+'_EDGE_DISCOVERY.csv',rows)
        # First online lag must refer to this H tail, never the previous series.
        invalid=np.asarray(tab[:-1,-1,101:117]);correct=np.asarray(tab[1:,0,117:133])
        traps.append({'background':case,'boundaries':63,'invalid_boundary_mismatches':int(np.any(abs(invalid-correct)>1e-6,axis=1).sum()),'status':'INVALID_CROSS_SERIES_LAG_DETECTED','invalid_features_used':False})
    csv_write('RAW_NULL_SELECTED_EDGES.csv',records);csv_write('SERIES_BOUNDARY_TRAPS.csv',traps)
    save('RAW_NULL_CONTROL.json',{'status':'COMPLETE','synthetic_only':True,'selected_edges':len(records),'mechanical_edges':sum(r['mechanical']=='MECHANICAL_DERIVED' for r in records),
        'interpretation':'Shared-input filters can predict each other on stationary raw noise. Selected edges are not evidence of an economic cause or a real break.',
        'stationary_phase_split':256,'series_per_background':64,'same_engine':True})

def main():
    p=argparse.ArgumentParser();p.add_argument('--null-graphs',action='store_true');a=p.parse_args()
    lock=diagnostic_lock();base=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
    assert digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')==read_json(BASE/'NEXT6_PROTOCOL_LOCK.json')['synthetic']['seed_lock_sha256']
    if a.null_graphs:null_graphs()
    else:
        for spec in lock['menu']:prepare(spec,base)

if __name__=='__main__':main()
