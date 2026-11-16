"""Resumable label-blind feature construction on the fixed ResearchPool."""
from pathlib import Path
import sys,os,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
from src.next6.common import BASE,save,read_json,digest,activity,checkpoint
from src.next6.nodes import NodeState,NAMES,KOREAN
from src.next6.data import source,reference_matrix
import numpy as np

def main():
    assert (BASE/'NEXT6_PROTOCOL_LOCK.json').exists()
    if (BASE/'data/MANIFEST.json').exists():print('Complete node cache retained');return
    with activity('prepare_label_blind_nodes','RELATION_ANALYSIS'):
        data=source();rows=data.rows();n=len(rows['target']);ns=len(data.ids)
        directory=BASE/'data';directory.mkdir(exist_ok=True)
        defs={'nodes':('nodes.npy',(n,16),np.float32),'states':('states.npy',(n,16),np.int8),
            'tabular':('tabular.npy',(n,245),np.float32),'H_initial':('H_initial.npy',(ns,16,16),np.float32),
            'H_states':('H_states.npy',(ns,16,16),np.int8),'H_descriptors':('H_descriptors.npy',(ns,7),np.float64)}
        matrices={k:np.lib.format.open_memmap(directory/f,mode='r+' if (directory/f).exists() else 'w+',dtype=d,shape=shape) for k,(f,shape,d) in defs.items()}
        checkpointpath=directory/'PROGRESS.json';progress=read_json(checkpointpath) if checkpointpath.exists() else {'completed_series':0}
        if progress.get('node_source_sha256',digest(ROOT/'src/next6/nodes.py'))!=digest(ROOT/'src/next6/nodes.py'):raise RuntimeError('Cannot resume changed node source')
        j=reference_matrix('J102',data);c=reference_matrix('C205',data)
        for ix in range(progress['completed_series'],ns):
            sid=int(data.ids[ix]);h,o,_=data.get(sid);a,b=data.series['online_offsets'][ix:ix+2]
            state=NodeState(h)
            matrices['H_initial'][ix]=state.pseudo_x[-16:]
            matrices['H_states'][ix]=state.pseudo_s[-16:]
            z=(h-np.median(h))/max(float(np.std(h)),1e-6)
            descriptors=[len(h),float(np.std(h)),float(np.mean(np.abs(z)>2)),float(np.corrcoef(h[:-1],h[1:])[0,1]),
                float(np.std(h[:state.fit_length])),float(np.std(h[state.fit_length:])),float(np.mean(state.degenerate))]
            matrices['H_descriptors'][ix]=np.nan_to_num(descriptors)
            x,s,t=state.replay(o)
            assert x.shape==(len(o),16) and t.shape==(len(o),144) and np.isfinite(t).all()
            matrices['nodes'][a:b]=x;matrices['states'][a:b]=s
            matrices['tabular'][a:b,:77]=j[a:b];matrices['tabular'][a:b,77:101]=c[a:b];matrices['tabular'][a:b,101:]=t
            if (ix+1)%100==0 or ix+1==ns:
                for v in matrices.values():v.flush()
                save('data/PROGRESS.json',{'completed_series':ix+1,'node_source_sha256':digest(ROOT/'src/next6/nodes.py')},False)
                checkpoint('label-blind nodes',f'{ix+1}/{ns} series')
        save('data/MANIFEST.json',{'status':'COMPLETE','series':ns,'online_rows':n,'node_names':list(NAMES),
            'node_source_sha256':digest(ROOT/'src/next6/nodes.py'),'protocol_sha256':digest(BASE/'NEXT6_PROTOCOL_LOCK.json'),
            'arrays':{k:v[0] for k,v in defs.items()},'sha256':{v[0]:digest(directory/v[0]) for v in defs.values()},
            'target_in_feature_api':False,'fit_H_fraction':.6,'H_pseudo_online_continues_into_O':True,'shadow_rows':0,'seal_rows':0})
        # Label tables are separate from feature arrays and used only for fold-local maps/evaluation.
        with (BASE/'OFFICIAL_TAU_INDEX_NEXT6.csv').open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['dataset_id','outer_fold','official_tau_index','online_length_analysis_only','historical_length','target_contract'])
            for ix,sid in enumerate(data.ids):
                a,b=data.series['online_offsets'][ix:ix+2];ha,hb=data.series['historical_offsets'][ix:ix+2]
                w.writerow([sid,data.guard.split.assignment(int(sid))[1],data.series['tau'][ix],b-a,hb-ha,'t>=tau or all0 if absent'])
        support=[];p=read_json(BASE/'NEXT6_PROTOCOL_LOCK.json')
        for fold in range(5):
            for phase,(lo,hi) in p['phase_intervals'].items():
                count=points=missing=0
                for ix,sid in enumerate(data.ids):
                    if data.guard.split.assignment(int(sid))[1]==fold:continue
                    tau=data.series['tau'][ix];a,b=data.series['online_offsets'][ix:ix+2]
                    if tau<0:continue
                    rel=np.arange(b-a)-tau;nr=int(((rel>=lo)&(rel<=hi if hi is not None else True)).sum())
                    count+=nr>0;points+=nr;missing+=nr==0
                support.append({'outer_fold':fold,'scope':'OUTER_TRAIN_DISCOVERY','phase':phase,'series_with_support':count,'points':points,'events_missing_phase':missing})
        with (BASE/'EVENT_SUPPORT_TABLE.csv').open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(support[0]));w.writeheader();w.writerows(support)
        data.guard.check_state()

if __name__=='__main__':main()
