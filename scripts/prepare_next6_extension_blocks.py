from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np
from src.next6_extension.common import *
from src.next6_extension.fresh_state import State,categorize,lineage
from src.next6.data import source

def main():
    data=source();parts=[];cal=[]
    with activity('prepare_finite_memory_RP_blocks','FEATURE_REPRESENTATION'):
        for i,sid in enumerate(data.ids):
            h,o,tau=data.get(int(sid));state=State(h);initial=state.H_previous_state.copy();out=state.replay(o);end=np.flatnonzero(out[:,4]);z=out[end,12:20];states=categorize(z,state.calibration['reference_valid']);prev=np.vstack((initial[None,:],states[:-1])) if len(states) else states.copy()
            start=end-15;phase=np.full(len(end),-1,dtype=np.int8)
            if tau<0:phase[:]=0
            else:phase[end<tau]=0;phase[start-16>=tau]=1
            parts.append({'dataset_id':np.full(len(end),sid,dtype=np.int32),'fold':np.full(len(end),data.series['fold'][i],dtype=np.int8),'time_end':end.astype(np.int32),'phase_clean':phase,
                'target_end':np.zeros(len(end),dtype=np.int8) if tau<0 else (end>=tau).astype(np.int8),'node_z':z.astype(np.float32),'states':states,'previous':prev,'global_row':data.series['online_offsets'][i]+end})
            cal.append({'dataset_id':int(sid),**state.calibration})
            if (i+1)%256==0:checkpoint('finite-memory block preparation',f'{i+1}/6400')
        dest=BASE/'block_cache';dest.mkdir(exist_ok=True);files={}
        for key in parts[0]:
            path=dest/(key+'.npy');assert not path.exists();np.save(path,np.concatenate([v[key] for v in parts]),allow_pickle=False);files[path.name]=digest(path)
        save('block_cache/MANIFEST.json',{'status':'COMPLETE','files_sha256':files,'series':len(data.ids),'blocks':sum(len(p['dataset_id']) for p in parts),'fresh_source_sha256':digest(ROOT/'src/next6_extension/fresh_state.py'),'design_lock_sha256':digest(BASE/'FRESH_STATE_DESIGN_LOCK.json'),'original_split_sha256':data.guard.split.digest})
        csv_write('FRESH_NODE_LINEAGE.csv',lineage());csv_write('H_BLOCK_CALIBRATION.csv',cal)
    data.guard.check_state()
if __name__=='__main__':main()
