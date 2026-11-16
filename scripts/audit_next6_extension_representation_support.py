from pathlib import Path
import sys,ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from src.next6_extension.common import *
from src.next6_extension.fresh_state import NAMES
from src.next6.data import source
from summarize_next6_extension import rows

def main():
    data=source();cal=rows(BASE/'H_BLOCK_CALIBRATION.csv');blocks=np.array([int(r['H_calibration_blocks']) for r in cal]);floors={int(r['dataset_id']):ast.literal_eval(r['spread_floor_nodes']) for r in cal};quantiles=np.quantile(blocks,[0,.25,.5,.75,1]);out=[]
    for j,name in enumerate(NAMES):out.append({'node':j,'name':name,'RP_series':6400,'reference_valid_series':sum(r['reference_valid']=='True' for r in cal),'spread_floor_series':sum(j in v for v in floors.values()),'spread_floor_fraction':sum(j in v for v in floors.values())/6400,'H_calibration_blocks_min':quantiles[0],'H_calibration_blocks_q25':quantiles[1],'H_calibration_blocks_median':quantiles[2],'H_calibration_blocks_q75':quantiles[3],'H_calibration_blocks_max':quantiles[4],'floor_not_retuned':True})
    csv_write('H_REFERENCE_SUPPORT_SUMMARY.csv',out)
    cache=BASE/'block_cache';states=np.load(cache/'states.npy',mmap_mode='r');ids=np.load(cache/'dataset_id.npy',mmap_mode='r');fold=np.load(cache/'fold.npy',mmap_mode='r');phase=np.load(cache/'phase_clean.npy',mmap_mode='r');rows_out=[]
    with activity('fresh_node_representation_support','BOTTLENECK_DESCRIPTION'):
        for partition,mask in [('outer_train_fold0',fold!=0),('outer_heldout_fold0',fold==0)]:
            for p in (0,1):
                use=mask&(phase==p);sid=ids[use];s=states[use];unique,code,count=np.unique(sid,return_inverse=True,return_counts=True);w=1/count[code]
                for j,name in enumerate(NAMES):
                    fractions=[float(np.sum(w*(s[:,j]==value))/len(unique)) for value in range(4)];assert abs(sum(fractions)-1)<1e-12
                    rows_out.append({'partition':partition,'phase':p,'node':j,'name':name,'series_with_clean_blocks':len(unique),'clean_blocks':len(s),'low_fraction':fractions[0],'normal_fraction':fractions[1],'high_fraction':fractions[2],'missing_fraction':fractions[3],'spread_floor_fraction_among_present_series':sum(j in floors[int(k)] for k in unique)/len(unique),'equal_series_phase_weight':True,'no_new_threshold_or_fit':True})
    csv_write('FRESH_NODE_STATE_OCCUPANCY.csv',rows_out);save('REPRESENTATION_SUPPORT_AUDIT_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'H_calibration_block_quantiles':quantiles.tolist(),'limited_volatility_dependence_spread_support':True,'floor_and_nodes_unchanged':True,'interpretation':'Descriptive support diagnostic; frequent floor use does not by itself prove the cause of missed detections.'});data.guard.check_state()
if __name__=='__main__':main()
