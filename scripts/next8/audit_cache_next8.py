"""Check stored RP coordinates against causal replay without using row labels."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
import numpy as np
from src.next8.common import BASE,save,read_json,digest,utc_now,activity,note
from src.next8.cache import FeatureCache
from src.next8.features import FeatureState,N_FEATURES,NAMES,DETAILED


def main():
    with activity('Audit feature cache hashes and original-series prefix/lag replay','CAUSAL_CACHE_AUDIT','B7_NORMALIZATION_AND_LINEAGE'):
        cache=FeatureCache(verify_values=True);data=cache.data
        selected=[]
        # Outcome-independent coverage: first ID of each existing fold x H bin.
        lengths=np.diff(data.series['historical_offsets'])
        for fold in range(5):
            for low,high in ((0,1249),(1250,1749),(1750,2499),(2500,3999),(4000,1000000)):
                members=[j for j,s in enumerate(data.ids) if data.guard.split.assignment(int(s))[1]==fold and low<=lengths[j]<=high]
                if members:selected.append(members[0])
        checks=[]
        for ix in selected:
            sid=int(data.ids[ix]);data.guard.admit([sid],purpose='NEXT8 H-only cached prefix consistency')
            ho=data.series['historical_offsets'];oo=data.series['online_offsets'];start=int(oo[ix])
            h=data.historical[ho[ix]:ho[ix+1]];o=data.online[oo[ix]:oo[ix+1]]
            engine=FeatureState(h);limit=min(65,len(o))
            for t in range(limit):
                for lag in (1,4,16):
                    # Before seeing current point, the lookup at (current row)
                    # must equal exactly this streaming state's past buffers.
                    np.testing.assert_array_equal(cache.lag(np.array([start+t]),np.arange(N_FEATURES),lag)[0],engine.lag(lag))
                value,state,previous,movement=engine.update(float(o[t]))
                np.testing.assert_array_equal(value,cache.values[start+t])
                np.testing.assert_array_equal(state,cache.states(np.array([start+t]),np.arange(N_FEATURES))[0])
                np.testing.assert_array_equal(previous,cache.states(np.array([start+t]),np.arange(N_FEATURES),previous=True)[0])
            checks.append({'dataset_id':sid,'fold':data.guard.split.assignment(sid)[1],'H_length':len(h),
                'prefix_points':limit,'all153_coordinates':True,'all_three_lags':'PASS','current_and_pre_state':'PASS'})
        support=[]
        counts=np.diff(data.series['online_offsets'])
        for j,name in enumerate(NAMES):
            support.append({'feature':j,'name':name,'H_valid_series':int(cache.series['valid'][:,j].sum()),
                'H_unknown_series':int((~cache.series['valid'][:,j]).sum()),
                'online_bound_points':int(cache.series['online_bound_count'][:,j].sum()),
                'online_bound_fraction':float(cache.series['online_bound_count'][:,j].sum()/counts.sum()),
                'detailed_eligible':j in DETAILED,'target_labels_used':False})
        save('H_NORMALIZATION_SUPPORT_NEXT8.json',{'status':'PASS','timestamp':utc_now(),'features':support,
            'H_reference_consistency':'PASS','short_H_rule':'<160: deterministic UNKNOWN',
            'support_counts_not_target_relation':True,'no_feature_menu_change_from_this_audit':True,
            'inherited_cumulative_age_caveat':'H40 warmup advances inherited cumulative/filter counters; Online continues those counters. Fixed calibration does not prove stationarity or prevent later clipping. Time/H/self-mechanical controls remain required.'})
        save('CAUSAL_CACHE_AUDIT_NEXT8.json',{'status':'PASS','timestamp':utc_now(),'checks':checks,
            'full_feature_file_hash_verified':True,'feature_manifest_sha256':digest(cache.directory/'MANIFEST.json'),
            'label_input_to_feature_engine':False,'source_sha256':digest(__file__)})
    note('Causal feature cache audit PASS','Whole cache hashes and outcome-independent fold/H coverage agree with scalar prefixes','B7_NORMALIZATION_AND_LINEAGE')


if __name__=='__main__':main()
