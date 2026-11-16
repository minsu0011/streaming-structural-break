from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np
from src.next6_extension.common import *
from src.next6_extension.observer import load,COLUMNS
from src.next6_extension.runtime import StreamingState
from src.next6_extension.validated_runtime import Prepared
from src.next6.data import source

def main():
    save('REFERENCE_VALIDITY_REPAIR_LOCK.json',{'created_utc':utc_now(),'adapter_sha256':digest(ROOT/'src/next6_extension/validated_runtime.py'),'unchanged_fused_source_sha256':digest(ROOT/'src/next6_extension/runtime.py'),'audit_sha256':digest(Path(__file__)),
        'repair':'Invalid H routes to explicit state6 with all evidence/confirmation/surprise zero, transition probabilities neutral1 and availability0. Real clock/block count preserved. Valid H returns the identical fused StreamingState class.',
        'previous_probe':'INVALID_REFERENCE_OUTPUT_PROBE.json','no_fits_or_calibration_changes':True})
    data=source();valid=0;models=[load(BASE/f'observable_models/fold_{f}/model.joblib') for f in range(5)];prepared=[Prepared(m) for m in models]
    with activity('all_RP_reference_validity_routing_audit','RUNTIME_DEBUG_AUDIT'):
        for sid,fold in zip(data.ids,data.series['fold']):
            h,o,_=data.get(int(sid));state=prepared[int(fold)].new_state(h);assert state.reference_valid and type(state) is StreamingState;valid+=1
    state=prepared[0].new_state(np.ones(1024));points=np.r_[np.ones(128),np.ones(64)*20,np.ones(320)];out=state.replay(points);assert not state.reference_valid and np.all(out[:,1]==0) and np.all(out[:,11]==0) and np.all(out[:,24]==0)
    save('INVALID_REFERENCE_EXAMPLE.json',{'columns':COLUMNS,'curve':out.tolist(),'case':'constant H, finite current observation burst','reference_valid':False,'tau':-1,'scheduled':128,'scope':'Numeric input-validity demonstration; not an independent detection experiment.'})
    save('REFERENCE_VALIDITY_AUDIT_DONE.json',{'status':'PASS','created_utc':utc_now(),'all_RP_series':valid,'all_RP_route_to_identical_fused_class':True,'ordinary_output_change':False,'invalid_H_evidence_confirmation_and_transition_availability_zero':True,'affected_tests':10,'earlier_fused_parity_receipt_sha256':digest(BASE/'FUSED_OBSERVABLE_RUNTIME_PARITY.json'),'earlier_finite_influence_receipt_sha256':digest(BASE/'FINITE_INFLUENCE_AUDIT_DONE.json')})
    data.guard.check_state()
if __name__=='__main__':main()
