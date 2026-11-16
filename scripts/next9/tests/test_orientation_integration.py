import src.next9
import numpy as np
from src.next9.data import FitScope
from src.next9.learning import fit_classifier
from src.next9.orientation_contract import orientation_design,fit_common_range,PAIR_MENU
from src.next9.inference import compare


def test_fixed_orientation_pipeline_on_heldout_toy_groups():
    ids=np.repeat(np.arange(256),20)
    times=np.tile(np.arange(20),256)
    f=np.tile(np.linspace(-2,2,20),256)
    state=np.where(ids%2==0,2,0)
    # Synthetic target has opposite F slopes; never used by state construction.
    y=((f*np.where(state==2,1,-1))>0).astype(int)
    tr=ids<128;va=~tr
    scope=FitScope(frozenset(range(128)),frozenset(range(128,256)),frozenset(), 'toy_pipeline')
    domain=fit_common_range(f[tr],state[tr],ids[tr],y[tr],scope,PAIR_MENU[0])
    assert domain.status=='SUPPORTED'
    args=(f,state,np.where(state==2,6,0),np.where(state==2,0,1),state)
    predictions={}
    for method in ['O1','O3_MOVE','O5_MOVE_REV','STATE_ONLY']:
        x=orientation_design(*args,method)
        model=fit_classifier(x[tr],y[tr],ids[tr],ids[tr],scope)
        predictions[method]=model.predict_proba(x[va])[:,1]
    predictions['CURRENT_F_DIRECTION']=f[va]
    keep=domain.mask(f[va],state[va])
    result=compare(y[va][keep],{k:v[keep] for k,v in predictions.items()},times[va][keep],ids[va][keep],
        'toy_orientation_pipeline',direction_states=state[va][keep],pair=(2,0),draws=20)
    # This fixture has constant F at a given t: signed same-time raw-F rank is
    # uninformative even though ordinary pooled AUC would show a slope. It must
    # not produce a conditional same-time direction certificate.
    assert result['orientation']['first']['point'] is None
    assert 'CURRENT_F_DIRECTION' not in result['logloss']
