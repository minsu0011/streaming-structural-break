"""Read immutable scale cutoffs and route using actual H length only."""
import numpy as np
from src.next7.common import BASE, read_json, digest


def load_calibration():
    result=read_json(BASE/'SCALE_CALIBRATION_FROZEN.json')
    assert result['status']=='FROZEN'
    assert result['protocol_sha256']==digest(BASE/'NEXT7_PROTOCOL_LOCK.json')
    return result


def cutoff(calibration,historical_length,scale_index=0,rule=1):
    lengths=[r['H_length'] for r in calibration['tables']]
    index=max(0,int(np.searchsorted(lengths,historical_length,side='right'))-1)
    return float(calibration['tables'][index]['cutoffs'][scale_index][rule])
