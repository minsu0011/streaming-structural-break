"""Frozen H-length/null calibration, before any changed-scenario or RP score."""
from pathlib import Path
import copy
import math
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, read_json, save, digest, csv_write, utc_now, activity, require_guard, note
from src.next7.scale import ScaleState, SCALES, RULES
from src.next6_extension.synthetic import generated


def main():
    require_guard()
    gate=read_json(BASE/'runtime/CORE_TEST_GATE.json')
    assert gate['status']=='PASS'
    tested_sources={name.replace('\\','/'):sha for name,sha in gate['source_sha256'].items()}
    for relative in ('src/next7/scale.py','src/next7/event.py'):
        assert tested_sources[relative]==digest(ROOT/relative)
    protocol=read_json(BASE/'NEXT7_PROTOCOL_LOCK.json')
    sources={name:digest(ROOT/name) for name in ('src/next7/scale.py','src/next7/event.py',
        'scripts/next7/calibrate_scale.py','src/next6_extension/fresh_runtime.py',
        'src/next6_extension/fresh_state.py','src/next6_extension/synthetic.py','src/next2/counterexamples.py')}
    lock=BASE/'G1_IMPLEMENTATION_LOCK.json'
    if not lock.exists():
        save('G1_IMPLEMENTATION_LOCK.json',{'status':'FROZEN','created_utc':utc_now(),
            'source_sha256':sources,'protocol_sha256':digest(BASE/'NEXT7_PROTOCOL_LOCK.json'),
            'before_calibration_and_scientific_evaluation':True})
    else:
        assert read_json(lock)['source_sha256']==sources
    target=BASE/'SCALE_CALIBRATION_FROZEN.json'
    if target.exists():
        print('Already complete; preserve calibration')
        return
    contract=protocol['calibration']
    tables=[]
    for hlen in contract['H_lengths']:
        directory=BASE/f'calibration/H{hlen}'
        done=directory/'DONE.json'
        if done.exists():
            tables.append(read_json(done))
            continue
        with activity(f'scale null calibration H{hlen}','CALIBRATION','F1_SIGNED_SCALE'):
            suite=copy.deepcopy(read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'))
            suite['historical_length']=hlen
            rows=[]
            cut=np.zeros((2,3))
            for background in contract['nulls']:
                cohort=[]
                for seed in range(256):
                    h,o,tau,info=generated(background,seed,partition=30,horizon=1024,control=True,lock=suite)
                    assert tau==-1
                    s=ScaleState(h)
                    out=s.replay(o)
                    maxima=out[:,[4,7,10,13,16,19]].max(axis=0).reshape(2,3)
                    cohort.append(maxima)
                    for channel,name in enumerate(SCALES):
                        for rule,rname in enumerate(RULES):
                            rows.append({'H_length':hlen,'background':background,'seed':seed,
                                'scale':name,'rule':rname,'maximum':float(maxima[channel,rule]),
                                'H_calibration_blocks':s.calibration['calibration_blocks'],
                                'H_center':s.calibration['center'],'H_dispersion':s.calibration['dispersion']})
                quantile=np.sort(np.asarray(cohort),axis=0)[math.ceil(.95*257)-1]
                cut=np.maximum(cut,quantile)
                print(utc_now(),'CALIBRATED',hlen,background,flush=True)
            assert np.isfinite(cut).all() and (cut>0).all()
            csv_write(f'calibration/H{hlen}/MAXIMA.csv',rows)
            record={'status':'FROZEN','created_utc':utc_now(),'H_length':hlen,
                'cutoffs':cut.tolist(),'scale_order':list(SCALES),'rule_order':list(RULES),
                'null_series':1024,'horizon':1024,'partition':30,'rank_per_background':math.ceil(.95*257),
                'max_over_backgrounds':True,'strict_greater':True,
                'MAXIMA_sha256':digest(directory/'MAXIMA.csv'),'source_sha256':sources,
                'no_validation_or_changed_scenario_used':True}
            save(f'calibration/H{hlen}/DONE.json',record)
            tables.append(record)
    save('SCALE_CALIBRATION_FROZEN.json',{'status':'FROZEN','created_utc':utc_now(),
        'tables':tables,'core_cut':protocol['frozen_hard4']['threshold'],
        'protocol_sha256':digest(BASE/'NEXT7_PROTOCOL_LOCK.json'),
        'implementation_lock_sha256':digest(lock),'all_H_lengths_before_evaluation':True})
    note('Scale null calibration frozen','Ready for independent fixed-seed diagnostic evaluation','F2_FRESH_CONFIRMATION')


if __name__=='__main__':
    main()
