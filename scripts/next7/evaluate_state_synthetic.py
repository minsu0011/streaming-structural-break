"""G1 preregistered state/event diagnostics on fixed paired synthetic paths."""
from pathlib import Path
import argparse
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE,save,read_json,digest,csv_write,utc_now,activity,require_guard
from src.next7.scale import ScaleState,SCALES,RULES
from src.next7.event import new_event_args,event_replay
from src.next7.calibration import load_calibration,cutoff
from src.next6_extension.fresh_runtime import SafeState
from src.next6_extension.synthetic import generated

VARIANTS=[('NEXT6_HARD4_EVENT',0,0,False)]+[
    (s+'_'+r,si,ri,True) for si,s in enumerate(SCALES) for ri,r in enumerate(RULES)]


def first(mask):
    index=np.flatnonzero(mask)
    return int(index[0]) if len(index) else -1


def measure(out,control,tau,info,horizon):
    p=out[:horizon]
    c=control[:horizon]
    latch=first(p[:,3]>0)
    clatch=first(c[:,3]>0)
    slatch=first(p[:,27]>0)
    changed=tau>=0 and tau<horizon
    end=info['intervention_end']
    row={'horizon':horizon,'observed_break':changed,'tau_offline_only':tau,
        'any_latch':latch>=0,'first_latch':latch,'control_any_latch':clatch>=0,
        'excess_latch_vs_paired_control':latch>=0 and clatch<0,
        'prebreak_latch':changed and 0<=latch<tau,
        'first_detection':changed and latch>=tau,
        'first_detection_delay':latch-tau if changed and latch>=tau else None,
        'scale_only_latch':slatch>=0,'scale_first_latch':slatch,
        'scale_first_detection':changed and slatch>=tau,
        'scale_first_detection_delay':slatch-tau if changed and slatch>=tau else None,
        'core_only_latch':bool(p[-1,26]>0),
        'confirmation_max':float(p[:,2].max()),'event_memory_final':float(p[-1,3]),
        'scale_event_memory_final':float(p[-1,27]),'event_memory_delta_final':float(p[-1,3]-c[-1,3]),
        'current_score_final':float(p[-1,1]),'current_state_final':int(p[-1,0]),
        'current_score_mean':float(p[:,1].mean()),'event_score_mean':float(p[:,3].mean()),
        'confirmed_episodes_capped':int(p[-1,4]),
        'first_shock_time':int(p[-1,19]),'first_confirmation_time':int(p[-1,20]),
        'event_without_confirmation':bool(np.any((p[:,3]>0)&(p[:,20]<0))),
        'event_monotone':bool(np.all(np.diff(p[:,3])>=0)),
        'paired_prechange_max_error':float(np.abs(p[:min(info['scheduled'],horizon)]-c[:min(info['scheduled'],horizon)]).max(initial=0))}
    if end is not None and end<horizon:
        columns=[0,1,2,7,9,10,11,12,13,14,15,16,17,18,25,28,29]
        mismatch=np.any(p[end:,columns]!=c[end:,columns],axis=1)
        last=np.flatnonzero(mismatch)
        row['last_current_mismatch_delay']=int(last[-1]) if len(last) else -1
        row['tail_after_128_available']=end+128<horizon
        row['current_exact_coupled_after_128']=bool(not mismatch[128:].any()) if end+128<horizon else None
        row['event_memory_contamination_after_128']=float(np.max(np.abs(p[end+128:,3]-c[end+128:,3]))) if end+128<horizon else None
        row['event_retained_after_recovery']=bool(np.any((p[end:,0]==1)&(p[end:,3]>0)))
    return row


def main(only_case=None):
    require_guard()
    protocol=read_json(BASE/'NEXT7_PROTOCOL_LOCK.json')
    calibration=load_calibration()
    implementation=read_json(BASE/'G1_IMPLEMENTATION_LOCK.json')
    for path,sha in implementation['source_sha256'].items():
        assert digest(ROOT/path)==sha,path
    cases=[only_case] if only_case else protocol['evaluation']['cases']
    assert all(case in protocol['evaluation']['cases'] for case in cases)
    lockfile=BASE/'G1_SYNTHETIC_EVALUATION_LOCK.json'
    execution={name:digest(ROOT/name) for name in ('scripts/next7/evaluate_state_synthetic.py',
        'src/next7/calibration.py','src/next7/scale.py','src/next7/event.py')}
    if not lockfile.exists():
        save('G1_SYNTHETIC_EVALUATION_LOCK.json',{'status':'FROZEN','created_utc':utc_now(),
            'source_sha256':execution,'calibration_sha256':digest(BASE/'SCALE_CALIBRATION_FROZEN.json'),
            'variants':VARIANTS,'evaluation':protocol['evaluation'],
            'candidate_accounting':'six scale/rule screens plus frozen hard4 reference; no RP score released'})
    else:
        assert read_json(lockfile)['source_sha256']==execution
    suite=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
    for case in cases:
        path=BASE/f'synthetic_g1/{case}'
        if (path/'DONE.json').exists():
            continue
        with activity('G1 paired state synthetic '+case,'ROBUSTNESS_RESEARCH','F3_CONFIRMED_EVENT_MEMORY'):
            horizon=16384 if case.startswith('N') else 4096
            horizons=[1024,4096,16384] if case.startswith('N') else [1024,4096]
            results=[]
            forensic=[]
            for seed in range(512):
                h,o,tau,info=generated(case,seed,partition=31,horizon=horizon,control=False,lock=suite)
                hc,oc,ctau,ci=generated(case,seed,partition=31,horizon=horizon,control=True,lock=suite)
                np.testing.assert_array_equal(h,hc)
                raw_equal=np.array_equal(o,oc)
                core=SafeState(h,threshold=calibration['core_cut']).replay(o)
                scale_state=ScaleState(h)
                scale=scale_state.replay(o)
                if raw_equal:
                    core_control,scale_control=core,scale
                else:
                    core_control=SafeState(h,threshold=calibration['core_cut']).replay(oc)
                    scale_control=ScaleState(h).replay(oc)
                valid=scale_state.calibration['reference_valid']
                for name,si,ri,include_scale in VARIANTS:
                    cut=cutoff(calibration,len(h),si,ri)
                    arguments=new_event_args(calibration['core_cut'],cut,si,ri,True,include_scale,valid)
                    out=event_replay(core,scale,arguments)
                    if raw_equal:
                        control=out
                    else:
                        control=event_replay(core_control,scale_control,
                            new_event_args(calibration['core_cut'],cut,si,ri,True,include_scale,valid))
                    for hh in horizons:
                        row={'case':case,'seed':seed,'variant':name,'scale_cut':cut,
                             **measure(out,control,tau,info,hh)}
                        assert not row['event_without_confirmation']
                        assert row['event_monotone'] and row['paired_prechange_max_error']==0
                        if row.get('tail_after_128_available'):
                            assert row['current_exact_coupled_after_128'],(case,seed,name)
                        results.append(row)
                    # Episode starts are exported; the streaming state stores no growing event list.
                    confirmed=out[:,17]>.5
                    episode=np.flatnonzero(confirmed&~np.r_[False,confirmed[:-1]])
                    for t in episode:
                        oldest=int(out[t,23])
                        forensic.append({'case':case,'seed':seed,'variant':name,'event_time':int(t),
                            'firstShockTime':int(out[t,19]),'firstConfirmationTime':int(out[t,20]),
                            'firstLatchTime':int(out[t,21]),'confirmationChannel':int(out[t,22]),
                            'confirmationBlocks':','.join(map(str,range(oldest,oldest+4))),
                            'confirmationStrength':float(out[t,2]),'eventDirection':int(out[t,8]),
                            'eventAge':int(out[t,5]),'tau_offline_only':tau,
                            'postchange_offline':tau>=0 and t>=tau})
                if seed%128==127:
                    print(utc_now(),'G1',case,seed+1,'/512',flush=True)
            csv_write(f'synthetic_g1/{case}/SERIES.csv',results)
            csv_write(f'synthetic_g1/{case}/EVENTS.csv',forensic)
            summary=[]
            for name,_,_,_ in VARIANTS:
                for hh in horizons:
                    cohort=[r for r in results if r['variant']==name and r['horizon']==hh]
                    record={'case':case,'variant':name,'horizon':hh,'series':len(cohort)}
                    for metric in ('any_latch','control_any_latch','excess_latch_vs_paired_control','prebreak_latch',
                        'first_detection','scale_only_latch','scale_first_detection','core_only_latch',
                        'confirmation_max','event_memory_final','event_memory_delta_final','current_score_mean','event_score_mean'):
                        record[metric]=float(np.mean([r[metric] for r in cohort]))
                    delays=[r['first_detection_delay'] for r in cohort if r['first_detection_delay'] is not None]
                    record['detected_only_delay_median']=float(np.median(delays)) if delays else None
                    record['detected_only_delay_count']=len(delays)
                    summary.append(record)
            csv_write(f'synthetic_g1/{case}/SUMMARY.csv',summary)
            save(f'synthetic_g1/{case}/DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'case':case,
                'seeds':512,'horizons':horizons,'variants':len(VARIANTS),'source_sha256':execution,
                'SERIES_sha256':digest(path/'SERIES.csv'),'EVENTS_sha256':digest(path/'EVENTS.csv'),
                'calibration_sha256':digest(BASE/'SCALE_CALIBRATION_FROZEN.json'),
                'prefix_pair_and_event_contract':'PASS','no_outer_RP_scores_used':True})
    require_guard().check_state()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--case')
    main(parser.parse_args().case)
