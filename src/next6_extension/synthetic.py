"""Fixed independent diagnostic partitions and paired finite interventions."""
import copy
import numpy as np
from src.next2.counterexamples import generate

BASE_CASES=('N0','N1','N2','N3','M1','M2','V1','V2','V3','V4','A1','A2','D1','D2','T1','T2','O1')
EXTRA_CASES=('OB16','REPEATED_SHOCK','TEMP_SCALE_PULSE','RETURN_NORMAL','LATE_PERSISTENT','SCALE_PLUS_SHOCK')
CASES=BASE_CASES+EXTRA_CASES

def generated(case,index,*,partition,horizon,control,lock):
    if case not in CASES:raise ValueError('Unknown fixed diagnostic')
    at=int(lock['tau_cycle'][index%len(lock['tau_cycle'])]);end=None;kind='null' if case.startswith('N') else 'persistent_change';source=case
    local=lock
    if case in ('OB16','REPEATED_SHOCK'):source='O1'
    elif case in ('TEMP_SCALE_PULSE','RETURN_NORMAL'):source='N2'
    elif case in ('LATE_PERSISTENT','SCALE_PLUS_SHOCK'):source='V2'
    if case=='LATE_PERSISTENT':local=copy.deepcopy(lock);local['tau_cycle']=[1536];at=1536
    forced_control=control or case in ('OB16','REPEATED_SHOCK','TEMP_SCALE_PULSE','RETURN_NORMAL')
    h,o,tau=generate(source,index,partition=partition,horizon=horizon,control=forced_control,lock=local)
    if case=='O1':kind='isolated_contamination';end=at+1
    if case=='OB16':
        kind='burst_contamination';end=at+16
        if not control:
            stop=min(end,horizon);o[at:stop]+=8.*np.where((np.arange(max(stop-at,0))+index)%2==0,1,-1).astype(np.float32)
    elif case=='REPEATED_SHOCK':
        kind='repeated_contamination';end=at+3*128+1
        if not control:
            for j in range(4):
                t=at+j*128
                if t<horizon:o[t]+=20.*(1 if (index+j)%2==0 else -1)
    elif case in ('TEMP_SCALE_PULSE','RETURN_NORMAL'):
        kind='temporary_change';end=at+(32 if case=='TEMP_SCALE_PULSE' else 256)
        if not control:o[at:min(end,horizon)]*=np.sqrt(2.);tau=at if at<horizon else -1
    elif case=='SCALE_PLUS_SHOCK':
        if not control and at+32<horizon:o[at+32]+=20.*(1 if index%2==0 else -1)
    if control:tau=-1
    return h,o,int(tau),{'scheduled':at,'intervention_end':end,'kind':kind,'temporary_change_target':'first change remains occurred even after return' if kind=='temporary_change' else None}

def first_alarm(score,threshold):
    ix=np.flatnonzero(np.asarray(score)>threshold);return int(ix[0]) if len(ix) else -1

def recovery(delta,start,tolerance):
    good=np.abs(delta[start:])<=tolerance
    if len(good)<16:return None
    run=np.convolve(good.astype(np.int16),np.ones(16,dtype=np.int16),'valid');ix=np.flatnonzero(run==16)
    return int(ix[0]) if len(ix) else None

def metrics(score,control,*,threshold,tau,info,horizon,units):
    p=np.asarray(score[:horizon]);c=np.asarray(control[:horizon]);first=first_alarm(p,threshold);at=info['scheduled'];changed=tau>=0 and tau<horizon
    post=np.flatnonzero(p[max(tau,0):]>threshold) if changed else np.empty(0,dtype=int)
    r={'horizon':horizon,'threshold':float(threshold),'observed_break':changed,'first_alarm':first,'any_alarm':first>=0,'prebreak_alarm':changed and first>=0 and first<tau,
        'postbreak_first_detection':changed and first>=tau,'postbreak_any_detection':changed and bool(len(post)),'first_detection_delay':first-tau if changed and first>=tau else None,
        'first_postbreak_alarm_delay':int(post[0]) if len(post) else None,'score_max':float(p.max()),'paired_prechange_max_error':float(np.abs(p[:min(at,horizon)]-c[:min(at,horizon)]).max(initial=0)),
        'score_units':units,'operational_state_not_break_probability':units=='fresh block evidence','kind':info['kind'],'tau':tau}
    end=info['intervention_end']
    if end is not None and end<horizon:
        tolerance=0. if units=='fresh block evidence' else .01
        delay=recovery(p-c,end,tolerance);r.update({'recovery_delay':delay,'recovery_censored':delay is None,'recovery_tolerance':tolerance,'paired_effect_after_128_max':float(np.abs(p[min(end+128,horizon):]-c[min(end+128,horizon):]).max(initial=0)),
            'paired_excess_alarm_after_128':bool(np.any((p[min(end+128,horizon):]>threshold)&(c[min(end+128,horizon):]<=threshold)))})
    return r
