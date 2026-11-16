"""Fixed M1..M5 input hierarchy, with explicit aliases for absent branches."""
import numpy as np
from src.next9.orientation_contract import fixed_nonlinear,one_hot
from src.next9.movement_screen import feature_inputs
from src.next9.transition_inputs import path_features


def assemble(bank,rows,candidates,rule_specs,run_maps,laws,method):
    if method not in ('M1','M2','M3','M4','M5'):
        raise ValueError('Registered joint method required')
    if len(candidates)>12 or len(rule_specs)>6 or not set(rule_specs)<=set(candidates):
        raise ValueError('Frozen candidate and conditional feature caps required')
    rows=np.asarray(rows,dtype=np.int64)
    parts=[];names=[]
    def add(prefix,x):
        x=np.asarray(x,dtype=np.float32)
        if x.ndim==1:x=x[:,None]
        parts.append(x);names.extend(prefix+'_'+str(k) for k in range(x.shape[1]))
    for j in candidates:
        f=bank.current(rows,[j])[:,0]
        add(f'F{j}_fixed',fixed_nonlinear(f))
        if method=='M1':continue
        state=bank.pre_state(rows,j)
        add(f'F{j}_own',feature_inputs(bank,rows,j,state))
        if method=='M2':continue
        for family,count in [('S1_DIRECTION',4),('S2_DIRECTION_RUN',13),('S3_DIRECTION_SLOPE',7),('S4_REVERSAL',5)]:
            s=state[family]
            if family=='S2_DIRECTION_RUN':s=run_maps[j][s]
            add(f'F{j}_{family}',one_hot(s,count))
        if method=='M3' or j not in rule_specs:continue
        spec=rule_specs[j];family,a,b=spec['pair']
        known=(f>=spec['low'])&(f<=spec['high'])&(state['S1_DIRECTION']!=3)
        # Each beta is learned once inside TRAIN. Every unsupported state and
        # out-of-training common range gives zero conditional contribution.
        add(f'F{j}_conditional',np.column_stack([np.where(known&(state[family]==s),f,0.) for s in [a,b]]))
        if method=='M5' and j in laws:
            add(f'F{j}_self_path',path_features(bank,rows,j,laws[j]))
    if not parts:
        return np.ones((len(rows),1),dtype=np.float32),('intercept_only',)
    x=np.column_stack(parts).astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Finite prefix-observable M design required')
    return x,tuple(names)


def alias_map(candidates,rule_specs,laws):
    if not candidates:
        return {m:'M1' for m in ['M1','M2','M3','M4','M5']}
    return dict(M1='M1',M2='M2',M3='M3',M4='M4' if rule_specs else 'M3',
        M5='M5' if laws and rule_specs else 'M4' if rule_specs else 'M3')
