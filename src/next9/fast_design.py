"""Exact float32 empty-rule M3/M5 schema, with unchanged fallback branches."""
import numpy as np
from numba import njit
from src.next9.joint_design import assemble


@njit(cache=False)
def empty_rule_design(past,current,epsilon,available,candidates,run_maps):
    out=np.zeros((1,len(candidates)*43),dtype=np.float32)
    for i in range(len(candidates)):
        j=candidates[i];offset=i*43;f=np.float64(current[j])
        out[0,offset]=f;out[0,offset+1]=abs(f)
        out[0,offset+2]=f if f>0. else 0.;out[0,offset+3]=f if f<0. else 0.;out[0,offset+4]=f*f
        lags=(1,2,4,8,16)
        for k in range(5):out[0,offset+5+k]=past[17-lags[k],j]
        v=np.empty(8,dtype=np.float64)
        for k in range(8):v[k]=np.float64(past[9+k,j])*(k-3.5)
        slope=(((v[0]+v[1])+(v[2]+v[3]))+((v[4]+v[5])+(v[6]+v[7])))/42.
        e=np.float64(epsilon[j]);mov=np.empty(15,dtype=np.int8)
        prior=3;age=15;to=3
        for k in range(15):
            delta=np.float64(past[k+2,j])-np.float64(past[k+1,j])
            direction=2 if delta>e else 0 if delta < -e else 1
            mov[k]=direction
            if direction!=1:
                if prior!=3 and prior!=direction:age=14-k;to=direction
                prior=direction
        last=mov[14];run=1
        for k in range(13,-1,-1):
            if mov[k]!=last:break
            run+=1
        out[0,offset+10]=slope
        out[0,offset+11]=np.float32(past[16,j]-past[15,j])
        out[0,offset+12]=run;out[0,offset+13]=age
        bucket=0 if run==1 else 1 if run<=3 else 2
        s1=last if available else 3
        s2=last*3+bucket if available else 9
        s3=last*2+int(abs(slope)>e) if available else 6
        recent=age<=3 and to==last
        s4=(3 if recent else 0) if last==2 else (2 if recent else 1) if last==0 else 4
        if not available:s4=4
        out[0,offset+14+s1]=1.
        out[0,offset+18+run_maps[i,s2]]=1.
        out[0,offset+31+s3]=1.
        out[0,offset+38+s4]=1.
    return out


def frozen_design(heads,bank):
    s=heads.spec
    if s['rule_specs'] or not s['candidates']:
        return assemble(bank,np.asarray([bank.time]),s['candidates'],
            {int(k):v for k,v in s['rule_specs'].items()},
            {int(k):np.asarray(v,dtype=int) for k,v in s['run_maps'].items()},heads.self_laws,'M5')
    if not hasattr(heads,'_empty_rule_layout'):
        candidates=np.asarray(s['candidates'],dtype=np.int64)
        maps=np.asarray([s['run_maps'][str(j)] for j in candidates],dtype=np.int64)
        assert maps.shape==(len(candidates),13) and np.all((maps>=0)&(maps<13))
        names=[]
        for j in candidates:
            for suffix,count in [('fixed',5),('own',9),('S1_DIRECTION',4),('S2_DIRECTION_RUN',13),('S3_DIRECTION_SLOPE',7),('S4_REVERSAL',5)]:
                names.extend(f'F{j}_{suffix}_{k}' for k in range(count))
        assert tuple(names)==tuple(s['input_names'])
        heads._empty_rule_layout=(candidates,maps,tuple(names))
    candidates,maps,names=heads._empty_rule_layout
    assert bank.value is not None and np.isfinite(bank.value).all()
    return empty_rule_design(bank.past,bank.value,bank.series['epsilon'][0,0],
                             bool(bank.series['H_length'][0]>=160),candidates,maps),names
