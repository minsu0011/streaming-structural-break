"""Generated frozen arithmetic for E003; see LEAN_NUMERIC_BUILD.json."""
from functools import lru_cache
import numpy as np
from numba import njit, typeof
from numba.experimental import jitclass
from src.features.config import CONFIG
from src.features.historical_calibration import CalibrationPolicy
from src.features.ar_order_input import historical_filter
from src.features.arch_input import historical_variance_filter, arch_one
from src.features.ar_residual_input import innovation_one
from src.features.streaming import StreamingFeatureState, feature_names
from src.next.whitening import fit_historical, innovation_update
from src.next.channel_evidence import make_channel_args, _logadd
from src.next.serial_channel_evidence import historical_correlations, sum_variances, serial_channel_step
from src.next2.complement_bank import normalized_one, historical_raw, raw_complement
from src.next3.packed_trees import packed_tree_margin

OLD_CONFIG = CONFIG.variant(normalization='median_mad', scales=(5,20,160))
OLD_POLICY = CalibrationPolicy(method='historical_median_mad')


# Extracted from src/next/fast_variance_initialization.py :: conditional_replay_only
@njit(cache=False)
def conditional_replay_only(residuals,parameters,initial_variance):
    conditional = np.full(3,initial_variance)
    for residual in residuals:
        centered = residual-parameters[0]
        squared = min(centered*centered,parameters[5])
        conditional[1] = conditional[0]
        conditional[0] = squared
        conditional[2] = (1.-parameters[7])*conditional[2]+parameters[7]*squared
    return conditional


# Extracted from src/next/fast_variance_initialization.py :: fast_variance_parameters
def fast_variance_parameters(residuals,mode):
    if mode not in ('arch1','arch2','ewma'):
        raise ValueError('Fast reference only supports the three variance modes')
    r = np.asarray(residuals,dtype=np.float64)
    mean = float(r.mean())
    sd = max(float(r.std()),1e-8)
    sq = (r-mean)**2
    cap = max(float(np.quantile(sq,.99)),1e-12)
    capped = np.minimum(sq,cap)
    variance = max(float(capped.mean()),1e-12)
    slopes = np.zeros(2)
    if mode in ('arch1','arch2'):
        order = 1 if mode=='arch1' else 2
        design = np.column_stack([capped[order-1-j:len(r)-1-j] for j in range(order)])
        target = capped[order:]
        center = design-design.mean(axis=0)
        gram = center.T@center
        ridge = max(float(np.trace(gram))/order*.001,1e-12)
        slopes[:order] = np.maximum(np.linalg.solve(gram+ridge*np.eye(order),center.T@(target-target.mean())),0.)
        if slopes.sum()>.95:
            slopes *= .95/slopes.sum()
    parameters = np.array([mean,sd,variance*(1.-slopes.sum()),slopes[0],slopes[1],cap,max(variance*.0001,1e-12),2./33.])
    conditional = conditional_replay_only(r,parameters,variance)
    return parameters,conditional


# Extracted from src/next2/pruned_runtime.py :: pair_channel_update
@njit(cache=False)
def pair_channel_update(residual, parameters, conditional, mode, output):
    h_variance = parameters[2]/max(1.-parameters[3]-parameters[4],.05)
    q1,q2 = conditional[0],conditional[1]
    variance = conditional[2] if mode == 5 else parameters[2]+parameters[3]*q1+parameters[4]*q2
    variance = max(variance,parameters[6])
    centered = residual-parameters[0]
    squared = centered*centered
    output[0] = squared/h_variance-1.
    output[1] = np.log(variance/h_variance)
    capped = min(squared,parameters[5])
    conditional[1] = q1
    conditional[0] = capped
    conditional[2] = (1.-parameters[7])*conditional[2]+parameters[7]*capped
    return output


# Extracted from src/next2/pruned_runtime.py :: historical_pair
@njit(cache=False)
def historical_pair(residuals, parameters, mode):
    h_variance = parameters[2]/max(1.-parameters[3]-parameters[4],.05)
    conditional = np.full(3,h_variance)
    result = np.empty((len(residuals),2))
    for j in range(len(residuals)):
        pair_channel_update(residuals[j],parameters,conditional,mode,result[j])
    return result


# Extracted from src/next2/dependence_runtime.py :: historical_z
@njit(cache=False)
def historical_z(residuals,parameters,variance):
    state=np.full(3,variance)
    z=np.empty(len(residuals))
    for j in range(len(z)):
        z[j]=normalized_one(residuals[j],parameters,state)
    return z,state


# Extracted from src/next2/dependence_runtime.py :: evidence
@njit(cache=False)
def evidence(values,ages,widths,ewma,cusum,cumulative,prefix,counter,output):
    t=counter[0]+1
    for channel in range(4):
        value=min(max(values[channel],-12.),12.)
        cursor=channel*6
        for k,half in enumerate((8.,32.)):
            alpha=1.-np.exp(-np.log(2.)/half)
            ewma[channel,k]=(1.-alpha)*ewma[channel,k]+alpha*value
            variance=max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-12)
            output[cursor+k]=ewma[channel,k]/np.sqrt(variance)
        cusum[channel,0]=max(0.,cusum[channel,0]+value-.25)
        cusum[channel,1]=max(0.,cusum[channel,1]-value-.25)
        output[cursor+2]=np.log1p(cusum[channel,0])
        output[cursor+3]=np.log1p(cusum[channel,1])
        cumulative[channel]+=value
        b025,b1,width=-1e300,-1e300,0.
        for k in range(len(ages)):
            age=ages[k]
            if age>t:
                break
            width+=widths[k]
            total=cumulative[channel]-prefix[(t-age)%len(prefix),channel]
            energy=total*total
            b025=_logadd(b025,np.log(widths[k])-.5*np.log1p(.25*age)+.5*.25*energy/(1.+.25*age))
            b1=_logadd(b1,np.log(widths[k])-.5*np.log1p(age)+.5*energy/(1.+age))
        prefix[t%len(prefix),channel]=cumulative[channel]
        output[cursor+4]=b025-np.log(width)
        output[cursor+5]=b1-np.log(width)
    counter[0]=t
    return output


# Extracted from src/next2/used_feature_runtime.py :: dependencies
def dependencies(columns):
    pairs=set()
    for column in columns:
        if column<6:
            continue
        scale,kind=divmod(int(column)-6,15)
        needs={0:(0,),1:(0,),2:(1,),3:(0,1),4:(2,),5:(3,),6:(4,),7:(5,),
            8:(12,13,14,15,16),9:(6,),10:(7,),11:(8,),12:(9,),13:(10,),14:(11,)}[kind]
        pairs.update((scale,j) for j in needs)
    return np.asarray(sorted(pairs),dtype=np.int64).reshape(-1,2)


# Extracted from src/next2/used_feature_runtime.py :: used_old_update
@njit(cache=False)
def used_old_update(point,reference,baseline,thresholds,ring,ew,counters,parameters,alphas,lags,pairs,lookup,columns,work,output):
    clip_z,drift,decay,moment_floor,rate_floor,tail_low,tail_high=parameters
    x=reference[0] if not np.isfinite(point) else point
    z=min(max((x-reference[0])/reference[1],-clip_z),clip_z)
    t=int(counters[0])+1
    counters[0]=t
    # Only requested current summaries are updated or transformed.
    for column in columns:
        if column==3:
            counters[1]=max(0.,counters[1]+z-drift)
        elif column==4:
            counters[2]=max(0.,counters[2]-z-drift)
        elif column==5:
            counters[3]+=(z-counters[3])/t
    for k in range(len(pairs)):
        scale,stat=pairs[k]
        if stat==0:
            value=z
        elif stat==1:
            value=z*z
        elif stat==2:
            value=abs(z)
        elif stat==3:
            value=1. if abs(z)>tail_low else 0.
        elif stat==4:
            value=1. if abs(z)>tail_high else 0.
        elif stat==5:
            value=1. if z>0. else 0.
        elif stat<10:
            value=z*ring[(int(counters[4])-lags[stat-6])%len(ring)]
        elif stat<12:
            error=z-reference[4]*ring[(int(counters[4])-1)%len(ring)]
            value=error if stat==10 else error*error
        else:
            cell=0
            for boundary in thresholds:
                cell+=int(z>boundary)
            value=1. if cell==stat-12 else 0.
        ew[k]+=alphas[scale]*(value-ew[k])
    for scale in range(3):
        counters[5+scale]=(1-alphas[scale])*counters[5+scale]+1.
    for k,column in enumerate(columns):
        if column==0:
            output[k]=z
        elif column==1:
            output[k]=abs(z)
        elif column==2:
            output[k]=min(max((x-reference[2])/reference[3],-clip_z),clip_z)
        elif column==3:
            output[k]=np.log1p(counters[1])
        elif column==4:
            output[k]=np.log1p(counters[2])
        elif column==5:
            output[k]=counters[3]
        else:
            scale,kind=divmod(column-6,15)
            root_n=np.sqrt(counters[5+scale])
            if kind==0 or kind==1:
                value=(ew[lookup[scale,0]]-baseline[0])*root_n
                output[k]=value if kind==0 else abs(value)
            elif kind==2:
                output[k]=np.log(max(ew[lookup[scale,1]],moment_floor)/max(baseline[1],moment_floor))
            elif kind==3:
                output[k]=np.log(max(ew[lookup[scale,1]]-ew[lookup[scale,0]]**2,moment_floor)/max(baseline[1]-baseline[0]**2,moment_floor))
            elif kind==4:
                output[k]=(ew[lookup[scale,2]]-baseline[2])*root_n
            elif kind<=7:
                stat=kind-2
                rate=baseline[stat]
                output[k]=(ew[lookup[scale,stat]]-rate)*root_n/np.sqrt(max(rate*(1-rate),rate_floor))
            elif kind==8:
                for j in range(5):
                    work[j]=abs(ew[lookup[scale,12+j]]-baseline[12+j])
                output[k]=np.sum(work)*root_n
            elif kind<=12:
                stat=kind-3
                output[k]=(ew[lookup[scale,stat]]-baseline[stat])*root_n
            elif kind==13:
                output[k]=(ew[lookup[scale,10]]-baseline[10])*root_n/np.sqrt(max(baseline[11],moment_floor))
            else:
                output[k]=np.log(max(ew[lookup[scale,11]],moment_floor)/max(baseline[11],moment_floor))
    ring[int(counters[4])%len(ring)]=z
    counters[4]+=1
    return output


# Extracted from src/next2/used_feature_runtime.py :: replay_old
@njit(cache=False)
def replay_old(points,args):
    result=np.empty((len(points),len(args[-1])),dtype=np.float32)
    for k in range(len(points)):
        result[k]=used_old_update(float(points[k]),*args)
    return result


# Extracted from src/next2/used_feature_runtime.py :: UsedOldState
class UsedOldState:
    def __init__(self,historical,columns):
        ar_ref,coefficients,ar_ring,residual=historical_filter(historical,OLD_CONFIG,8)
        ar_args=(ar_ref,coefficients,ar_ring,np.zeros(1,dtype=np.int64))
        vparams,previous,transformed=historical_variance_filter(residual,OLD_CONFIG)
        original=StreamingFeatureState(transformed,config=OLD_CONFIG)
        columns=np.asarray(columns,dtype=np.int64)
        if not len(columns) or np.any((columns<0)|(columns>=51)) or len(np.unique(columns))!=len(columns):
            raise ValueError('Unique used old ABCD columns required')
        pairs=dependencies(columns)
        lookup=np.full((3,17),-1,dtype=np.int64)
        for k,(scale,stat) in enumerate(pairs):
            lookup[scale,stat]=k
        ew=np.asarray([original.ew[scale,stat] for scale,stat in pairs],dtype=np.float64)
        counters=np.zeros(8)
        counters[4]=len(original.ring)
        args=(original.reference,original.baseline,original.thresholds,original.ring,ew,counters,
            original.parameters,original.alphas,original.lags,pairs,lookup,columns,np.empty(5),np.empty(len(columns),dtype=np.float32))
        temporary=tuple(a.copy() if isinstance(a,np.ndarray) else a for a in args)
        matrix=replay_old(np.asarray(transformed,dtype=np.float32),temporary)
        if not len(matrix):
            matrix=np.zeros((1,len(columns)),dtype=np.float32)
        burn=min(OLD_POLICY.burn_in,max(0,len(matrix)//2))
        values=np.ascontiguousarray(matrix[burn:].T,dtype=np.float64)
        center=np.median(values,axis=1)
        scale=np.maximum(np.median(np.abs(values-center[:,None]),axis=1)*1.4826,OLD_POLICY.standard_deviation_floor)
        mask=~np.isin(columns,[3,4,5])
        self.call_args=(ar_args,(vparams,previous),args,center,scale,mask,float(OLD_POLICY.clip),np.empty(len(columns),dtype=np.float32))
        self.ewma_cells=len(ew)


# Extracted from src/next2/used_feature_runtime.py :: used_arch_step
@njit(cache=False)
def used_arch_step(point,args):
    ar_args,variance_args,old_args,center,scale,mask,clip,output=args
    residual=innovation_one(point,*ar_args)
    value=arch_one(float(residual),*variance_args)
    raw=used_old_update(float(value),*old_args)
    for j in range(len(raw)):
        output[j]=min(max((raw[j]-center[j])/scale[j],-clip),clip) if mask[j] else raw[j]
    return output


# Extracted from src/next2/shared_blend_runtime.py :: shared_step
@njit(cache=False)
def shared_step(point, filter_args, parameters, conditional, pair_calibration, pair_work,
                pair_evidence, age_variances, ring, counter, center, calibration, work, empty, dependence_evidence):
    residual = innovation_update(point, *filter_args)
    # normalized_one's pre-update variance and centered residual, without a
    # second mutation of the recurrence shared by both branches.
    variance = max(parameters[2]+parameters[3]*conditional[0]+parameters[4]*conditional[1], parameters[6])
    centered = residual-parameters[0]
    z = min(max(centered/np.sqrt(variance), -4.), 4.)
    pair_channel_update(residual, parameters, conditional, 3, pair_work)
    for j in range(2):
        pair_work[j] = (pair_work[j]-pair_calibration[0, j])/pair_calibration[1, j]
    left = serial_channel_step(pair_work, pair_evidence, age_variances)
    raw_complement(z, empty, ring, counter, center, 3, work)
    for j in range(4):
        work[j] = (work[j]-calibration[0, j])/calibration[1, j]
    right = evidence(work, *dependence_evidence)
    return left, right


# Extracted from src/next2/shared_blend_runtime.py :: SharedChannels
class SharedChannels:
    def __init__(self, historical):
        fit = fit_historical(historical, 8)
        residuals = fit['historical_innovations']
        parameters, conditional = fast_variance_parameters(residuals, 'arch1')
        pair = historical_pair(residuals, parameters, 3)
        pair = pair[min(160, len(pair)//4):]
        pair_calibration = np.stack([pair.mean(axis=0), np.maximum(pair.std(axis=0), 1e-6)])
        pair_evidence = make_channel_args(2, 512)
        correlations = historical_correlations((pair-pair_calibration[0])/pair_calibration[1])
        age_variances = sum_variances(pair_evidence[0], correlations)
        # Do not initialize C205's H replay by inverting omega/(1-alpha).
        variance = max(float(np.minimum((residuals-parameters[0])**2, parameters[5]).mean()), 1e-12)
        z, replayed = historical_z(residuals, parameters, variance)
        np.testing.assert_array_equal(replayed, conditional)
        center = np.asarray([z.mean(), (z*z).mean(), np.abs(z).mean()])
        empty = np.empty(0)
        raw, ring, counter = historical_raw(z, empty, center, 3, 4)
        raw = raw[min(160, len(raw)//4):]
        calibration = np.stack([raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-6)])
        ages = np.asarray([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int64)
        dependence_evidence = (ages, np.diff(np.r_[0, ages]).astype(float),
            np.zeros((4, 2)), np.zeros((4, 2)), np.zeros(4), np.zeros((129, 4)),
            np.zeros(1, dtype=np.int64), np.empty(24, dtype=np.float32))
        filter_args = tuple(fit[k] for k in ('reference', 'coefficients', 'ring', 'counter'))
        self.call_args = (filter_args, parameters, conditional, pair_calibration, np.empty(2),
            pair_evidence, age_variances, ring, counter, center, calibration, np.empty(4), empty, dependence_evidence)
        self.pair_output = pair_evidence[-1]
        self.dependence_output = dependence_evidence[-1]

    def update(self, point):
        return shared_step(float(point), *self.call_args)


# Extracted from src/next2/shared_blend_runtime.py :: shared_feature_step
@njit(cache=False)
def shared_feature_step(point, args):
    old_args, shared_args, old_positions, new_positions, new_columns, output, models = args
    old = used_arch_step(point, old_args)
    new, dependence = shared_step(point, *shared_args)
    for j in range(len(old_positions)):
        output[old_positions[j]] = old[j]
    for j in range(len(new_positions)):
        output[new_positions[j]] = new[new_columns[j]]
    return output


# Extracted from src/next3/packed_shared_runtime.py :: packed_model_step
@njit(cache=False)
def packed_model_step(args):
    old_args, shared_args, old_positions, new_positions, new_columns, output, models = args
    primary_tree, complement_tree, weight = models
    margin_a = packed_tree_margin(output, *primary_tree)
    margin_b = packed_tree_margin(shared_args[-1][-1], *complement_tree)
    a = np.float64(np.float32(1./(1.+np.exp(-margin_a))))
    b = np.float64(np.float32(1./(1.+np.exp(-margin_b))))
    return np.float32(weight*a+(1.-weight)*b)


# Extracted from src/next3/packed_shared_runtime.py :: packed_kernel
@njit(cache=False)
def packed_kernel(point, args):
    shared_feature_step(point, args)
    return packed_model_step(args)


# Extracted from src/next/fused_runtime.py :: _resident_class
@lru_cache(maxsize=32)
def _resident_class(kernel,argument_type):
    @jitclass([('args',argument_type)])
    class ResidentNext:
        def __init__(self,args):
            self.args = args
        def predict_one(self,point):
            return kernel(point,self.args)
    return ResidentNext


# Extracted from src/next/fused_runtime.py :: unique_array_bytes
def unique_array_bytes(value):
    seen = set()
    def visit(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item,np.ndarray):
            return item.nbytes
        if isinstance(item,(tuple,list)):
            return sum(visit(v) for v in item)
        return 0
    return visit(value)
