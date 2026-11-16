"""Precommitted multihorizon eligibility; no feature chosen from break score."""
import math


def holm(p_values):
    result={}
    previous=0.
    ordered=sorted(p_values,key=lambda k:(p_values[k],str(k)))
    for i,key in enumerate(ordered):
        p=p_values[key]
        if not math.isfinite(p) or not 0<=p<=1:
            raise ValueError('Valid p-value required')
        adjusted=min(1.,max(previous,p*(len(ordered)-i)))
        result[key]=adjusted;previous=adjusted
    return result


def panel_pass(records):
    if sorted(r['horizon'] for r in records)!=[1,4,16]:
        raise ValueError('Every horizon required; cannot pick a single best horizon')
    supported=all(r['groups']>=64 for r in records)
    no_negative=all(r[k]['mean_gain']>=0 for r in records for k in ['persistence_contrast','H_center_contrast'])
    improved=sum(all(r[k]['mean_gain']>0 for k in ['persistence_contrast','H_center_contrast']) for r in records)>=2
    return supported and no_negative and improved


def mode_pass(panels,pooled):
    if len(panels)!=3 or sorted(r['horizon'] for r in pooled)!=[1,4,16]:
        raise ValueError('Exactly three original selection panels and horizons')
    replicate=sum(panel_pass(p) for p in panels)>=2
    good_horizons=sum(all(r[k]['mean_gain']>0 and r[k]['holm_adjusted_p']<.05 for k in ['persistence_contrast','H_center_contrast']) for r in pooled)
    no_negative=all(r[k]['mean_gain']>=0 for r in pooled for k in ['persistence_contrast','H_center_contrast'])
    return replicate and no_negative and good_horizons>=2


def select(records,early_only=False):
    eligible=[]
    for r in records:
        modes=['EARLY_SELECT'] if early_only else ['NORMAL','EARLY_LATE']
        if all(mode_pass(r['panels'][m],r['pooled'][m]) for m in modes):
            score=min(v['H_center_contrast']['mean_gain']/max(v['H_center_MSE'],1e-8) for m in modes for v in r['pooled'][m])
            eligible.append((score,int(r['feature'])))
    eligible.sort(key=lambda v:(-v[0],v[1]))
    return [j for score,j in eligible[:12]],len(eligible)
