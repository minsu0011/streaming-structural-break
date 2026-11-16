"""TRAIN-only common support for direct own/H and graph direction contrasts."""
import numpy as np
from src.next8.screening import common_range,_state_support


def directional_derivatives(weights,anchor,state,*,cross=False,own_stratum=None):
    z=anchor/4.;base=weights.get('F_linear',0.)+2*z*weights.get('F_square',0.)
    if cross:
        if own_stratum not in (0,1,2):raise ValueError('Graph direction holds own/H condition fixed')
        base+=weights.get(f'own_state_{own_stratum}_times_F',0.)+weights.get(f'cross_state_{state}_times_F',0.)
    else:base+=weights.get(f'own_state_{state}_times_F',0.)
    if z>0:left=right=(base+weights.get('F_positive',0.))/4.
    elif z<0:left=right=(base-weights.get('F_negative',0.))/4.
    else:left=(base-weights.get('F_negative',0.))/4.;right=(base+weights.get('F_positive',0.))/4.
    direction=1 if min(left,right)>0 else -1 if max(left,right)<0 else 0
    return {'left_derivative':float(left),'right_derivative':float(right),'direction':direction,
        'at_fixed_nonlinear_kink':bool(z==0 and left!=right),
        'ambiguous_fixed_nonlinearity_not_certified_as_dynamic':direction==0}


def training_cases(f,own,cross_state,y,groups,refit,*,cross=False):
    if refit['status']!='COMPLETE':return [{'status':refit['status'],'reason':refit.get('reason','Unfitted conditional refit')}]
    f=np.asarray(f);own=np.asarray(own);cross_state=np.asarray(cross_state);y=np.asarray(y);groups=np.asarray(groups)
    defining=cross_state if cross else own;cases=[]
    for stratum in ((0,1,2) if cross else (None,)):
        base=(own==stratum) if cross else own<3
        domain=common_range(f[base],defining[base])
        if domain is not None:
            domain=[max(domain[0],refit['domain'][0]),min(domain[1],refit['domain'][1])]
            if domain[1]-domain[0]<=1e-3:domain=None
        if domain is None:
            cases.append({'status':'INSUFFICIENT_SUPPORT','own_stratum':stratum,'reason':'No common TRAIN feature range across defining states within this fixed own/H condition'});continue
        keep=base&(f>=domain[0])&(f<=domain[1])&(defining<3)
        support=_state_support(groups[keep],y[keep],defining[keep]);anchor=float(np.clip(0.,*domain))
        derivatives={str(s):directional_derivatives(refit['derivative_weights_raw_design'],anchor,s,cross=cross,own_stratum=stratum) for s in range(3)}
        pairs=[]
        for a,b in ((0,1),(2,1),(0,2)):
            supported=all(support[s]['series']>=64 and min(support[s]['positive_series'],support[s]['negative_series'])>=32 for s in (a,b))
            sa=derivatives[str(a)]['direction'];sb=derivatives[str(b)]['direction']
            pairs.append({'state_pair':[a,b],'directions':[sa,sb],'opposing_directions':sa*sb<0,
                'training_pair_supported':supported,'eligible_for_orientation_replication':bool(supported and sa*sb<0)})
        cases.append({'status':'FROZEN','own_stratum':stratum,'domain':domain,'F_anchor':anchor,
            'training_support':support,'training_derivatives':derivatives,'pairs':pairs,
            'graph_own_condition_held_fixed':cross,'boundary_policy':'At F=0 require both one-sided derivatives to share the claimed sign; no subgradient-only certification'})
    return cases
