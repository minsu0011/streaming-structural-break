"""Numeric causal state and coordinate evaluation for frozen joint models."""
import re
import numpy as np
from numba import njit
from src.models.compact_trees import tree_margin,export_histgb
from src.next8.h_context import HStateSpec
from src.next8.features import UNKNOWN


@njit(cache=True)
def state_of(value,j,low,high,valid):
    if not valid[j]:return 3
    return 0 if value<low[j] else 2 if value>high[j] else 1


@njit(cache=True)
def conditions(history,t,args):
    variables,kinds,H_states,low,high,valid,parent_ids,parent_count,prior,own_table,own_support,pair_table,pair_support,own,cross=args
    for k,j in enumerate(variables):
        own[k]=state_of(history[(t-1)%32,j],j,low,high,valid) if kinds[k]==0 else H_states[k]
        cross[k]=3
        if t<1 or parent_count[k]==0 or not valid[j]:continue
        a=state_of(history[(t-1)%32,j],j,low,high,valid)
        if a==3 or own_support[k,a]<64:continue
        logs=np.empty(3);base=own_table[k,a]
        for s in range(3):logs[s]=np.log(base[s])
        active=0
        for m in range(parent_count[k]):
            p=parent_ids[k,m]
            b=state_of(history[(t-2)%32,p],p,low,high,valid)
            if b==3 or pair_support[k,m,a,b]<64:continue
            for s in range(3):
                ratio=min(max(pair_table[k,m,a,b,s]/base[s],.25),4.)
                logs[s]+=np.log(ratio)
            active+=1
        if active<1:continue
        maximum=max(logs[0],max(logs[1],logs[2]));total=0.
        for s in range(3):logs[s]=np.exp(logs[s]-maximum);total+=logs[s]
        top=np.float32(-1.);second=np.float32(-1.);winner=0
        for s in range(3):
            p=np.float32(logs[s]/total)
            if p>top:second=top;top=p;winner=s
            elif p>second:second=p
        if top>=np.float32(.60) and np.float32(top-second)>=np.float32(.15):cross[k]=winner


@njit(cache=True)
def coordinates(values,history,t,context,own,cross,args):
    codes,features,parameters,variable_index,output=args
    for k in range(len(codes)):
        op=codes[k];j=features[k];param=parameters[k];v=variable_index[k]
        result=np.float32(0.)
        if op==0:result=np.float32(1.)
        elif op==5:result=context[param] if param<6 else np.float32(np.log1p(t))
        elif op==6:result=np.float32(history[(t-param)%32,j]/np.float32(4.))
        else:
            f=np.float32(values[j]/np.float32(4.))
            if op in (7,8,9,10,12,13) and v>=0 and own[v]==3:f=np.float32(0.)
            if op in (1,7):result=f
            elif op in (2,8):result=f if f>0 else np.float32(0.)
            elif op in (3,9):result=-f if f<0 else np.float32(0.)
            elif op in (4,10):result=np.float32(f*f)
            elif op==11:result=np.float32(own[v]==param)
            elif op==12:result=f if own[v]==param else np.float32(0.)
            elif op==13:result=f if cross[v]==param else np.float32(0.)
        output[k]=result
    return output


@njit(cache=True)
def score(values,args):
    kind,center,scale,coefficient,intercept,tree,work,constant,trace=args
    if kind==0:
        trace[0]=0.;trace[1]=constant
        return np.float32(constant)
    for k in range(len(values)):work[k]=(np.float64(values[k])-center[k])/scale[k]
    if kind==2:margin=tree_margin(work,*tree)
    else:
        margin=intercept
        for k in range(len(values)):margin+=work[k]*coefficient[k]
    if margin>=0:p=1./(1.+np.exp(-margin))
    else:
        e=np.exp(margin);p=e/(1.+e)
    trace[0]=margin;trace[1]=p
    return np.float32(p)


def dummy_tree():
    return (np.array([0],dtype=np.int32),np.array([-1],dtype=np.int32),np.zeros(1),
        np.array([-1],dtype=np.int32),np.array([-1],dtype=np.int32),np.zeros(1),np.zeros(1,dtype=bool),0.)


def head_arguments(model):
    tree=dummy_tree();n=len(model.names);constant=0.;coefficient=np.zeros(n);intercept=0.
    if model.constant_probabilities is not None:kind=0;constant=float(model.constant_probabilities[1])
    elif hasattr(model.estimator,'coef_'):
        kind=1;coefficient=model.estimator.coef_[0].astype(np.float64).copy();intercept=float(model.estimator.intercept_[0])
    else:kind=2;tree=export_histgb(model.estimator).args()
    return (kind,model.center.copy(),model.scale.copy(),coefficient,intercept,tree,np.zeros(n),constant,np.full(2,np.nan))


def coordinate_arguments(model,selection,method):
    variables=list(map(int,selection['variable_features']));codes=[];features=[];parameters=[];variable_index=[];used=set()
    operations={'stable_linear':1,'stable_positive':2,'stable_negative':3,'stable_square':4,
                'variable_linear':7,'variable_positive':8,'variable_negative':9,'variable_square':10}
    for name in model.names:
        op=0;j=0;param=0;v=-1
        if name=='constant_context':pass
        elif name.startswith('known_H_time_context_'):op=5;param=int(name.rsplit('_',1)[1])
        else:
            match=re.fullmatch(r'F(\d{3})_(.+)',name)
            if match is None:raise ValueError('Unknown frozen runtime coordinate '+name)
            j=int(match.group(1));tail=match.group(2);used.add(j)
            if j in variables:v=variables.index(j)
            if tail in operations:op=operations[tail]
            elif tail.startswith('own_lag'):op=6;param=int(tail[7:])
            else:
                match=re.fullmatch(r'(own|graph)_condition_(\d)(_times_F)?',tail)
                if match is None:raise ValueError('Unknown state coordinate '+name)
                param=int(match.group(2));op=13 if match.group(1)=='graph' else 12 if match.group(3) else 11
            # M2 has fixed nonlinear F evidence even when a selected H/own
            # orientation condition abstains. It does not use conditional gating.
            if method=='M2' and op in (7,8,9,10):v=-1
        codes.append(op);features.append(j);parameters.append(param);variable_index.append(v)
    return (np.asarray(codes,dtype=np.int32),np.asarray(features,dtype=np.int32),
        np.asarray(parameters,dtype=np.int32),np.asarray(variable_index,dtype=np.int32),np.zeros(len(codes),dtype=np.float32)),used


def state_arguments(feature_state,selection,graph_models,*,use_graph=True):
    variables=np.asarray(selection['variable_features'],dtype=np.int32);n=len(variables)
    kinds=np.zeros(n,dtype=np.int8);H_states=np.full(n,3,dtype=np.int8)
    parents=np.full((n,2),-1,dtype=np.int32);counts=np.zeros(n,dtype=np.int32)
    prior=np.full((n,3),1/3);own_table=np.full((n,3,3),1/3);own_support=np.zeros((n,3),dtype=np.int32)
    pair_table=np.full((n,2,3,3,3),1/3);pair_support=np.zeros((n,2,3,3),dtype=np.int32)
    required=set()
    for k,j in enumerate(variables):
        spec=selection['orientation_state_specs'][str(j)]
        if spec['kind']!='OWN_F':
            kinds[k]=1;estimator=HStateSpec(**spec)
            H_states[k]=int(estimator.states_from_context(feature_state.context,feature_state.valid[j]))
        else:required.add(int(j))
        model=graph_models.get(int(j)) if use_graph else None
        if model is not None:
            if len(model.parents)>2:raise ValueError('Cross parent runtime budget')
            required.add(int(j));required.update(map(int,model.parents));counts[k]=len(model.parents)
            parents[k,:len(model.parents)]=model.parents;prior[k]=model.prior
            own_table[k]=model.own_table;own_support[k]=model.own_support
            for m in range(len(model.parents)):pair_table[k,m]=model.pair_tables[m];pair_support[k,m]=model.pair_support[m]
    args=(variables,kinds,H_states,feature_state.low.copy(),feature_state.high.copy(),feature_state.valid.copy(),
        parents,counts,prior,own_table,own_support,pair_table,pair_support,np.full(n,3,dtype=np.int8),np.full(n,3,dtype=np.int8))
    return args,required
