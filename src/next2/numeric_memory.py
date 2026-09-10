"""Deduplicated retained NumPy allocations, with explicit shared/model boundaries."""
import numpy as np


def inventory(value):
    found={};visited=set()
    def visit(item,path):
        if id(item) in visited:
            return
        visited.add(id(item))
        if isinstance(item,np.ndarray):
            owner=item
            while isinstance(owner.base,np.ndarray):
                owner=owner.base
            key=(int(owner.__array_interface__['data'][0]),int(owner.nbytes))
            if not owner.nbytes:
                return
            if key not in found:
                found[key]={'owner':owner,'bytes':int(owner.nbytes),'paths':[],
                    'dtype':str(owner.dtype),'shape':list(owner.shape),'numpy_owns_data':bool(owner.flags.owndata)}
            found[key]['paths'].append(path)
        elif isinstance(item,(tuple,list)):
            for j,v in enumerate(item):visit(v,path+'/'+str(j))
        elif isinstance(item,dict):
            for k,v in item.items():visit(v,path+'/'+str(k))
    visit(value,'root')
    return found


def prepared_arrays(prepared):
    if hasattr(prepared,'primary') and hasattr(prepared,'complement'):
        return (prepared_arrays(prepared.primary),prepared_arrays(prepared.complement))
    return {k:v for k,v in vars(prepared).items() if k!='model' and isinstance(v,(np.ndarray,tuple,list))}


def model_arrays(model):
    if hasattr(model,'primary') and hasattr(model,'complement'):
        return (model_arrays(model.primary),model_arrays(model.complement))
    return model.compact.args(),model.columns


def state_arguments(state):
    if hasattr(state,'primary') and hasattr(state,'complement'):
        return state_arguments(state.primary),state_arguments(state.complement)
    return state.call_args[:-1]


def accounting(state):
    state_alloc=inventory(state_arguments(state))
    shared=inventory(prepared_arrays(state.prepared))
    model=inventory((prepared_arrays(state.prepared),model_arrays(state.prepared.model)))
    private=set(state_alloc)-set(shared)
    return {'retained_state_numeric_bytes_including_shared_selectors':sum(v['bytes'] for v in state_alloc.values()),
        'private_per_series_numeric_bytes':sum(state_alloc[k]['bytes'] for k in private),
        'shared_selector_bytes_reachable_from_state':sum(state_alloc[k]['bytes'] for k in set(state_alloc)&set(shared)),
        'model_and_prepared_shared_numeric_bytes':sum(v['bytes'] for v in model.values()),
        'total_one_stream_plus_one_model_numeric_bytes':sum(v['bytes'] for v in {**model,**state_alloc}.values()),
        'private_allocation_count':len(private),'state_allocation_count':len(state_alloc)},state_alloc,private
