"""Explicit retained numeric-memory boundaries for the nested NEXT3 runtime."""
from src.next2.numeric_memory import inventory,prepared_arrays,model_arrays


def accounting(state):
    from src.next3.precomputed_shared_runtime import PrecomputedSharedDetector
    if not isinstance(state,PrecomputedSharedDetector):
        from src.next2.numeric_memory import accounting as original_accounting
        return original_accounting(state)
    args=state.call_args
    state_alloc=inventory((args[0][:-1],*args[1:]))
    shared=inventory((prepared_arrays(state.prepared),state.prepared.dependence_table,state.prepared.dependence_logs))
    model=inventory((prepared_arrays(state.prepared),model_arrays(state.prepared.model),
        state.prepared.dependence_table,state.prepared.dependence_logs))
    private=set(state_alloc)-set(shared)
    return {'retained_state_numeric_bytes_including_shared_selectors':sum(v['bytes'] for v in state_alloc.values()),
        'private_per_series_numeric_bytes':sum(state_alloc[k]['bytes'] for k in private),
        'shared_selector_bytes_reachable_from_state':sum(state_alloc[k]['bytes'] for k in set(state_alloc)&set(shared)),
        'model_and_prepared_shared_numeric_bytes':sum(v['bytes'] for v in model.values()),
        'total_one_stream_plus_one_model_numeric_bytes':sum(v['bytes'] for v in {**model,**state_alloc}.values()),
        'private_allocation_count':len(private),'state_allocation_count':len(state_alloc)},state_alloc,private
