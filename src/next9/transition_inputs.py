"""Prefix state input assembly, including previous-issued transition law."""
import numpy as np
from src.next9.transitions import summaries


def state_inputs(bank,rows,j):
    states=bank.pre_state(rows,j)
    return states,bank.series['context'][bank.series_index[rows]]


def probabilities(law,states,context):
    return law.probabilities(states['S1_DIRECTION'],states['direction_run_capped15'],states['slope8'],context)


def path_features(bank,rows,j,law):
    rows=np.asarray(rows,dtype=np.int64)
    state,context=state_inputs(bank,rows,j)
    current=probabilities(law,state,context)
    previous=np.tile(law.table.mean(axis=0),(len(rows),1))
    available=bank.times[rows]>=1
    pstate,pcontext=state_inputs(bank,rows[available]-1,j)
    previous[available]=probabilities(law,pstate,pcontext)
    epsilon=bank.series['epsilon'][bank.series_index[rows],0,j]
    result=summaries(state['S1_DIRECTION'],state['direction_run_capped15'],state['reversal_age_capped15'],
        state['slope8'],epsilon,previous,current)
    # At first Online point no previous Online law was issued. Expose no
    # surprise contribution there, rather than pretending an H-seed prediction.
    result[~available,2]=0.
    return result
