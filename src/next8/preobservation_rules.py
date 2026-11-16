"""Read the state-dependent linear rule before accepting the next point."""
import numpy as np
from src.next8.runtime_head import conditions
from src.next8.joint_models import preselected_linear_rule,raw_coefficients


def preview(state):
    args=state.model_args;original=args[0]
    # Only two scratch state outputs are replaced. All learned maps and tables
    # are read-only here; neither the feature engines nor Online counter move.
    scratch=(*original[:-2],original[-2].copy(),original[-1].copy())
    conditions(args[3],state.counter,scratch)
    method=state.prepared.method;model=state.prepared.model;records=[]
    linear=method in ('M4','M5')
    weights=dict(zip(model.names,raw_coefficients(model)[0])) if linear else {}
    for k,j in enumerate(original[0]):
        own=int(scratch[-2][k]);cross=int(scratch[-1][k]);j=int(j);prefix=f'F{j:03d}_'
        fixed={name:float(weights.get(prefix+name,0.)) for name in ('variable_positive','variable_negative','variable_square')}
        beta=preselected_linear_rule(model,j,own,cross) if linear else None
        records.append({'feature':j,'own_or_H_condition':own,'forecast_graph_condition':cross,
            'linear_beta_per_normalized_F':beta,'linear_component_direction':int(np.sign(beta)) if beta is not None else None,
            'fixed_nonlinear_weights':fixed,'complete_local_derivative_known_before_F':bool(linear and not any(fixed.values())),
            'current_F_observed':False,'future_state_confirmation_observed':False,
            'UNKNOWN_current_F_contribution_zero':own==3,'graph_UNKNOWN_cross_contribution_zero':cross==3})
    return {'time_online':state.counter,'actual_frozen_method':method,'rules':records,
        'linear_rule_available_before_current_observation':linear,
        'fixed_nonlinear_response_is_not_dynamic_orientation':True,
        'preview_changes_runtime_buffers':False}
