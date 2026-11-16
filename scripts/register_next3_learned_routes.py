"""Freeze exact learned-gate recipes and their nested base training plan."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,atomic_json,digest,utc_now,progress


def main():
    shared={'created_utc':utc_now(),'BOTTLENECK':'B1',
        'HYPOTHESIS':'A very small causal route trained only on inner base OOF can allocate C205 without post-hoc mechanism labels.',
        'EXPECTED_EFFECT':'Improve incremental between-series ranking beyond fixed 80/20 while preserving supported mechanisms.',
        'RISKS':'Surrogate log loss is not official TS-AUC; cross-fit distribution shift and reused DEV selection can negate gains.',
        'PRIMARY_METRIC':'Official grouped TS-AUC relative to frozen 80/20',
        'SECONDARY_DIAGNOSTIC':'SCALE/DEPENDENCE slices, bounded weights, ALT/H, nonlinear calibration and runtime',
        'STOP_RULE':'Preflight and nested lineage must pass before fit. Failed screen never receives full CV.',
        'policy_sha256':digest(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json'),
        'sources_sha256':{name:digest(ROOT/'src/next3'/name) for name in ('learned_router.py','nested.py','router.py','frozen.py')},
        'inner_partition_rule':'Every available original fold within outer train is held out once; no outer validation rows/groups enter any inner fit.',
        'base_recipe':'Frozen J102/C205 100 trees, W3; identical base inner OOF reused across learned routers',
        'finalist_admission':{'mean_delta_min':-.0008,'median_delta_min':-.001,'worst_delta_min':-.001},
        'gate_training_inputs':['J102_score','C205_score','scale_strength','dependence_strength','raw_energy_current','forecast_current'],
        'posthoc_mechanism_inputs_or_targets':False}
    atomic_json(BASE/'hypotheses/N001_NESTED_BASE_OOF.json',{**shared,'CHANGE':'Build reusable honest inner base OOF; no new candidate scoring in this task'},immutable=True)
    details={
        'R2_LOGISTIC_GATE':{'CHANGE':'w=.3*sigmoid(beta*train-standardized six features), predicted mixture (1-w)J+wC',
            'target':'DEV binary point labels, W3 computed from outer gate training rows only',
            'loss':'weighted mixture binary log loss plus .01 ridge, intercept unpenalized',
            'solver':'L-BFGS-B, maxiter100, ftol1e-10, gtol1e-6, maxls30',
            'initial':'intercept log(2), all other coefficients zero; no alternate starts or hyperparameter sweep',
            'standardization':'Outer gate training means/SD only, SD floor 1e-6, standardized feature clip [-8,8]'},
        'R3_DEPTH2_GATE':{'CHANGE':'Decision tree depth2, minimum leaf 5% of training rows, random_state0',
            'target':'C205 inner-OOF point log loss lower than J102 inner-OOF point log loss',
            'weight':'W3 times max(abs(C205 loss-J102 loss),1e-8)',
            'output':'w=.3*leaf probability(C205 better), nearest .1 with half-up rounding; w in {0,.1,.2,.3}'}}
    for name,detail in details.items():
        atomic_json(BASE/'hypotheses'/f'{name}.json',{**shared,'kind':name,**detail},immutable=True)
    progress('Learned routes preregistered','Logistic/depth2 exact settings and grouped inner-OOF lineage frozen before base fits')
    print('NEXT3 LEARNED ROUTES REGISTERED',flush=True)


if __name__=='__main__':main()
