"""Register two fully specified heuristic routes before scoring either."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next3.router import HEURISTICS,GATE_NAMES


def main():
    for kind in HEURISTICS:
        atomic_json(BASE/'hypotheses'/f'{kind}.json', {
            'created_utc':utc_now(),'HYPOTHESIS':(
                'Use dependence specialist when its causal score exceeds the primary; fall back toward J102 otherwise.'
                if kind=='R1_SCORE_GAP' else
                'Compare standardized local dependence and variance evidence to reduce specialist weight during scale-dominated points.'),
            'BOTTLENECK':'B1','CHANGE':kind,'EXPECTED_EFFECT':'Retain dependence gain while limiting scale dilution; overall competition score remains primary.',
            'RISKS':'Different model calibration and noisy current evidence can distort between-series ranking; reused DEV selection remains conditional.',
            'PRIMARY_METRIC':'Original grouped TS-AUC relative to matched frozen 80/20',
            'SECONDARY_DIAGNOSTIC':'Supported mechanism-plus-null scores, pair contribution, weight frequency, time/age and synthetic robustness',
            'STOP_RULE':'Use immutable NEXT3 screen gate; failed screen never receives full CV.',
            'kind':kind,'inputs':GATE_NAMES,'learned_parameters':0,'new_base_fits':0,
            'implementation_sha256':digest(ROOT/'src/next3/router.py'),
            'policy_sha256':digest(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json'),
            'finalist_admission':{'mean_delta_min':-.0008,'median_delta_min':-.001,'worst_delta_min':-.001},
            'formula':('w=0/.1/.2/.3 for C-J <=-.1 / (-.1,0] / (0,.1] / >.1'
                if kind=='R1_SCORE_GAP' else 'r=dep/(1+scale); w=0 when dep<1; otherwise w=0/.1/.2/.3 for r<=1 / (1,2] / (2,4] / >4'),
            'scientific_feature_change':False,'posthoc_mechanism_used_for_training_or_routing':False},immutable=True)
    document = '''# NEXT3 causal router protocol

Frozen J102/C205 80/20 is R0. Only current-point component scores and already causal local evidence enter the router. Mechanism signatures, tau, ID, fold and future length are prohibited inputs.

R1_SCORE_GAP uses weights 0, 0.1, 0.2, 0.3 at the fixed C205−J102 cutoffs −0.1, 0, +0.1. R1_EVIDENCE_RATIO uses maximum absolute dependence EWMA8/32 across four lags divided by one plus maximum absolute raw-energy/forecast EWMA8/32. Dependence strength below 1 falls back to zero specialist weight; fixed ratio cutoffs are 1, 2, 4. Both preserve float32 component inputs and final float32 blending.

Stage S reuses exactly the original 3,000 screen IDs, folds 0 and 4, and matched frozen base-model OOF predictions. Mean delta versus 80/20 must be at least −0.001 and each fold delta at least −0.004. Only passing heuristics proceed to original full five-fold OOF. These heuristics fit no meta-model and do not inspect validation targets to set parameters.

Any later logistic/depth-2/residual combiner must first generate grouped inner base OOF entirely inside outer training. Original outer validation series and historical groups are absent from every inner/base/router training set. Simply training a gate on existing full five-fold OOF and scoring those same rows is forbidden. Learned menus require their own immutable exact formula/solver registration before results.

Candidate quality is measured against frozen 80/20, with J102 retained as a secondary scientific comparator. Conditional bootstrap does not adjust repeated DEV adaptation. Mechanism/age diagnostics are evaluation-only; no oracle gate is nominated. Failed screen receipts remain preserved.
'''
    with (ROOT/'ROUTER_PROTOCOL.md').open('x',encoding='utf-8') as stream:
        stream.write(document)
    progress('Two heuristic routes registered','Exact formulas and screen/full admission fixed before scoring')
    print('NEXT3 HEURISTIC ROUTERS REGISTERED',flush=True)


if __name__=='__main__':main()
