"""Freeze three SCALE views and enqueue them only for completed ALT/H models."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    pp=BASE/'diagnostic_policy/ALT_SCALE_DIAGNOSTIC_POLICY.json'
    names=['scripts/forensic_next4_robust_splits.py','src/next4/robustness.py','src/next4/diagnostics.py',
        'src/next/bootstrap.py','src/validation/group_bootstrap.py']
    sources={name:digest(ROOT/name) for name in names}
    if not pp.exists():
        candidates=['B1_SPLIT_ENERGY_BRANCHES','B2_LOG_ENERGY_SEMITAIL',*read_json(BASE/'diagnostic_policy/CONDITIONAL_ROBUSTNESS_VALIDATION.json')['candidates']]
        atomic_json(pp,{'created_utc':utc_now(),'candidates':candidates,'sources_sha256':sources,
            'splits':['ALT_1','ALT_2','ALT_3','LEAVE_ONE_H_REGIME_OUT'],
            'scopes':['SCALE','observed_scale_decrease','observed_scale_increase'],'bootstrap_seed':20260908,'bootstrap_replicates':2000,
            'scope_definition':'Original frozen SCALE signature and reconstructed raw-window SD direction, plus original no-break controls; evaluation only.',
            'support_definition':'Report actual positive series in the scope, no-break series and folds with positive observations. Direction subsets below 100 positive series remain unsupported diagnostics.',
            'empty_scope':'Use the official scorer fallback 0.5 for an empty or pairless fold; mark zero positive support and never claim it as informative evidence.',
            'reference_caching':'One baseline per existing split/fold/scope, keyed by exact row coordinates, shared group multiplicities and policy SHA.',
            'new_fits':0,'no_primary_gate_substitution':True,
            'stop_rule':'One fixed three-scope analysis for each candidate that actually completes its preregistered ALT/H refits. No altered mechanism definitions, seeds or candidate parameters.'},immutable=True)
    else:assert read_json(pp)['sources_sha256']==sources
    for candidate in read_json(pp)['candidates']:
        job='RS_'+candidate
        if (BASE/'queue'/(job+'.json')).exists() or not (BASE/'robustness'/candidate/'RESULTS.json').exists():continue
        dependency='B401_FIXED_ALT_H_REPLICATION' if candidate=='B2_LOG_ENERGY_SEMITAIL' else 'B601_ROBUSTNESS_ALT_H' if candidate=='B1_SPLIT_ENERGY_BRANCHES' else 'RB_'+candidate
        register_job({'id':job,'kind':'validation','priority':117,'depends_on_success':[dependency],
            'argv':['scripts/forensic_next4_robust_splits.py','--candidate',candidate],
            'description':'Fixed SCALE/direction views on existing ALT/H fitted OOF; no models or new partitions',
            'retry_safe':False,'policy_sha256':digest(pp)})
        print('NEXT4 ALT SCALE QUEUED',candidate,flush=True)
    progress('ALT/H mechanism diagnostic registration','Original fixed SCALE and raw-SD directions; original primary confidence gates remain unchanged')


if __name__=='__main__':main()
