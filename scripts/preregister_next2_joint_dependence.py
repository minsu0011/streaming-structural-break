"""Two small feature integrations, motivated by the admitted dependence signal."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.registered_extensions import register_extension,registration_hash
from src.next.extensions import extension_hash
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    path=ROOT/'configs/next2/JOINT_DEPENDENCE_CANDIDATES_V1.json'
    if path.exists():
        raise RuntimeError('Joint dependence hypotheses already frozen')
    candidates=[]
    for eid,summary in [('J201_JOINT_DEPENDENCE_EWMA32','ewma32'),('J202_JOINT_DEPENDENCE_BAYES025','bayes025')]:
        ext={'kind':'next2_joint_dependence','settings':{'summary':summary}}
        reg={'kind':ext['kind'],'module':'next2_joint_dependence'}
        register_extension(ext,reg)
        candidates.append({'experiment_id':eid,'parent_id':'J102_SERIAL_VARIANCE_512','generation':'NEXT2_JOINT_V1',
            'phase':'ADAPTIVE_SMALL_COMPLEMENT_INTEGRATION',
            'hypothesis':'Four conditional dependence summaries may exploit the admitted complementary signal with one learner.',
            'change':'Add lag 1/2/4/8 '+summary+' features to exact J102 statistics',
            'counterexample_target':'dependence / mixed changes','engine_config':{'order':8,'normalization':'sd','cdf_bins':16},
            'groups':['NEXT2_JOINT_DEPENDENCE'],'base':'arch8','base_family_selection':'ABCD',
            'learner':'M4_lightgbm','weighting':'W3','params':{},'extension':ext,
            'extension_frozen_hash':extension_hash(ext),'extension_registration':reg,
            'registration_source_sha256':registration_hash()})
    write_json(path,{'version':'NEXT2_JOINT_V1','created_utc':utc_now(),
        'policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),
        'hypothesis_sha256':sha256(ROOT/'docs/NEXT2_JOINT_DEPENDENCE_HYPOTHESIS.md'),
        'prior_observations_sha256':{name:sha256(ROOT/f'artifacts/next2/experiments/{name}/RESULTS.json') for name in ('C205_DEPENDENCE','BL2_C205_DEPENDENCE_80')},
        'candidates':candidates,'adaptive_nomination':True,'new_candidate_predictions_inspected_before_freeze':False})
    print('NEXT2 JOINT DEPENDENCE FROZEN',len(candidates),flush=True)


if __name__=='__main__':
    main()
