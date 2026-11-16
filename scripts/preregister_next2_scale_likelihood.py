"""Five source-motivated scale-down likelihood contrasts, no learner tuning."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.registered_extensions import register_extension,registration_hash
from src.next.extensions import extension_hash
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    path=ROOT/'configs/next2/SCALE_LIKELIHOOD_CANDIDATES_V1.json'
    if path.exists():
        raise RuntimeError('Scale likelihood menu already frozen')
    recipes=[('S201_SCALE_LIKELIHOOD',{'channel_pair':'raw_forecast','cap':0,'include_parent':'none'}),
        ('S202_SCALE_LIKELIHOOD_CAP4',{'channel_pair':'raw_forecast','cap':4,'include_parent':'none'}),
        ('S203_CONDITIONAL_SCALE_LIKELIHOOD',{'channel_pair':'raw_conditional','cap':0,'include_parent':'none'}),
        ('S204_PARENT_PLUS_LOWER',{'channel_pair':'raw_forecast','cap':0,'include_parent':'lower'}),
        ('S205_PARENT_PLUS_LOG_RATIOS',{'channel_pair':'raw_forecast','cap':0,'include_parent':'ratios'})]
    candidates=[]
    for eid,settings in recipes:
        ext={'kind':'next2_scale_likelihood','settings':settings}
        registration={'kind':ext['kind'],'module':'next2_scale_likelihood'}
        register_extension(ext,registration)
        candidates.append({'experiment_id':eid,'parent_id':'Y001_LONG_VARIANCE_ARCH1','generation':'NEXT2_SCALE_V1',
            'phase':'SCALE_DOWN_LIKELIHOOD','hypothesis':'Directional Gaussian variance likelihood may recover decreases obscured by standardized squared-energy mean evidence.',
            'change':settings,'counterexample_target':'variance decrease / ARCH / isolated outlier',
            'engine_config':{'order':8,'normalization':'sd','cdf_bins':16},'groups':['NEXT2_SCALE_LIKELIHOOD'],
            'base':'arch8','base_family_selection':'ABCD','learner':'M4_lightgbm','weighting':'W3','params':{},
            'extension':ext,'extension_frozen_hash':extension_hash(ext),'extension_registration':registration,
            'registration_source_sha256':registration_hash()})
    write_json(path,{'version':'NEXT2_SCALE_V1','created_utc':utc_now(),
        'policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),
        'hypothesis_sha256':sha256(ROOT/'docs/NEXT2_SCALE_LIKELIHOOD_HYPOTHESIS.md'),
        'candidates':candidates,'score_inspected_before_freeze':False})
    print('NEXT2 SCALE LIKELIHOOD FROZEN',len(candidates),flush=True)


if __name__=='__main__':
    main()
