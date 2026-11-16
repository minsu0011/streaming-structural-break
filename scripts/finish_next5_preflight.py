"""Close preflight only after actual guard, ResearchPool loader and cold checks."""
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next5.guard import ResearchGuard


def main():
    out=BASE/'preflight';guard=ResearchGuard(ROOT)
    for name in ('PF001_GUARD_SCORER_TESTS','PF002_RESEARCH_RAW','PF013_BL80_COLD_RECOVERY','PF014_B2_COLD_RECOVERY','PF015_D2_COLD_RECOVERY'):
        r=read_json(BASE/'jobs'/name/'STATE.json');assert r['status']=='COMPLETE' and r['exit_code']==0
    tree=ET.parse(out/'GUARD_SCORER_TESTS.xml');cases=tree.findall('.//testcase')
    assert cases and not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
    comparisons=[]
    for name,original in [('BL80','artifacts/next4/cold_api/BL80_1.json'),('B2','artifacts/next4/cold_api_readonly_v2/B2_1.json'),
                          ('D2','artifacts/next4/finalist_validation/D2_DISCRETE_SCALE_MIXTURE/COLD/cold_1.json')]:
        r=read_json(out/(name+'_COLD.json'));old=read_json(ROOT/original)
        assert r['status']=='PASS' and r['prediction_sha256']==old['prediction_sha256']
        comparisons.append({'model':name,'synthetic_cold_prediction_sha256':r['prediction_sha256'],'current_receipt_sha256':digest(out/(name+'_COLD.json')),'reference_receipt_sha256':digest(ROOT/original)})
    pointer=read_json(BASE/'data/RESEARCH_POINTER.json');assert pointer['series']==6400 and pointer['shadow_rows']==pointer['seal_rows']==0
    guard.check_state()
    result={'status':'PASS','created_utc':utc_now(),'tests':len(cases),'test_failures':0,'test_errors':0,'test_skips':0,
        'guard_scorer_test_xml_sha256':digest(out/'GUARD_SCORER_TESTS.xml'),'cold_reference_checks':comparisons,
        'research_data':pointer,'frozen_references_sha256':digest(out/'FROZEN_REFERENCES.json'),
        'shadow_labels_loaded':False,'all_seals_unopened':True,'new_reduced_uses':0,'new_public_submissions':0,'new_cloud_submissions':0,
        'new_fits':0,'source_sha256':digest(Path(__file__)),
        'cold_invocation_failure_preserved_and_resolved':True,'recovery_receipt_sha256':digest(out/'COLD_FAILURE_RECOVERY.json'),
        'old_full_DEV_inference_replayed_in_NEXT5':False,
        'reference_replay_scope':'Synthetic cold inference only, so no prospective Shadow features or outcomes are read.'}
    atomic_json(out/'NEXT5_PREFLIGHT.json',result,immutable=True);atomic_json(ROOT/'NEXT5_PREFLIGHT.json',result,immutable=True)
    old=read_json(ROOT/'NEXT5_SHADOW_GUARD.json');shutil.copyfile(ROOT/'NEXT5_SHADOW_GUARD.json',out/'SHADOW_GUARD_BEFORE_TESTS.json')
    atomic_json(ROOT/'NEXT5_SHADOW_GUARD.json',{**old,'status':'PASS','completed_tests':len(cases),'finalized_utc':utc_now(),
        'preflight_sha256':digest(out/'NEXT5_PREFLIGHT.json'),'labels_before_ID_admission_rejected':True})
    session=read_json(BASE/'SESSION.json');session.update(modeling_blocked=False,modeling_block_reason=None,preflight_completed_utc=utc_now())
    atomic_json(BASE/'SESSION.json',session)
    progress('NEXT5 preflight passed',str(len(cases))+' tests, three cold references and ResearchPool-only raw cache',phase='PREFLIGHT')
    print('NEXT5 PREFLIGHT COMPLETE',len(cases),'tests',flush=True)


if __name__=='__main__':main()
