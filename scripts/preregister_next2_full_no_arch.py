"""Source-audit-driven whole-pipeline ARCH-off control, before any scoring."""
from pathlib import Path
import sys
import json
import copy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.registered_extensions import register_extension,registration_hash
from src.next.extensions import extension_hash,names_and_groups
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    target=ROOT/'configs/next2/FULL_NO_ARCH_CANDIDATE_V1.json'
    if target.exists():
        print('NEXT2 WHOLE ARCH-OFF LOCK RETAINED',flush=True)
        return
    parent=json.loads((ROOT/'artifacts/next/experiments/Y001_LONG_VARIANCE_ARCH1/RESULTS.json').read_text())['candidate']
    candidate=copy.deepcopy(parent)
    extension={'kind':'next2_full_no_arch','settings':{'version':'full_pipeline_no_arch_v1'}}
    registration={'kind':'next2_full_no_arch','module':'next2_full_no_arch'}
    register_extension(extension,registration)
    candidate.update(experiment_id='D218_FULL_PIPELINE_NO_ARCH',parent_id='Y001_LONG_VARIANCE_ARCH1',generation='NEXT2_WHOLE_COMPONENT_CONTROL',
        hypothesis='Removing conditional ARCH normalization from both legacy ABCD and new variance branches tests the complete ARCH family; D208 removed it only from the new branch.',
        change='Both branches retain their original AR(8) whitening, but neither fits ARCH slopes nor divides observations by a time-varying conditional variance.',
        counterexample_target='ARCH robustness and exact component attribution',base='none',groups=['NEXT2_FULL_NO_ARCH'],
        extension=extension,extension_registration=registration,extension_frozen_hash=extension_hash(extension),
        registration_source_sha256=registration_hash(),keep_names=names_and_groups(extension)[0])
    candidate.pop('base_family_selection',None)
    candidate['phase']='SOURCE_AUDIT_WHOLE_COMPONENT_CONTROL'
    write_json(target,{'created_utc':utc_now(),'policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),
        'scope':'One source-audit-driven complete ARCH-family removal, same fixed screen/full gates.',
        'candidates':[candidate]})
    print('NEXT2 WHOLE ARCH-OFF CANDIDATE LOCKED',flush=True)


if __name__=='__main__':
    main()
