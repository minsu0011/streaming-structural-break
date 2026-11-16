"""Freeze ResearchPool-safe timing sample and a finite balanced artifact order."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next5.guard import ResearchGuard
from src.next5.scheduler import register_job


def main():
    old=read_json(ROOT/'configs/next_latency_policy.json');api=read_json(BASE/'runtime_validation/API_POLICY_LOCK.json');guard=ResearchGuard()
    ids=[sid for sid in old['series_ids'] if sid in guard.research_ids];guard.admit(ids,purpose='Freeze timing membership before loading observations')
    candidates=list(api['candidates']);labels=['FROZEN_PRIMARY','ResearchPool_BL80',*candidates]
    orders=[labels,list(reversed(labels)),labels[4:]+labels[:4]]
    directory=ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1'
    sources=('scripts/profile_next5_runtime.py','src/next5/runtime.py','src/next5/streaming.py','src/next5/synthetic.py',
        'src/next2/shared_inference.py','src/next2/shared_blend_runtime.py','src/next5/data.py','src/next5/guard.py')
    path=BASE/'runtime_validation/LATENCY_POLICY_LOCK.json'
    atomic_json(path,{'created_utc':utc_now(),'status':'FROZEN_BEFORE_NEXT5_TIMING','candidates':candidates,'series_ids':ids,
        'sample_policy':'Keep original latency-sample order, intersect by ID with the frozen ResearchPool before any observation loading. No replacement, new seed or outcome-driven selection.',
        'original_sample_series':len(old['series_ids']),'research_sample_series':len(ids),'original_policy_sha256':digest(ROOT/'configs/next_latency_policy.json'),
        'continuous_points':old['continuous_points'],'block_points':old['block_points'],'measured_checkpoint_points':old['measured_checkpoint_points'],
        'block_orders':orders,'primary_point_gate':old['primary_point_gate'],'amortized_gate':old['amortized_gate'],
        'continuous_stream':old['continuous_stream'],'isolation':old['isolation'],
        'scope':'Already-fitted PRIMARY fold0 ResearchPool artifacts and unchanged historical frozen PRIMARY. Timing is not a held-out quality estimate.',
        'baseline_conservatism':'Historical frozen direct state is timed as published; new scalar adapter also checks input/output validity. No synthetic overhead is added to slow the baseline.',
        'frozen_primary_directory':directory.relative_to(ROOT).as_posix(),'frozen_primary_model_sha256':digest(directory/'model.sbnext2'),
        'exact_primary_OOF_required_for_timing_to_support_promotion':True,'no_parameters_or_repetition_changes_after_timing':True,
        'sources_sha256':{name:digest(ROOT/name) for name in sources},'new_fits':0,'shadow_rows':0,'seal_rows':0},immutable=True)
    register_job({'id':'LATENCY_FIXED_ARTIFACT_BLOCKS','kind':'validation','priority':88,
        'depends_on_success':['API_'+cid for cid in candidates],'argv':['scripts/profile_next5_runtime.py'],
        'description':'Three finite balanced blocks of frozen PRIMARY, ResearchPool BL80 and seven candidates: actual 10M scalar updates each, same ResearchPool-only sample','retry_safe':False})
    progress('Finite runtime study preregistered',str(len(ids))+' original sample IDs remain inside ResearchPool; three fixed balanced orders and no new model fits')
    print('NEXT5 RUNTIME SAMPLE',len(ids),'models',len(labels),'actual continuous updates',len(labels)*3*old['continuous_points'],flush=True)


if __name__=='__main__':main()
