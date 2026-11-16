from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now

q=read_json(BASE/'runtime/TASK_QUEUE.json')
script='scripts/next9/qualify_next9_eager_cache.py'
task=dict(id='QUALIFY_EAGER_SCHEMA_CACHE',status='READY',script=script,source_sha256=digest(ROOT/script),arguments=[],
          gates=[],phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True)
assert not any(r['id']==task['id'] for r in q)
gated=[]
for r in q:
    if r['script'].endswith(('replay_next9_research_pool_v5.py','replay_next9_synthetic_portable_v5.py')) and not (BASE/'runtime/queue'/r['id']).exists():
        r={**r,'gates':[*r['gates'],'EAGER_SCHEMA_CACHE_REPLAY_ADMITTED_NEXT9.json']};gated.append(r['id'])
atomic(BASE/'registrations/EAGER_SCHEMA_CACHE_DIAGNOSIS_NEXT9.json',dict(timestamp=utc_now(),task=task,gated_unstarted=gated,
    reason='Full-object pickle includes first-call derived schema cache; independently test fitted immutability before admission.'))
# Rebuild rather than relying on a loop-local replacement.
out=[{**r,'gates':[*r['gates'],'EAGER_SCHEMA_CACHE_REPLAY_ADMITTED_NEXT9.json']} if r['id'] in gated else r for r in q]
atomic(BASE/'runtime/TASK_QUEUE.json',[task,*out],immutable=False)

fig=BASE/'FIGURES'
files=['FIG1_153_MOVEMENT_ATLAS.png','FIG2_PERSISTENCE_CENTER_OWN_V2.png',
       'FIG3_ACTUAL_PRECURRENT_TRAJECTORIES.png','FIG4_MATCHED_CURRENT_F_ORIENTATION.png',
       'FIG5_SELF_TRANSITION_MATRICES.png','FIG6_TRANSITION_STATE_TIMELINE.png',
       'FIG7_CONDITIONAL_STATE_COEFFICIENTS.png','FIG8_FROZEN_SELF_AND_CROSS_GRAPH.png',
       'FIG9_MATCHED_MODEL_COMPARISON.png','FIG10_TIME_H_AND_EARLY_LATE.png']
atomic(fig/'ALL_TEN_VISUALLY_INSPECTED_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
    files_sha256={p:digest(fig/p) for p in files},combined_pdf_sha256=digest(fig/'NEXT9_TEN_FIGURES.pdf'),
    inspection='Actual rendered images viewed. Labels, ranges, legends and scientific absence statements readable; no clipped panels.',
    notes=['FIG2 uses previously inspected V2 with corrected legend placement.',
           'FIG3 and FIG6 retain fixed actual series 8 even with flat F72; no visually selected substitute.',
           'FIG7 absence is explicitly not an estimated zero effect; FIG8 has zero cross edges.',
           'FIG9 shows frozen held-out results and negative M7 delta; FIG10 retains wide early-time uncertainty.']))
print(dict(status='REGISTERED',gated=len(gated)))
