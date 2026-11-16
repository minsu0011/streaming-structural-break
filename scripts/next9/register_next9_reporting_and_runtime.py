from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

with activity('Freeze17field cards,20question reporting and matched scalar latency protocol','REPORT_RUNTIME_PROTOCOL','B9_NORMALIZATION_STATE_AVAILABILITY'):
    paths=['scripts/next9/write_next9_scientific_reports.py','src/next9/feature_meanings.py','scripts/next9/benchmark_next9_latency.py']
    for path in paths:compile((ROOT/path).read_text(encoding='utf-8'),path,'exec')
    lock='REPORT_RUNTIME_IMPLEMENTATION_LOCK_NEXT9.json'
    atomic(BASE/lock,dict(status='FROZEN',timestamp=utc_now(),source_sha256={p:digest(ROOT/p) for p in paths},
        report='Evidence-derived Korean17field cards for every A candidate;20explicit A/B/C answers, frozen-center decomposition,8path summaries',
        latency='After owned workers quiesce,10excludedfold/mode procedures x24fixed NONE paths x512points. Alternate BL/M7 order eachpoint;32warmup observations excluded. Bothmedian<=3andp99<=5perprocedure. No speed-driven model reselection.',
        latency_run_by_controller_after_queue_complete=True))
    script=paths[0];gates=[lock,'SCIENTIFIC_RESULTS_NEXT9.json','figure_data/DONE.json']+[f'center_decomposition/PART_{p}_DONE.json' for p in range(4)]
    task=dict(id='WRITE_SCIENTIFIC_REPORTS',status='READY',script=script,source_sha256=digest(ROOT/script),arguments=[],gates=gates,
        phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True)
    queue=read_json(BASE/'runtime/TASK_QUEUE.json');assert not any(x['id']==task['id'] for x in queue)
    atomic(BASE/'registrations/SCIENTIFIC_REPORT_TASK_NEXT9.json',dict(timestamp=utc_now(),task=task))
    atomic(BASE/'runtime/TASK_QUEUE.json',queue+[task],immutable=False)
