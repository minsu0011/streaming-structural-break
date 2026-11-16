"""Prospective empty-previous-law runtime repair, preserving numerical fits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now


def main():
    with activity('Versioned first-point empty-law runtime repair before any live replay execution','RUNTIME_REPAIR','B9_NORMALIZATION_STATE_AVAILABILITY'):
        loader=ROOT/'src/next9/runtime_loading.py'
        source=loader.read_text(encoding='utf-8').replace('from src.next9.streaming import FrozenHeads',
            'from src.next9.streaming import FrozenHeads\nfrom src.next9.runtime_laws import RuntimeSelfLaw,RuntimeCrossLaw')
        source=source.replace("laws[j]=joblib.load(d/'LAW.joblib')","laws[j]=RuntimeSelfLaw(joblib.load(d/'LAW.joblib'))")
        source=source.replace("cross_laws[j]=joblib.load(d/'CROSS_LAW.joblib')","cross_laws[j]=RuntimeCrossLaw(joblib.load(d/'CROSS_LAW.joblib'))")
        atomic(ROOT/'src/next9/runtime_loading_v2.py',source)
        groups=[('replay_next9_research_pool','LIVE_REPLAY_IMPLEMENTATION_LOCK'),('replay_next9_reverse_order','REVERSE_ORDER_IMPLEMENTATION_LOCK')]
        new_scripts={}
        for name,lock in groups:
            source=(ROOT/f'scripts/next9/{name}.py').read_text(encoding='utf-8').replace('src.next9.runtime_loading import','src.next9.runtime_loading_v2 import')
            source=source.replace(lock+'_NEXT9.json',lock+'_V2_NEXT9.json')
            path=f'scripts/next9/{name}_v2.py';atomic(ROOT/path,source);new_scripts[lock]=path
        added=['src/next9/runtime_loading_v2.py','src/next9/runtime_laws.py','scripts/next9/tests/test_runtime_laws.py',*new_scripts.values()]
        for name in added:compile((ROOT/name).read_text(encoding='utf-8'),name,'exec')
        for name,lockname in groups:
            old=BASE/(lockname+'_NEXT9.json');record=read_json(old)
            record.update(timestamp=utc_now(),supersedes_sha256=digest(old),
                source_sha256={**record['source_sha256'],**{p:digest(ROOT/p) for p in added}},
                repair='At first Online point, previous-issued law query is empty. Scikit classifier rejects zero rows; inference-only wrappers return0x4 and preserve first surprise0. Nonempty fixed probabilities agree to1e-14; no learned weights changed.',
                regression_test_sha256=digest(BASE/'tests/RUNTIME_LAW_FIRST_POINT_TESTS.xml'),
                before_any_live_replay_job=True,original_fits_and_sources_preserved=True)
            atomic(BASE/(lockname+'_V2_NEXT9.json'),record)
        queue=read_json(BASE/'runtime/TASK_QUEUE.json');output=[];replacements=[]
        for item in queue:
            lockname='LIVE_REPLAY_IMPLEMENTATION_LOCK' if item['id'].startswith('LIVE_RP_') else 'REVERSE_ORDER_IMPLEMENTATION_LOCK' if item['id'].startswith('REVERSE_RP_') else None
            if lockname is None:output.append(item);continue
            assert not (BASE/'runtime/queue'/item['id']).exists(),item['id']
            new={**item,'id':item['id']+'_V2','script':new_scripts[lockname],
                'source_sha256':digest(ROOT/new_scripts[lockname]),
                'gates':[g.replace(lockname+'_NEXT9.json',lockname+'_V2_NEXT9.json') for g in item['gates']]}
            output.append(new);replacements.append(dict(original=item,replacement=new))
        assert len(replacements)==40
        atomic(BASE/'repairs/FIRST_POINT_EMPTY_LAW_RUNTIME_REPAIR_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
            diagnosed_before_actual_job_failure=True,original_T2_self_and_cross_zero_row_errors_reproduced_in_regression=True,
            corrected_first_surprise_zero_and_finite=True,nonempty_probability_max_tolerance=1e-14,
            no_scientific_selection_or_fitted_parameter_changes=True,old_sources_and_unexecuted_registrations_preserved=True,
            initial_shell_construction_error='PowerShell quoting rejected an inline copy command; checked that none of3targetfiles existed. Replaced with this structured Python script.',
            replacements=replacements))
        atomic(BASE/'registrations/LIVE_AND_REVERSE_REPLAY_TASKS_V2.json',dict(timestamp=utc_now(),tasks=[r['replacement'] for r in replacements]))
        atomic(BASE/'runtime/TASK_QUEUE.json',output,immutable=False)


if __name__=='__main__':main()
