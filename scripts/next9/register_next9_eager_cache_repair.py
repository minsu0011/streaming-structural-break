"""Admit preinitialized immutable schema cache, preserving every failed attempt."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic as _atomic,read_json,digest,utc_now,activity


def atomic(path,value,**kwargs):
    path=Path(path)
    if path.exists() and path.suffix=='.py' and isinstance(value,str):
        assert path.read_text(encoding='utf-8')==value,'Unexpected changed generated source: '+str(path)
        compile(value,str(path),'exec')
        return
    if path.suffix=='.py' and isinstance(value,str):compile(value,str(path),'exec')
    return _atomic(path,value,**kwargs)


def main():
    proof=read_json(BASE/'EAGER_SCHEMA_CACHE_QUALIFICATION_NEXT9.json');assert proof['status']=='PASS'
    for p,sha in proof['source_sha256'].items():assert digest(ROOT/p)==sha
    q=read_json(BASE/'runtime/TASK_QUEUE.json')
    mapping={'replay_next9_research_pool_v5':'replay_next9_research_pool_v6',
             'replay_next9_synthetic_portable_v5':'replay_next9_synthetic_portable_v6'}
    # Existing LIVE workers finish naturally, preserving successful per-series
    # predictions. No running source or task identity is ever changed.
    for item in q:
        if Path(item['script']).stem not in mapping:continue
        directory=BASE/'runtime/queue'/item['id']
        if directory.exists():assert (directory/'TERMINAL.json').exists(),'Still running: '+item['id']
    with activity('Admit diagnosed eager schema cache, preserve failed runs, and bind final independent statistical validation',
                  'SOFTWARE_REPAIR','B9_NORMALIZATION_STATE_AVAILABILITY'):
        added=['src/next9/runtime_loading_v6.py']
        for old,new in mapping.items():
            text=(ROOT/f'scripts/next9/{old}.py').read_text(encoding='utf-8')
            text=text.replace('src.next9.runtime_loading_v5','src.next9.runtime_loading_v6')
            text=text.replace('LIVE_REPLAY_IMPLEMENTATION_LOCK_V5_NEXT9.json','LIVE_REPLAY_IMPLEMENTATION_LOCK_V6_NEXT9.json')
            atomic(ROOT/f'scripts/next9/{new}.py',text);added.append(f'scripts/next9/{new}.py')
        source=(ROOT/'src/next9/repair_resolution.py').read_text(encoding='utf-8')
        old="""        else:
            match=re.fullmatch(r'STATE_TRANSPORT_F([0-4])(?:_FULL_HELDOUT_V2)?',name)"""
        new="""        elif re.fullmatch(r'LIVE_RP_F[0-4]_(NORMAL|EARLY)_P[01]_V2_META_V3_MEMBERSHIP_V4_FAST_V5',name):
            replacement=name+'_CACHE_V6'
            receipt='EAGER_SCHEMA_CACHE_QUALIFICATION_NEXT9.json'
        else:
            match=re.fullmatch(r'STATE_TRANSPORT_F([0-4])(?:_FULL_HELDOUT_V2)?',name)"""
        assert old in source;source=source.replace(old,new)
        atomic(ROOT/'src/next9/repair_resolution_v2.py',source)
        source=(ROOT/'src/next9/delivery_payload_v4.py').read_text(encoding='utf-8')
        assert "trees=['" in source
        source=source.replace("trees=['","trees=['pair_kernel_audit','",1)
        atomic(ROOT/'src/next9/delivery_payload_v5.py',source)
        supplements={'admit_next9_final_audit_v5':'admit_next9_final_audit_v6',
                     'audit_next9_execution_v3':'audit_next9_execution_v4',
                     'finalize_next9_reports_v8':'finalize_next9_reports_v9',
                     'reproduce_next9_archive_v7':'reproduce_next9_archive_v8'}
        for old,new in supplements.items():
            source=(ROOT/f'scripts/next9/{old}.py').read_text(encoding='utf-8')
            source=source.replace('src.next9.repair_resolution import','src.next9.repair_resolution_v2 import')
            source=source.replace('src.next9.delivery_payload_v4','src.next9.delivery_payload_v5')
            source=source.replace('replay_next9_synthetic_portable_v5.py','replay_next9_synthetic_portable_v6.py')
            source=source.replace('reproduce_next9_archive_v7','reproduce_next9_archive_v8')
            if old=='admit_next9_final_audit_v5':
                token="    assert read_json(BASE/'LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json')['status']=='PASS'"
                assert token in source
                source=source.replace(token,token+"\n    assert read_json(BASE/'EAGER_SCHEMA_CACHE_QUALIFICATION_NEXT9.json')['status']=='PASS'\n    assert read_json(BASE/'INDEPENDENT_TS_AUC_AUDIT_NEXT9.json')['status']=='PASS'")
            if old=='finalize_next9_reports_v8':
                source=source.replace("required=['","required=['EAGER_SCHEMA_CACHE_QUALIFICATION_NEXT9.json','INDEPENDENT_TS_AUC_AUDIT_NEXT9.json','",1)
                source=source.replace('현재 추론 진입점은',
                    '입력 스키마 캐시를 첫 호출 전에 초기화한 v6 loader로 replay 객체 해시를 검증했다. 모든 10개 절차에서 학습 파라미터와 예측 불변을 별도 검사했다. TS-AUC는 정렬 순위와 다른 명시적 관측 쌍 계산으로 모든 도메인·2000회 bootstrap을 교차 검증했다.\n\n현재 공개 추론 진입점은')
            compile(source,new,'exec');atomic(ROOT/f'scripts/next9/{new}.py',source)
        added+=['src/next9/repair_resolution_v2.py','src/next9/delivery_payload_v5.py',*[f'scripts/next9/{v}.py' for v in supplements.values()]]
        for family in ['LIVE_REPLAY','PORTABLE_FIXTURE']:
            old=BASE/f'{family}_IMPLEMENTATION_LOCK_V5_NEXT9.json';r=read_json(old)
            r.update(timestamp=utc_now(),supersedes_sha256=digest(old),
                source_sha256={**r['source_sha256'],**{p:digest(ROOT/p) for p in added}},
                cache_qualification_sha256=digest(BASE/'EAGER_SCHEMA_CACHE_QUALIFICATION_NEXT9.json'))
            atomic(BASE/f'{family}_IMPLEMENTATION_LOCK_V6_NEXT9.json',r)
        out=[];changes=[]
        for item in q:
            stem=Path(item['script']).stem
            if stem not in mapping:out.append(item);continue
            terminal=BASE/'runtime/queue'/item['id']/'TERMINAL.json'
            if terminal.exists() and read_json(terminal)['status']=='COMPLETE':out.append(item);continue
            script=f'scripts/next9/{mapping[stem]}.py'
            gates=[g.replace('_IMPLEMENTATION_LOCK_V5_', '_IMPLEMENTATION_LOCK_V6_') for g in item['gates']]
            replacement={**item,'id':item['id']+'_CACHE_V6','script':script,'source_sha256':digest(ROOT/script),'gates':gates}
            assert not (BASE/'runtime/queue'/replacement['id']).exists()
            out.append(replacement);changes.append(dict(original=item,replacement=replacement))
        atomic(BASE/'repairs/EAGER_SCHEMA_CACHE_REPAIR_NEXT9.json',dict(status='PASS',timestamp=utc_now(),changes=changes,
            source_sha256={p:digest(ROOT/p) for p in added},qualification_sha256=digest(BASE/'EAGER_SCHEMA_CACHE_QUALIFICATION_NEXT9.json'),
            original_failed_logs_and_successful_series_preserved=True,learned_parameters_and_predictions_unchanged=True,
            explanation='The first-call derived schema cache changed whole-head pickle bytes. Initialize the identical deterministic layout before taking the full immutable-object snapshot; retain full-object hashing with no excluded fields.',
            final_controllers=supplements))
        atomic(BASE/'runtime/TASK_QUEUE.json',out,immutable=False)
        atomic(BASE/'EAGER_SCHEMA_CACHE_REPLAY_ADMITTED_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
            repair_sha256=digest(BASE/'repairs/EAGER_SCHEMA_CACHE_REPAIR_NEXT9.json')))
        print(dict(status='PASS',versioned_tasks=len(changes)))


if __name__=='__main__':main()
