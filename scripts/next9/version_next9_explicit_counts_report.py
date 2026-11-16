"""Preserve frozen v1 and register explicit numeric EARLY/raw-pattern answers."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now

old=ROOT/'scripts/next9/write_next9_scientific_reports.py'
new='scripts/next9/write_next9_scientific_reports_v2.py'
queue=read_json(BASE/'runtime/TASK_QUEUE.json');task=next(r for r in queue if r['id']=='WRITE_SCIENTIFIC_REPORTS')
assert not (BASE/'runtime/queue'/task['id']).exists()
source=old.read_text(encoding='utf-8');atomic(ROOT/new,source)
added="""        raw4features=len({key[0] for key,value in raw.items() if len(value)>=4})
        early_counts=defaultdict(int)
        for fold in range(5):
            for j in read_json(BASE/'candidates'/f'outer_{fold}'/'FROZEN.json')['strict_early_candidates']:early_counts[j]+=1
        early_repeated=sum(j in candidates and count>=4 for j,count in early_counts.items())
"""
source=source.replace(added,'')
source=source.replace("f'NORMAL A 후보 중 엄격한 EARLY 후보 선정에도 4/5개 이상 outer-TRAIN 절차에서 선정된 피처는 {early_repeated}개다. 초기 A 기준은 EARLY→LATE 보조 예측 유지도 요구했다. 이 후보선정 결과와 별도 outer 보조 예측 확인은 구분한다.'", "f'엄격한 EARLY 후보 선정과 원래 EARLY→LATE 보조 예측 기준을 통과한 후보 수는 각 candidates/outer_*/FROZEN.json에 고정되어 있다. 합집합을 전체 fold 공통 통과로 바꾸지 않는다.'")
source=source.replace("f'대조모형 검정을 무시한 원시 반대 방향 패턴이 같은 피처·상태쌍·방향으로 4/5 fold 이상 나타난 피처는 {raw4features}개, 조합은 {raw4}개다. 이것만으로 가변 피처를 인증하지 않는다.'", "f'대조모형 검정을 무시한 원시 반대 방향 패턴이 같은 피처·상태쌍·방향으로 4/5 fold 이상 나타난 조합은 {raw4}개다. 이것만으로 가변 피처를 인증하지 않는다.'")
import hashlib
assert hashlib.sha256(source.encode('utf-8')).hexdigest()==task['source_sha256'],'Frozen v1 reconstruction must be byte-exact'
atomic(old,source,immutable=False)
replacement={**task,'id':'WRITE_SCIENTIFIC_REPORTS_V2','script':new,'source_sha256':digest(ROOT/new)}
queue=[replacement if r['id']==task['id'] else r for r in queue]
atomic(BASE/'repairs/EXPLICIT_COUNTS_REPORT_VERSION_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
    reason='Answer EARLY candidate count and number of raw-pattern features explicitly instead of only pointing to files.',
    old_registered_source_restored_byte_exact=True,old_task_never_executed=True,scientific_data_or_rules_unchanged=True,
    original=task,replacement=replacement,editing_note='An edit was initially applied to the unexecuted v1; before execution copied to v2 and restored v1 to its exact registered SHA.'))
atomic(BASE/'runtime/TASK_QUEUE.json',queue,immutable=False)
