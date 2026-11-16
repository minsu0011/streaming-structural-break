"""Keep all scientific answers; disclose the subsequent graph-inventory repair."""
from src.next5.korean_reports_after_test_repair import documents as previous_documents


def documents(context):
    result = previous_documents(context)
    key = 'RESEARCH_REPORT_NEXT5.md'
    text = result[key]
    old = '[최종 실패 분류표](FAILURE_CLASSIFICATION_NEXT5.csv)의 33개 행은 실제 실행 오류 5건(Python queue 4건과 native 기록 오류 1건), 시작 전 대체 8건, screen 탈락 1건, full 승격 기준 탈락 19건을 구분한다. 연구 단계의 32개 행에 최종 전체 suite 오류 한 건을 추가했다. 총 12개 queue quarantine은 원래 실패 상태를 보존하고 성공 근거를 연결했다.'
    new = '[최종 실패 분류표](FAILURE_CLASSIFICATION_NEXT5.csv)의 34개 행은 실제 실행 오류 6건(Python queue 5건과 native 기록 오류 1건), 시작 전 대체 8건, screen 탈락 1건, full 승격 기준 탈락 19건을 구분한다. 연구 단계의 32개 행에 최종 suite와 실행 이력 감사의 오류 두 건을 추가했다. 총 13개 queue quarantine은 원래 실패 상태를 보존하고 성공 근거를 연결했다.'
    assert text.count(old) == 1
    text = text.replace(old, new)
    text += '\n[최종 정책 파일 수집 복구](docs/NEXT5_FINAL_GRAPH_POLICY_SCOPE_REPAIR.md): 실행 이력 감사가 두 완료된 pytest 작업의 임시 합성 fixture까지 정책으로 수집해, 의도적으로 손상한 8바이트 JSON에서 실패했다. 두 정확한 fixture 경로의 정책 파일 44개만 실제 정책 수집에서 제외하고 원본 바이트를 보존했다. 나머지 정책·소스 해시와 모든 실행 순서 검사는 유지했다. 5개 범위 계약, 완료된 그래프 재검사, 실패 해결 영수증 및 마지막 그래프 검증으로 복구했다. 기존 1,363개 전체 테스트와 과학적 결과는 변경하지 않았으며 새 fit·예측·bootstrap draw는 없다. 원래 두 PLAN과 실패 소스를 보존하고 PLAN_AFTER_GRAPH_REPAIR.json에 별도 복구를 연결했다.\n'
    result[key] = text
    return result
