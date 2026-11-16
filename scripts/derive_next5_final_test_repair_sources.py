"""Create reviewable variants without changing any original audit-pinned source."""
from pathlib import Path
import ast
import difflib
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, read_json, digest, atomic_json, utc_now


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    records = []
    out = BASE / 'maintenance/FINAL_TEST_ACCESS_REPAIR'
    for name, sha in read_json(BASE / 'final_audit/PLAN.json')['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    def derive(source, target, change):
        before = (ROOT / source).read_text(encoding='utf-8-sig')
        after = change(before)
        ast.parse(after)
        with (ROOT / target).open('x', encoding='utf-8', newline='') as stream:
            stream.write(after)
        diff = out / (Path(target).stem + '.diff')
        with diff.open('x', encoding='utf-8', newline='') as stream:
            stream.write(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=source, tofile=target)))
        records.append({'original': source, 'original_sha256': digest(ROOT / source),
            'derived': target, 'derived_sha256': digest(ROOT / target),
            'diff': diff.relative_to(ROOT).as_posix(), 'diff_sha256': digest(diff)})

    def suite(text):
        text = replace_once(text, "'src.next5.test_access_guard'", "'src.next5.final_test_compatibility'")
        text = replace_once(text, "ROOT/'src/next5/test_access_guard.py'", "ROOT/'src/next5/final_test_compatibility.py'")
        return replace_once(text, "'access_guard_sha256':digest(out/'ACCESS_GUARD.json'),",
            "'access_guard_sha256':digest(out/'ACCESS_GUARD.json'),'test_compatibility_sha256':digest(out/'TEST_COMPATIBILITY.json'),")
    derive('scripts/run_next5_full_tests.py', 'scripts/run_next5_full_tests_after_access_repair.py', suite)

    def graph(text):
        return replace_once(text, '    states, inputs, edges, rows = {}, {}, {}, []',
            '    from src.next5.final_test_repair_evidence import verify_execution_changes\n'
            '    verify_execution_changes(jobs, paths, allowed_late_dependencies)\n'
            '    states, inputs, edges, rows = {}, {}, {}, []')
    derive('scripts/audit_next5_execution_graph.py', 'scripts/audit_next5_execution_graph_after_test_repair.py', graph)

    def korean(text):
        text = replace_once(text, '최종 quarantine 11개는', '연구 단계 quarantine 11개는')
        lines = text.splitlines(True)
        positions = [i for i, line in enumerate(lines) if "handoff += '[최종 실패 분류표]" in line]
        assert len(positions) == 1
        paragraphs = [
            '[최종 실패 분류표](FAILURE_CLASSIFICATION_NEXT5.csv)의 33개 행은 실제 실행 오류 5건(Python queue 4건과 native 기록 오류 1건), 시작 전 대체 8건, screen 탈락 1건, full 승격 기준 탈락 19건을 구분한다. 연구 단계의 32개 행에 최종 전체 suite 오류 한 건을 추가했다. 총 12개 queue quarantine은 원래 실패 상태를 보존하고 성공 근거를 연결했다.',
            '최초 최종 suite는 1,357개 중 실패 8건·설정 오류 37건이었다. 고정 ID·그룹·역할·폴드 메타데이터를 blanket reader guard가 막은 것이 직접 원인이었고 관측값 접근은 차단됐다. 해시와 네 열의 schema를 검증하는 좁은 예외, 같은 과거 모델 가중치만 읽는 테스트 어댑터로 복구했다. 모든 원래 테스트와 추가 경계 계약을 별도 경로에서 다시 실행했다. 원래 소스·실패 로그·감사 PLAN을 보존하고 PLAN_AFTER_TEST_REPAIR와 파생 소스의 정확한 diff를 연결했다. 최종 감사 시작 시각이나 시간을 소급하지 않았다.',
            '이 분류표는 모든 shell·개발·감독 예외의 총수는 아니다. [과거 감독 상태 파일 읽기 오류](docs/NEXT5_MONITOR_EXCEPTION_REVIEW.md)는 같은 latency worker의 정상 완료와 원본을 보존한 별도 검토다. [보고서 준비 명령 인코딩 오류](docs/NEXT5_REPORT_ENCODING_REPAIR.md)도 두 실패와 성공한 UTF-8 출력 검사를 따로 기록했다. 이들 기록은 모델 성능 실패나 새로운 후보 학습으로 합산하지 않는다.'
        ]
        lines[positions[0]] = ''.join('    handoff += ' + repr(paragraph + '\n\n') + '\n' for paragraph in paragraphs)
        return ''.join(lines)
    derive('src/next5/korean_reports.py', 'src/next5/korean_reports_after_test_repair.py', korean)

    def publication(text):
        text = replace_once(text, 'from src.next5.korean_reports import documents',
            'from src.next5.korean_reports_after_test_repair import documents')
        text = replace_once(text, "    plan = read(BASE / 'final_audit/PLAN.json')",
            "    from src.next5.final_test_repair_evidence import verify_sources\n"
            "    verify_sources()\n"
            "    read(BASE / 'final_audit/PLAN.json')\n"
            "    read(BASE / 'maintenance/FINAL_TEST_ACCESS_REPAIR/POLICY_LOCK.json')\n"
            "    read(BASE / 'maintenance/FINAL_TEST_ACCESS_REPAIR/REPAIR.json')\n"
            "    read(BASE / 'maintenance/FINAL_TEST_ACCESS_REPAIR/RESOLVED.json', 'PASS')\n"
            "    plan = read(BASE / 'final_audit/PLAN_AFTER_TEST_REPAIR.json')")
        text = replace_once(text, "    tests = read(audit_root / 'full_tests/RESULTS.json', 'PASS')",
            "    tests = read(BASE / 'final_audit/final_tests_repair_v1/full_tests/RESULTS.json', 'PASS')\n"
            "    read(BASE / 'final_audit/final_tests_repair_v1/full_tests/TEST_COMPATIBILITY.json')")
        return replace_once(text, 'from src.next5.failure_reports import completed_failure_rows',
            'from src.next5.final_test_repair_evidence import completed_failure_rows')
    derive('scripts/publish_next5_final_reports.py', 'scripts/publish_next5_final_reports_after_test_repair.py', publication)
    atomic_json(out / 'DERIVATION.json', {'created_utc': utc_now(),
        'status': 'DERIVED_WITH_ORIGINAL_SOURCES_PRESERVED', 'records': records,
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0,
        'source_script_sha256': digest(Path(__file__))}, immutable=True)
    print('FOUR VERSIONED AUDIT/REPORT SOURCES DERIVED', flush=True)


if __name__ == '__main__':
    main()
