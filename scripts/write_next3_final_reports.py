"""Prepare review drafts or publish Korean reports after all final evidence passes."""
from pathlib import Path
import argparse
import datetime
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.bottlenecks import record_board
from src.next3.final_report_facts import load_facts
from src.next3.final_validation import successful_receipt, tests_proof
from src.next3.korean_final_reports import supporting_reports
from src.next.guard import SealGuard
from scripts.prepare_next3_independent_reproduction import time_gate
from scripts.publish_next3_final_evidence import replace_bytes


def run_context(preview):
    now = datetime.datetime.now(datetime.timezone.utc)
    session = read_json(BASE/'SESSION.json')
    start = datetime.datetime.fromisoformat(session['started_utc'])
    audit = datetime.datetime.fromisoformat(session['audit_started_utc']) if session.get('audit_started_utc') else None
    account = read_json(BASE/'runtime/ACCOUNTING.json')
    observed = datetime.datetime.fromisoformat(account['timestamp'])
    if not 0 <= (now-observed).total_seconds() < 180:
        raise RuntimeError('A fresh observed supervisor accounting snapshot is required')
    run = {'preview': preview, 'observed_utc': now.isoformat(), 'started_utc': start.isoformat(),
        'wall_seconds': (now-start).total_seconds(),
        'research_wall_seconds': ((audit or now)-start).total_seconds(),
        'audit_wall_seconds': (now-audit).total_seconds() if audit else 0.0,
        'accounting_timestamp': account['timestamp'],
        'productive_research_seconds': account['productive_seconds'],
        'audit_productive_seconds': account['audit_productive_seconds'],
        'unobserved_seconds': account['unobserved_seconds'], 'raw_fits': 0,
        'tests': None, 'runtime': None, 'accounting_snapshot': account,
        'final_core_audit_and_actual_archive_completion_authorities_are_separate': True}
    if preview:
        if audit is not None: raise RuntimeError('Do not create pre-audit previews during final audit')
        return run
    timing = time_gate()
    publication = successful_receipt(BASE/'final_audit/publication/RESULTS.json')
    for name, sha in publication['files_sha256'].items():
        if digest(ROOT/name) != sha: raise RuntimeError('Published evidence changed before final reports: '+name)
    selection = successful_receipt(ROOT/'DEV_CANDIDATES_NEXT3.json', statuses=('FROZEN_NEXT3_DEV_SELECTION',))
    if selection['scientific_role_count'] != 2 or selection['selected_quality_delta_vs_starting_performance'] != 0.0:
        raise RuntimeError('Final report templates require the actual two unchanged scientific roles')
    reproduction = successful_receipt(BASE/'independent_reproduction/RESULTS.json')
    if reproduction['raw_reproduction_fits'] != 3: raise RuntimeError('Both final raw training roles are required')
    runtime = successful_receipt(BASE/'final_audit/runtime/RESULTS.json')
    if runtime['total_actual_public_API_points'] != 30_000_000: raise RuntimeError('Final actual public API verification incomplete')
    tests = tests_proof(BASE/'final_audit/FULL_TESTS.xml', minimum_count=798,
        required_classnames={'tests.test_ts_auc', 'tests.test_scorer_official_checks',
            'tests.test_next3_final_validation', 'tests.test_next3_final_runtime_probe'},
        not_before_timestamp=datetime.datetime.fromisoformat(timing['audit_started_utc']).timestamp())
    run.update(raw_fits=3, tests=tests, runtime=runtime, publication_sha256=digest(BASE/'final_audit/publication/RESULTS.json'))
    return run


def final_board():
    items = read_json(BASE/'bottlenecks/BOTTLENECK_BOARD.json')['bottlenecks']
    decisions = {
        'B1': ('PARTIALLY_RESOLVED', 'Fixed routing menu explains part of the DEPENDENCE/SCALE trade-off but produced no admitted scientific replacement.'),
        'B2': ('UNCHANGED', 'No supported real-DEV plus ARCH/tail scale-down repair passed every frozen gate. Preserve current roles.'),
        'B3': ('UNCHANGED', 'No new accepted complement or smaller necessary C205 subset; I2 gain is below the frozen minimum effect.'),
        'B4': ('PARTIALLY_RESOLVED', 'Claims, pair-matrix numerical audit and frozen finite-menu accounting are explicit; adaptive DEV generalization uncertainty remains unresolved.'),
        'B5': ('DEFERRED', 'No approved correction for accumulating long-null alarms and late first-detection utility; keep these joint future criteria.'),
        'B6': ('PARTIALLY_RESOLVED', 'D005/D006 identify conditional legacy EWMA/serial memory persistence; no deployable reset/remedy was selected.'),
        'B7': ('DEFERRED', 'Early-age ranking remains weak; its small official pair mass does not justify reopening a scientific family after the finite menu closes.'),
        'B8': ('PARTIALLY_RESOLVED', 'Same-prediction lean runtime passes measured memory and cost gates and final raw/API/tests. Actual archive replay and external receipt are separately required; cross-platform runtime is unverified.')}
    assert {item['id'] for item in items} == set(decisions)
    for item in items:
        status, reason = decisions[item['id']]
        item.update(status=status, final_assessment=reason, next_test=reason)
        for path in (ROOT/'DEV_CANDIDATES_NEXT3.json', ROOT/'REPRODUCTION_NEXT3.json',
                     BASE/'final_audit/runtime/RESULTS.json', BASE/'final_audit/FULL_TESTS.xml'):
            item['evidence'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
    return record_board(items, 'Final report closure after two-role raw training, actual 30M public API verification and fresh full tests. Scientific promotions zero; final core audit and actual archive proof remain separate.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', help='Unique maintenance preview name; no final root files are published')
    args = parser.parse_args(); preview = args.preview is not None
    if preview and not re.fullmatch(r'[A-Za-z0-9_-]+', args.preview): raise ValueError('Safe preview name required')
    run = run_context(preview); facts = load_facts(); reports = supporting_reports(facts, run)
    guard = SealGuard(ROOT); guard.check_state()
    if any(item['opened'] for item in guard.lock['partitions'].values()): raise RuntimeError('Seals must remain unopened')
    if len(re.findall(r'^\d+\. \*\*', reports['00_GPT_READ_FIRST_NEXT3.md'], flags=re.MULTILINE)) != 44:
        raise RuntimeError('Exactly 44 Korean numbered answers required')
    out = BASE/'maintenance/report_previews'/args.preview if preview else BASE/'final_reports'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'FACTS.json', facts, immutable=True)
    atomic_json(out/'RUN_CONTEXT.json', run, immutable=True)
    atomic_json(out/'ACCOUNTING_SNAPSHOT.json', run['accounting_snapshot'], immutable=True)
    destination = out if preview else ROOT
    for name, content in reports.items():
        replace_bytes(destination/name, content.encode('utf-8'))
    if not preview:
        replace_bytes(ROOT/'NEXT3_PREFLIGHT.json', (BASE/'preflight/NEXT3_PREFLIGHT.json').read_bytes())
        retained_forensics = out/'forensic_inputs'; retained_forensics.mkdir()
        additions = {
            'SCALE_DOWN_FORENSIC.md': '\n\n최종 해석: D005/D006은 단일 이상치 이후 state 잔류를 대조한 별개 진단이다. 이 실험에서 legacy EWMA 영향이 확인됐다는 사실만으로 ARCH/tail scale-down 실패의 단일 원인을 확정하지 않는다. R2/R3 확률 손실과 TS-AUC의 유한 후보 순서 차이는 일반적인 logistic ranking inconsistency 정리가 아니다. 수정 수식과 적용 범위는 `docs/NEXT3_RANKING_SURROGATE_METHODOLOGY_V2.md`를 따른다. 과학 역할을 바꾼 감소 문제 해결책은 없다.\n',
            'DEPENDENCE_FORENSIC.md': '\n\n최종 검증: D007의 다른 pair-matrix 계산으로 기존 fold score 90,045개와 대비 배열 24,012개가 정확히 재현됐다. 새 통계적 독립 검증은 아니다. D005/D006에서 단일 이상치의 늦은 잔류는 legacy EWMA 묶음 및 J serial evidence와 연관된 직접 상태 대조 효과를 보였다. C205의 개별 lag cell 필요성을 이 합성 intervention에서 추론하지 않는다. 다음 변경은 real DEV 전체·supported mechanism·long-null·늦은 변화 utility를 함께 통과해야 한다.\n'}
        for name, extra in additions.items():
            original = (ROOT/name).read_bytes(); replace_bytes(retained_forensics/name, original)
            content = original.decode('utf-8-sig').replace(
                'Scientific-menu review; engineering work and final audit remain in progress.',
                'Finite scientific menu closed; scientific roles unchanged. Final core audit and actual archive verification are separate receipts.')
            replace_bytes(ROOT/name, (content+extra).encode('utf-8'))
        board = final_board(); atomic_json(ROOT/'BOTTLENECK_BOARD_FINAL.json', board)
        for name, path in [('BOTTLENECK_HISTORY.csv', BASE/'bottlenecks/BOTTLENECK_HISTORY.csv'),
                           ('RESOURCE_LOG.csv', BASE/'runtime/RESOURCE_LOG.csv'),
                           ('HANG_RECOVERY_LOG.csv', BASE/'runtime/HANG_RECOVERY_LOG.csv')]:
            replace_bytes(ROOT/name, path.read_bytes())
        atomic_json(ROOT/'MAINTENANCE_LOG.json', {
            'status': 'FINAL_REPORT_OBSERVED_SNAPSHOT', 'created_utc': utc_now(),
            **facts['maintenance'], 'accounting': run['accounting_snapshot'],
            'wall_seconds_at_report': run['wall_seconds'], 'science_fits': 129,
            'separate_final_raw_reproduction_fits': 3, 'total_completed_fits': 132,
            'new_scientific_recipes': 13, 'completed_full_recipes': 12,
            'source_accounting_sha256': digest(out/'ACCOUNTING_SNAPSHOT.json'),
            'final_wall_closure_authority': 'FINAL_AUDIT_NEXT3.json and actual Desktop archive external receipt',
            'no_unobserved_time_recredited': True})
        atomic_json(ROOT/'SEAL_GUARD_NEXT3.json', {
            'status': 'PASS', 'created_utc': utc_now(), 'all_seals_unopened': True,
            'seals_opened': {name: False for name in guard.lock['partitions']},
            'DEV_series': len(guard.dev_ids), 'seal_series': len(guard.seal_ids),
            'DEV_seal_overlap': len(guard.dev_ids & guard.seal_ids),
            'seal_lock_sha256': guard.lock_sha, 'reduced_ledger_sha256': guard.ledger_sha,
            'reduced_new_usage': 0, 'public_submission_new': 0, 'cloud_submission_new': 0,
            'boundary': 'Application-level admission and audited Python value access, not an OS sandbox.'})
    files = {name: digest(destination/name) for name in reports}
    if not preview:
        for name in ('BOTTLENECK_BOARD_FINAL.json', 'BOTTLENECK_HISTORY.csv', 'RESOURCE_LOG.csv',
                     'HANG_RECOVERY_LOG.csv', 'MAINTENANCE_LOG.json', 'SEAL_GUARD_NEXT3.json',
                     'NEXT3_PREFLIGHT.json', 'SCALE_DOWN_FORENSIC.md', 'DEPENDENCE_FORENSIC.md'):
            files[name] = digest(ROOT/name)
    source_paths = [Path(__file__), ROOT/'src/next3/final_report_facts.py', ROOT/'src/next3/korean_final_reports.py']
    retained = out/'original_sources'; retained.mkdir()
    for path in source_paths:
        replace_bytes(retained/path.name, path.read_bytes())
    atomic_json(out/'RESULTS.json', {
        'status': 'REVIEW_PREVIEW_NOT_FINAL_DELIVERY' if preview else 'PASS', 'created_utc': utc_now(),
        'preview': preview, 'required_numbered_answers': 44, 'language': 'Korean',
        'files_sha256': files, 'facts_sha256': digest(out/'FACTS.json'),
        'run_context_sha256': digest(out/'RUN_CONTEXT.json'),
        'accounting_snapshot_sha256': digest(out/'ACCOUNTING_SNAPSHOT.json'),
        'source_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths},
        'retained_source_sha256': {path.name: digest(retained/path.name) for path in source_paths},
        'raw_reproduction_fits': run['raw_fits'], 'new_scientific_fits': 0,
        'actual_archive_completion_claimed_internally': False,
        'same_agent_review_not_independent_human': True}, immutable=True)
    progress('Korean NEXT3 report '+('preview' if preview else 'publication'),
        '44 answers and five supporting reports written from fixed evidence; final ZIP completion is an external proof',
        phase='ACTIVE_RESEARCH' if preview else 'FINAL_AUDIT')
    print('NEXT3 KOREAN REPORTS', 'PREVIEW' if preview else 'PASS', out, flush=True)


if __name__ == '__main__': main()
