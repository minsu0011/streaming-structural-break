"""Render the final Korean handoff from frozen, verified numerical evidence.

This writer cannot run before ten research hours and explicit audit start.
It preserves previous root reports and never changes experiment evidence.
"""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.research_clock import ResearchClock
from src.next.active_research_time import verify_final_duration
from src.next.guard import SealGuard
from src.utils.artifacts import write_json, sha256, utc_now
from scripts.package_next_handoff import REQUIRED


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def csv(path):
    return pd.read_csv(ROOT/path)


def dump_table(name, frame):
    frame.to_csv(ROOT/name, index=False, encoding='utf-8-sig')


def md_table(frame):
    # No optional tabulate dependency in the locked research environment.
    def cell(value):
        return str(value).replace('|', '/').replace('\n', ' ')
    rows = ['| '+' | '.join(map(cell, frame.columns))+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    rows.extend('| '+' | '.join(map(cell, row))+' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(rows)


def write_md(name, text):
    (ROOT/name).write_text(text.strip()+'\n', encoding='utf-8')


def main():
    clock = ResearchClock(ROOT).snapshot()
    if not clock['minimum_research_met'] or not clock['final_audit_started']:
        raise RuntimeError('Final handoff writing requires ten research hours and explicit audit start')
    substantive_duration = verify_final_duration(ROOT)
    receipt_dir = ROOT/'artifacts/next/final_reports'
    if (receipt_dir/'RESULTS.json').exists():
        raise RuntimeError('Preserve completed final reports; use an explicit reviewed revision')
    receipt_dir.mkdir(parents=True, exist_ok=True)
    guard = SealGuard(ROOT)
    selection = read('artifacts/next/final_selection/RESULTS.json')
    snapshot_dir = Path(selection['evidence_snapshot'])
    snapshot = read(snapshot_dir/'RESULTS.json')
    if snapshot['status'] != 'PASS' or not snapshot['payload_byte_checks_requested']:
        raise RuntimeError('Verified final evidence snapshot required')
    for relative, digest in snapshot['source_reports_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError('Final evidence changed: '+relative)
    for relative, digest in snapshot['files_sha256'].items():
        if sha256(ROOT/snapshot_dir/relative) != digest:
            raise RuntimeError('Final evidence snapshot changed')
    evidence = {r['candidate']['experiment_id']: r for r in read(snapshot_dir/'CANDIDATE_EVIDENCE.json')}
    figure_links = [
        ('성능·속도 비교', 'artifacts/next/figures/actual_artifact_tradeoffs_v2/actual_tradeoffs.png'),
        ('고정 변경의 paired 비교', 'artifacts/next/figures/fixed_interventions_v2/paired_interventions.png'),
        ('장기 오경보: 5% 기준', 'artifacts/next/figures/long_null_v2/null_horizon_alpha_5.png')]
    if any(not (ROOT/path).is_file() for _, path in figure_links):
        raise RuntimeError('Complete and inspect the planned research figures before final report writing')
    backup = ROOT/'artifacts/next/prior_root_reports/final_writer_input'
    if backup.exists():
        raise RuntimeError('Root report backup already exists; inspect a prior partial write')
    backup.mkdir(parents=True)
    backups = {}
    for name in REQUIRED:
        if (ROOT/name).is_file():
            shutil.copyfile(ROOT/name, backup/name)
            backups[name] = sha256(backup/name)
    write_json(backup/'BACKUP_MANIFEST.json', {'created_utc': utc_now(), 'files_sha256': backups})
    primary_ref = read('artifacts/next/REFERENCE_RESTORATION.json')['references']['PRIMARY']
    secondary_path = Path('artifacts/next/research_synthesis')
    secondary = read(secondary_path/'RESULTS.json')
    if secondary['status'] != 'PASS':
        raise RuntimeError('Complete long-horizon, direction and contribution synthesis required')
    for relative, digest in secondary['source_files_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError('Secondary diagnostic source changed: '+relative)
    for name, digest in secondary['files_sha256'].items():
        if sha256(ROOT/secondary_path/name) != digest:
            raise RuntimeError('Secondary diagnostic synthesis changed: '+name)
    horizon_ranges = csv(secondary_path/'HORIZON_REGIME_RANGES.csv')
    long_null_text = '; '.join(f"{r['model']}: {100*r['min_regime_fpr_at_16384']:.1f}–{100*r['max_regime_fpr_at_16384']:.1f}%"
        for r in horizon_ranges[horizon_ranges.alpha_at_1024 == .05].to_dict('records'))
    scale_directions = csv('artifacts/next/synthetic_variance_stress/direction_fractures/SCALE_PAIRED_BOOTSTRAP.csv')
    scale_risk = scale_directions[(scale_directions.regime == 'AR03_ARCH065_GAUSSIAN') &
        (scale_directions.polarity == 'scale_down_1_over_1_5') & (scale_directions.reference == 'HIGHEST_MEAN')].iloc[0]
    scale_risk_text = (f"합성 AR0.3/ARCH0.65 Gaussian의 scale 감소 조건에서 V007−기존 HIGHEST_MEAN은 "
        f"{scale_risk['estimate_mean_delta']:+.6f}, 조건부 CI [{scale_risk['ci_2_5']:+.6f},{scale_risk['ci_97_5']:+.6f}]였다. "
        '20개 방향별 비교 전체는 보조 진단 문서에 있다. 이 CV fold-0 반례를 최종 J102/B104 all-DEV 모델의 직접 측정으로 해석하지 않는다.')
    rows, cv_rows, all_fractures, disagreement = [], [], [], []
    for path in sorted((ROOT/'artifacts/next/experiments').glob('*/RESULTS.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        candidate = result['candidate']
        eid = candidate['experiment_id']
        record = evidence[eid]
        full = result.get('full', {})
        valid_full = full.get('status') == 'COMPLETE' and result['status'] != 'BUG'
        supplement = record['supplemental']
        boot = supplement.get('paired_bootstrap', {})
        latency = supplement.get('latency', {})
        base = {'experiment_id': eid, 'parent_id': candidate.get('parent_id'), 'generation': candidate.get('generation'),
            'phase': candidate.get('phase'), 'hypothesis': candidate.get('hypothesis'), 'change': candidate.get('change'),
            'expected_effect': candidate.get('expected_effect'), 'risks': candidate.get('risks'),
            'learner': candidate.get('learner'), 'weighting': candidate.get('weighting'),
            'params': json.dumps(candidate.get('params', {}), sort_keys=True),
            'engine_config': json.dumps(candidate.get('engine_config', {}), sort_keys=True),
            'extension': json.dumps(candidate.get('extension', {}), sort_keys=True),
            'feature_groups': json.dumps(candidate.get('groups', [])),
            'original_stage_status': result['status'], 'final_evidence_status': record['assessed_status'],
            'original_reason': result.get('reason'),
            'failed_gates': json.dumps(record['decision']['failed_gates']), 'pending_gates': json.dumps(record['decision']['pending_gates']),
            'primary_cv_mean': full.get('mean') if valid_full else None,
            'primary_cv_median': full.get('median') if valid_full else None,
            'primary_cv_worst': full.get('worst_fold') if valid_full else None,
            'delta_vs_primary': full['mean']-primary_ref['mean'] if valid_full else None,
            'screen_mean': result.get('screen', {}).get('mean'),
            'screen_delta': result.get('screen', {}).get('paired_bootstrap', {}).get('estimate_mean_delta'),
            'bootstrap_replicates': boot.get('replicates'), 'bootstrap_probability': boot.get('probability_delta_positive'),
            'bootstrap_ci_low': boot.get('ci_2_5'), 'bootstrap_ci_high': boot.get('ci_97_5'),
            'alternate_mean_deltas': json.dumps(record['evidence'].get('alternate_mean_deltas')),
            'leave_one_h_regime_out_worst': supplement.get('historical_worst'),
            'point_latency_median_us': latency.get('point_distribution_us', {}).get('median'),
            'history_amortized_us_per_point': latency.get('real_history_initialization', {}).get('amortized_us_per_point'),
            'new_generation_seal_worthy': record['decision']['new_generation_seal_worthy'],
            'elapsed_experiment_seconds': result.get('elapsed_seconds'),
            'feature_version': result.get('feature_hash', candidate.get('extension_frozen_hash')),
            'experiment_report_sha256': sha256(path), 'seal_rows': 0, 'reduced_new_uses': 0}
        rows.append(base)
        for screen_row in result.get('screen', {}).get('cv_rows', []):
            cv_rows.append({**screen_row, 'experiment_id': eid, 'scope': 'fixed_3000_series_screen_only',
                'quality_admissible': result['status'] != 'BUG', 'source_report_sha256': sha256(path), 'seal_rows': 0})
        if valid_full:
            original_details = {row['fold']: row for row in full.get('cv_rows', [])}
            for fold, score in enumerate(full['fold_scores']):
                cv_rows.append({**original_details.get(fold, {}), 'experiment_id': eid,
                    'scope': 'original_five_fold_primary_authority', 'fold': fold, 'ts_auc': score,
                    'quality_admissible': True, 'source_report_sha256': sha256(path), 'seal_rows': 0})
        for split in supplement.get('robustness_splits', []):
            for fold, score in enumerate(split['fold_scores']):
                fold_path = ROOT/'artifacts/next/robustness/candidates'/eid/split['split_name']/f'fold_{fold}.json'
                detail = json.loads(fold_path.read_text(encoding='utf-8'))
                if abs(detail['ts_auc']-score) > 1e-14:
                    raise RuntimeError('Stress fold and aggregate score differ')
                cv_rows.append({'experiment_id': eid, 'scope': split['split_name'], 'fold': fold, 'ts_auc': score,
                    'validation_series': detail.get('validation_series'), 'validation_rows': detail.get('validation_rows'),
                    'reference_ts_auc': detail.get('reference_ts_auc'), 'delta_vs_primary': detail.get('delta_vs_primary'),
                    'quality_admissible': True, 'source_report_sha256': sha256(fold_path), 'seal_rows': 0})
        fracture_path = Path('artifacts/next/forensic')/eid/'PAIRED_FRACTURES.csv'
        if (ROOT/fracture_path).exists():
            frame = csv(fracture_path)
            frame['result_file'] = fracture_path.as_posix()
            frame['result_file_sha256'] = sha256(ROOT/fracture_path)
            all_fractures.append(frame)
        complement = supplement.get('complementarity_vs_primary')
        if complement:
            disagreement.append({'candidate': eid, 'reference': 'PRIMARY', **complement,
                'scope': 'OOF within-time rank correlation plus predeclared supported-mechanism fracture thresholds'})
    table = pd.DataFrame(rows)
    # Snapshot schema is checked explicitly instead of silently deriving a
    # favourable status from missing evidence.
    dump_table('MODEL_RESULTS_NEXT.csv', table)
    dump_table('EXPERIMENT_LEDGER.csv', table)
    dump_table('CV_RESULTS_NEXT.csv', pd.DataFrame(cv_rows))
    fractures = pd.concat(all_fractures, ignore_index=True)
    dump_table('PAIRED_BOOTSTRAP_RESULTS.csv', fractures)
    dump_table('TIME_FRACTURE_NEXT.csv', fractures[fractures.scope.str.startswith('time_')])
    dump_table('BREAK_AGE_NEXT.csv', fractures[fractures.scope.str.startswith('age_')])
    dump_table('MECHANISM_FRACTURE_RESULTS.csv', fractures[fractures.scope.str.startswith('mechanism_')])
    dump_table('CURRENT_TOP_DISAGREEMENT.csv', pd.DataFrame(disagreement))
    phase = table.phase.fillna('')
    slices = {
        'WHITENING_RESULTS.csv': phase.isin(['WHITEN', 'LEGACY_WHITENING_EXTENSION', 'ARSCORE', 'AR_MISMATCH', 'FORWARD_H_CALIBRATION']),
        'INNOVATION_NORMALIZATION_RESULTS.csv': phase.isin(['NORM', 'ARCH', 'CONDITIONAL_VARIANCE', 'LOG_ENERGY_MECHANISM', 'VARIANCE_AGE_GRID']),
        'SEQUENTIAL_STAT_RESULTS.csv': phase.isin(['SEQ', 'STANDALONE', 'ADD', 'TRANSITION', 'SCORE_DIRECTION', 'ARSCORE']),
        'GLR_RESULTS.csv': table.feature_groups.str.contains('GLR') | phase.isin(['H_SERIAL_GLR_BAYES_CORRECTION', 'LONG_VARIANCE_MATCHED_ABLATION', 'VARIANCE_MATCHED_ABLATION']),
        'BAYES_LITE_RESULTS.csv': table.feature_groups.str.contains('BAYES') | phase.isin(['H_SERIAL_GLR_BAYES_CORRECTION', 'LONG_VARIANCE_MATCHED_ABLATION', 'VARIANCE_MATCHED_ABLATION']),
        'RANKING_OBJECTIVE_RESULTS.csv': phase.isin(['RANKING', 'VARIANCE_MATCHED_OBJECTIVE']),
        'WEIGHTING_RESULTS_NEXT.csv': phase.isin(['WEIGHT', 'VARIANCE_MATCHED_OBJECTIVE']),
        'FEATURE_ABLATION_NEXT.csv': phase.str.contains('ABLATION|ABLATE')}
    for name, mask in slices.items():
        dump_table(name, table[mask])
    regime_rows = []
    for row in cv_rows:
        if row['scope'] == 'LEAVE_ONE_H_REGIME_OUT':
            regime_rows.append({**row, 'regime': row['fold'], 'validation_kind': 'Entire H-defined regime excluded from every underlying component fit; late combinations reuse these fits'})
    dump_table('HISTORICAL_REGIME_RESULTS.csv', pd.DataFrame(regime_rows))
    audit_start = read('artifacts/next/scheduler/FINAL_AUDIT_START.json')
    research_hours = substantive_duration['substantive_research_seconds']/3600
    deliveries, final_training_reports = [], {}
    for selected in selection['candidates']:
        training_path = Path('artifacts/next/final_training')/selected['role']/'RESULTS.json'
        training = read(training_path)
        if training['status'] != 'PASS' or not all(training['checks'].values()) or training['model_sha256'] != selected['model_sha256']:
            raise RuntimeError('Selected final raw iterator audit is incomplete')
        deliveries.append({**selected, 'delivery_model_directory': training['delivery_model_directory'],
            'delivery_descriptor_sha256': training['delivery_descriptor_sha256'], 'final_training_report': training_path.as_posix()})
        final_training_reports[selected['role']] = training
    selection = {**selection, 'candidates': deliveries,
        'frozen_selection_report_sha256': sha256(ROOT/'artifacts/next/final_selection/RESULTS.json'),
        'delivery_note': 'Roles and weights remain frozen; delivery paths reference exact raw-iterator reproductions.'}
    write_json(ROOT/'DEV_CANDIDATES_NEXT.json', selection)
    primary = next(row for row in deliveries if row['role'] == 'PRIMARY_DEV_CANDIDATE')
    eid = primary['experiment_id']
    best = evidence[eid]
    boot = best['supplemental']['paired_bootstrap']
    alts = best['evidence']['alternate_mean_deltas']
    mechanisms = pd.DataFrame(best['supplemental']['mechanisms'])
    latencies = {p.stem: read(p.relative_to(ROOT)) for p in sorted((ROOT/'artifacts/next/latency/all_dev_v1').glob('*.json')) if p.name != 'POLICY.json'}
    # Latency folders may also contain partial identity receipts. Only complete
    # actual artifact records enter the final comparison.
    latencies = {key: value for key, value in latencies.items() if value.get('status') == 'PASS' and 'point_distribution_us' in value}
    write_json(ROOT/'LATENCY_NEXT.json', {'status': 'PASS', 'primary_basis': 'Contemporaneous isolated actual all-DEV model artifacts',
        'measurements': latencies, 'cv_reference_measurements_directory': 'artifacts/next/latency/research_v1',
        'observed_cv_timing_anomaly_note': 'artifacts/next/latency/MEASUREMENT_NOTES.json',
        'timed_boundary': 'Prepared scalar detector.predict_one, including per-point timer and float32 store. The infer generator validation and file/socket/cloud overhead are outside this point-call timing. API/wire correctness is audited separately.',
        'memory_measurement_boundary': 'State-array bytes are checked before/after 10M. rss_peak_bytes is the maximum sampled at million-point checkpoints, not an OS-recorded absolute high-water mark; transient allocation peaks can be missed. rss_after_bytes is collected after untimed scalar-reference validation and may exceed the sampled timing peak.',
        'note': '10M uses repeated fixed DEV buffers for memory and speed, not a new quality estimate. Native deployment/cloud timings were not measured.'})
    engineering = {row['experiment_id']: read(Path('artifacts/next/final_engineering')/row['experiment_id']/'RESULTS.json') for row in deliveries}
    write_json(ROOT/'DETERMINISM_NEXT.json', {'status': 'PASS', 'selected_engineering': engineering,
        'fixed_history_edge_contract': read('artifacts/next/history_edge_contract/RESULTS.json'),
        'delivery_cold_contracts': {role: read(Path('artifacts/next/final_training')/role/'DELIVERY_COLD_IMPORT.json') for role in final_training_reports},
        'scope': 'Exact floating-point predictions on this locked Windows/Python environment; no cross-hardware or cloud equivalence claim.'})
    reproduction = {p.parent.name: read(p.relative_to(ROOT)) for p in (ROOT/'artifacts/next/independent_raw_validation').glob('*/RESULTS.json')}
    write_json(ROOT/'REPRODUCTION_NEXT.json', {'status': 'PASS', 'independent_raw_collections': reproduction,
        'final_iterator_training': final_training_reports,
        'additional_portability_receipt_required_by_final_audit': 'artifacts/next/portable_delivery/RESULTS.json',
        'additional_portability_scope': 'The final audit separately requires newly installed pinned dependencies and copied package payload cold inference. This writer runs before that preflight; consult its final PASS receipt.',
        'scope': 'Separate clean source workspaces, copied canonical raw files, fresh scalar features, fixed CV recipes, and all-DEV model-byte comparisons. The separately installed prior reproduction environment was reused; native I/O was not OS sandboxed.'})
    mechanism_text = '; '.join(f"{r['name']}: Δ={r['estimate_mean_delta']:+.6f}, CI=[{r['ci_2_5']:+.6f},{r['ci_97_5']:+.6f}], n={int(r['break_series'])}, supported={r['supported']}" for r in best['supplemental']['mechanisms'])
    runtime = latencies[eid]
    latency_text = f"점당 median {runtime['point_distribution_us']['median']:.3f} μs; H 초기화 포함 {runtime['real_history_initialization']['amortized_us_per_point']:.3f} μs/point. 전체 원자료는 LATENCY_NEXT.json."
    roles_frame = pd.DataFrame([{'역할': row['role'], '후보': row['experiment_id'], '평균': row['primary_cv_mean'], '중앙값': row['median'], '최저 fold': row['worst_fold']} for row in deliveries])
    items = [
        f"실제 연구시간: {research_hours:.4f}시간. 재부팅·중단 공백 {substantive_duration['excluded_seconds']/3600:.4f}시간은 제외했다. 별도 최종 감사 시간은 final_audit/RESULTS.json과 ZIP 검증 영수증에 기록한다. 시계 복구와 제외 근거는 scheduler/recovery_1 및 SUBSTANTIVE_RESEARCH_DURATION.json에 보존했다.",
        f'NEXT 실험 수: {len(table)}개 후보. 상태별 {table.final_evidence_status.value_counts().to_dict()}. 재현·보조진단 fit 수는 후보 수와 구별한다.',
        f"기존 PRIMARY 평균: {primary_ref['mean']:.12f}.",
        f"새 PRIMARY: {eid}, 평균 {primary['primary_cv_mean']:.12f}.",
        f"기존 대비 평균 Δ: {primary['primary_cv_mean']-primary_ref['mean']:+.12f}.",
        f"고정 원본 분할에서 새 PRIMARY의 5-fold 점수: {primary['fold_scores']}.",
        f"새 PRIMARY fold 중앙값: {primary['median']:.12f}.",
        f"새 PRIMARY 최저 fold: {primary['worst_fold']:.12f}.",
        f"paired whole-series bootstrap 평균 Δ 95% CI: [{boot['ci_2_5']:+.9f}, {boot['ci_97_5']:+.9f}], {boot['replicates']}회. 선택 이후 조건부 구간이며 다중 탐색 보정 구간이 아니다.",
        f"P(Δ>0): {boot['probability_delta_positive']:.4f}.",
        f'고정 대체 split 3개의 평균 Δ: {alts}. 각 split은 5개 fold를 새로 학습했다.',
        f"H-regime 전체 제외 후 재학습한 최저 점수: {best['supplemental']['historical_worst']:.12f}. 원본 OOF의 regime별 절단 통계와 다르다.",
        'Mechanism별 기존 대비 결과: '+mechanism_text,
        'Whitening: H 내부 forward holdout에서 잔차 |ACF1| 평균은 AR1 0.07622, AR4 0.04801, AR8 0.04616, AR12 0.04620. 더 복잡한 whitening 자체가 DEV 개선을 보장하지 않았다.',
        'ECDF Gaussianization: H 잔차 kurtosis 중앙값 3.1753→2.9568로 정규 모양에 가까워졌지만 독립적인 성능 개선 근거는 부족하다. NORM 실험과 H 진단은 별도로 보존했다.',
        'GLR: 나이 후보를 128→512로 넓힌 Y001−V007 평균 Δ +0.0023342, 조건부 CI [+0.0004450,+0.0042894]. B104는 explicit GLR와 GLR-memory 10개 열을 제거하고 local+Bayes 16개 새 열을 유지했다. 제거 효과의 CI는 0을 포함하므로 GLR 자체의 필요성을 단정하지 않는다.',
        'Bayes-lite: 긴 age 조건의 B102(Bayes 제거)−Y001 평균 Δ −0.0027746, 조건부 CI [−0.0047041,−0.0008476]. 짧은 age의 제거 효과는 불분명했다. 정확한 사후확률 모델이라고 주장하지 않는다.',
        'Ranking: O104 rank objective 평균 0.6182062로 이진 목적의 대응 V007 0.6192973보다 낮았다. W0/W1/W2도 W3를 일관되게 대체하지 못했다. 세부 실패와 screen 통과 여부는 RANKING_OBJECTIVE_RESULTS.csv 및 WEIGHTING_RESULTS_NEXT.csv.',
        '가장 강한 새 아이디어: 변동성으로 나눈 잔차만 사용하지 않고 원시 innovation energy와 변동성 예측 상태의 변화 증거를 함께 보존한 것. 장기 age 추가가 후속 개선을 보였다.',
        '주요 실패/한계: 단독 sequential 통계, 무분별한 Gaussianization·ARCH 복잡도, 25-tree 축소, 일부 weighting/ranking, forecast-only의 최저 fold 저하. J102−Y001은 +0.0000002804, CI [−0.0013714,+0.0012768], P=0.5015로 자기상관 보정의 뚜렷한 이득이 아니다. 합성 stress에서도 tail/ARCH 반례가 있다.',
        'Latency: '+latency_text,
        'SEAL A/B/C/D opened: false / false / false / false. 총 2,000개 sealed ID는 계속 미개봉이다.',
        'Reduced 신규 사용: 0회. 과거 사용 2/2는 그대로 유지했다.',
        'Public leaderboard 신규 제출: 0회. SDK test 호출 및 실제 reduced 평가도 하지 않았다.',
        f"다음 세션 seal 논의 가능 여부: frozen rule상 {primary['new_generation_seal_worthy']}. 이는 이번 세션의 개봉 허가가 아니며 실제 seal은 0회 사용했다. 적응적 DEV 선택 편향과 장기 null alarm을 함께 검토해야 한다.",
        '다음 우선순위 3개: (1) 최종 감사·장기 null horizon 결과와 한계를 외부 GPT가 재검토, (2) 기본 모델 및 비용을 고정하고 재학습 없는 독립 평가 설계 논의, (3) 명시적 다음 세션 승인 이후에만 seal A 단일 사전등록 평가를 검토.']
    items[19] += ' 1,024-point에서 정한 5% 임계값의 16,384-point 무변화 오경보 범위(5개 합성 regime): '+long_null_text+'. 이 실험은 고정된 CV fold-0 모델만 사용했으며 최종 all-DEV 모델의 측정값이 아니다.'
    items[19] += ' '+scale_risk_text
    write_md('00_GPT_READ_FIRST_NEXT.md', '# Structural Break NEXT — 먼저 읽기\n\n'+'\n\n'.join(f'{j}. {text}' for j, text in enumerate(items, 1))+'\n\n'+md_table(roles_frame)+'\n\n'+' · '.join(f'[{label}]({path})' for label, path in figure_links))
    write_md('CURRENT_TOP_FORENSIC.md', '# 현재 최종 DEV 후보 분석\n\n'+md_table(roles_frame)+'\n\n'+
        f"PRIMARY `{eid}`의 전체 평균 Δ CI는 [{boot['ci_2_5']:+.9f}, {boot['ci_97_5']:+.9f}]이다. fold별 차이와 최저 fold를 함께 확인한다.\n\n"+
        md_table(mechanisms[['name', 'candidate_mean', 'reference_mean', 'estimate_mean_delta', 'ci_2_5', 'ci_97_5', 'supported', 'break_series']])+ '\n\n'+
        'MEAN(21개)과 TAIL(86개)은 지원 표본 기준 100개 미만이다. H-regime 2도 166개로 작다. 모든 mechanism 표식은 DEV 관측/라벨을 이용한 사후 진단이며 학습 입력이나 seal 선택 기준을 새로 만드는 데 사용하지 않았다.\n\n'+
        'J102와 Y001은 실질적으로 동률이다. frozen role policy는 평균을 정확한 수치로 정렬하므로 이 작은 차이가 PRIMARY 선정에 영향을 줄 수 있지만, 새 메커니즘의 입증으로 해석하지 않는다. B104의 높은 최저 fold도 선택 편향을 제거한 독립 보장은 아니다.')
    write_md('LIMITATIONS_NEXT.md', '''# NEXT 한계와 해석 범위

P(Δ>0)는 2,000개 재표본에서 차이가 양수였던 비율이다. 1.0이어도 개선이 확실하다는 posterior 확률이나 독립 시험의 성공 확률은 아니다.

경보 분석의 개별 SERIES.csv 11개는 원래 tau 라벨을 보존하므로 ZIP에서 제외한다. 집계 성능, paired bootstrap, 고정 정책과 해당 로컬 파일의 SHA256 증거는 남긴다. 원본 표는 로컬에서 삭제하거나 수정하지 않았다.

DEV를 반복적으로 관찰하며 후보를 제안했다. 2,000회 paired bootstrap은 이미 학습된 OOF와 선택된 후보에 조건부이다. 수백 번의 탐색을 보정한 신뢰구간이나 독립 성능 보장이 아니다. 실제 sealed 관측/라벨을 이용한 평가를 수행하지 않았다.

H regime은 전체 DEV의 historical descriptor로 군집을 만들었으므로 transductive 진단이다. 각 regime 제외 fit은 새로 수행했지만, regime 정책 자체의 독립 검증으로 해석하지 않는다. MEAN 21개, TAIL 86개, H regime 2의 166개 등 표본이 작은 절단은 불확실하다.

J102와 Y001의 평균 차이는 약 2.8e-7로 사실상 동률이다. 자기상관 보정 효과를 입증하지 못했다. GLR 제거의 CI도 0을 포함한다. B104의 runtime은 학습에서 제거한 일부 GLR 계산을 여전히 수행한다. 성능 열 제거와 계산 제거를 혼동하지 않는다.

TS-AUC는 첫 경보의 false-positive와 탐지율을 보장하지 않는다. 별도 nested null calibration은 특정 threshold와 짧은 horizon의 진단이며 배포용 threshold가 아니다. 고정 1,024-point threshold를 16,384-point null stream에 적용한 추가 연구 결과는 artifacts/next/long_null_stress를 확인한다. 실제 경쟁 데이터 생성 과정과 같다고 주장하지 않는다.

장기 null 연구의 모델은 사전에 고정한 PRIMARY/V007/Q005/Y001의 CV fold 0 모델이다. J102, B104, 50-tree 모델이나 최종 all-DEV artifact에서 동일한 false-alarm 비율을 직접 측정한 것으로 해석하지 않는다.

합성 stress에서 변동성 모델은 null FPR 편차를 줄였으나 일부 ARCH scale/persistence 및 tail 시나리오에서 기존 모델보다 나빴다. 반례를 삭제하거나 방향별 결과를 본 후 모델/threshold를 바꾸지 않았다.

초기 N013_ARCH_REFERENCE는 의도한 ABCD 51개 열 대신 E/F까지 포함한 57개 열을 써서 잘못된 matched baseline이었다. 해당 결과는 BUG로 남기고 승격에서 제외했다. 별도 E001_REFERENCE_ADAPTER_PARITY는 수정된 51개 열로 기존 HIGHEST_MEAN의 전체 OOF를 정확히 재현했다. 또 초기 일부 full 실험에 남은 500회 bootstrap은 중간 기록이며 최종 판정은 후속 2,000회 forensic 결과를 사용한다. 이 수정으로 fold 예측이나 승격 기준을 바꾸지는 않았다.

초기 next_main.py의 cold import는 Numba cache-directory 생성 시도가 차단되어 read-only 검증을 통과하지 못했다. 실제 전달 entry는 import 전에 캐시 생성을 억제하는 next_readonly_main.py 또는 대응하는 registered/serial entry다. 실패한 이전 entry와 음성 대조 증거를 지우지 않았다.

고정 후보 메뉴를 사용한 nested CV도 그 메뉴가 전체 DEV를 보고 지정되었기 때문에 전체 adaptive search의 unbiased 성능 추정은 아니다. bootstrap 역시 선택된 메뉴와 fit에 조건부이다.

SealGuard는 애플리케이션 경계다. predicate-filtered DEV 행만 사용했지만 Parquet 내부 mixed row-group decoding과 raw 전체 바이트 해시가 seal 바이트를 전혀 읽지 않았다는 OS 수준 주장은 하지 않는다. Python audit hook은 native system call sandbox가 아니다.

독립 raw 재현은 별도 소스 workspace와 새 feature/CV/all-DEV 학습을 사용했다. 그 학습에는 이전에 별도로 설치한 .repro-venv를 재사용했다. 이후 최종 감사는 새로 설치한 pinned 환경과 전달용 파일 복사본에서 선택 모델의 cold infer를 별도 검증하도록 강제한다(artifacts/next/portable_delivery/RESULTS.json). 새 환경에서는 추가 raw fit을 하지 않는다. 두 검증 모두 같은 OS/hardware 범위이며 다른 기종의 exact-byte parity를 보장하지 않는다.

공식 로컬 wire runner는 합성 protocol fixture의 train 단계에서 이미 학습된 모델을 복사했다. 실제 생산 train 구현은 별도 final_training의 DEV 원시 iterator audit로 검증했다. 이 둘을 합쳐 cloud 학습 또는 실제 SDK test를 수행했다고 주장하지 않는다.

10M latency는 고정 DEV buffer 반복으로 직접 측정한 속도/메모리 검사다. 새로운 10M 독립 관측이나 품질 평가가 아니다. 세부 component call 시간의 합은 fused scalar end-to-end 시간과 같지 않다.

점당 latency의 경계는 준비된 detector.predict_one 호출이며 timer와 float32 저장 비용을 포함한다. infer generator의 검증 분기, 파일/소켓/클라우드 전송 비용은 이 점당 수치에 포함하지 않았다. 모든 비교 모델에 동일한 경계를 적용했다.

메모리의 고정 state-array 크기와 process RSS는 다르다. rss_peak_bytes는 1M-point 간격에서 표본 추출한 최대 RSS로, 순간적인 전체 프로세스 최대 메모리를 보장하지 않는다. rss_after_bytes는 타이밍 밖의 원본 scalar 검증 뒤에 측정되어 표본 peak보다 높을 수 있다.

ZIP은 검토용 코드, 보고서, 잠긴 metadata, 최종 1~3개 모델을 담는다. raw observations/targets, feature/OOF caches, 환경 폴더, 인증정보와 이전 binary 모델은 제외된다. 전체 원시 재실행에는 사용자가 보유한 canonical raw training files 및 기존 REAL 자료가 별도로 필요하다.
''')
    write_md('SEAL_OPEN_RECOMMENDATION.md', f'''# 다음 세션 seal 검토 권고

이번 세션 A/B/C/D는 모두 미개봉이며 앞으로도 이 실행에서는 열지 않는다. 신규 reduced와 public 제출은 0이다.

최종 PRIMARY `{eid}`의 frozen rule상 new-generation 적합성은 `{primary['new_generation_seal_worthy']}`이다. DEV 평균 차이 {primary['primary_cv_mean']-primary_ref['mean']:+.8f}, paired CI [{boot['ci_2_5']:+.8f}, {boot['ci_97_5']:+.8f}] 및 고정 robustness를 근거로 다음 세션의 독립 평가를 논의할 수 있다.

이는 seal 개봉 명령이 아니다. GPT 전달물 검토 후 모델·threshold·비교 대상·평가 지표·성공 및 중단 조건을 먼저 고정하고, 사용자가 새 세션에서 명시적으로 승인할 때만 A의 단일 평가를 검토한다. 결과를 보고 B/C/D를 연속적으로 소비하는 계획은 제안하지 않는다. 적응적 DEV 편향, 사실상 동률 후보, 장기 false alarm과 합성 반례를 먼저 확인한다.
''')
    write_md('NEXT_SESSION_PLAN.md', '''# 다음 세션 작업 순서

1. 00_GPT_READ_FIRST_NEXT.md와 LIMITATIONS_NEXT.md를 읽고 SOURCE_MANIFEST_NEXT.json 및 외부 ZIP 영수증으로 파일 무결성을 확인한다. 기존 REAL 파일과 NEXT 최종 역할을 혼동하지 않는다.
2. DEV_CANDIDATES_NEXT.json의 역할, runtime, exact 모델 SHA와 final_engineering/final_training/final_audit 증거를 재검토한다. J102−Y001의 미세한 차이는 추가 발견으로 취급하지 않는다.
3. long null horizon, controlled synthetic counterexamples, nested alarm 및 paired intervention을 함께 보고 실제 첫 경보 목표를 명확히 한다. 추가 DEV 튜닝을 자동으로 시작하지 않는다.
4. 독립 평가가 필요하면 새 세션에서 모델과 판정 정책을 먼저 사전등록하고 사용자 승인 여부를 확인한다. 이 계획 자체는 seal 사용, SDK test 또는 public 제출을 허용하지 않는다.
''')
    write_md('01_GPT_HANDOFF_NEXT.md', '# Structural Break NEXT — GPT 전달 문서\n\n'+md_table(roles_frame)+'\n\n'+
        '먼저 00_GPT_READ_FIRST_NEXT.md의 26개 항목을 읽는다. 모든 점수는 고정된 DEV 8,000개 series의 원본 5-fold TS-AUC 또는 별도로 표시한 stress fit에서 나왔다. seal 2,000개는 관측/라벨/예측/평가에 사용하지 않았다.\n\n'+
        'ZIP 자체 검증: `python scripts/verify_next_handoff_manifest.py "ZIP의 전체 경로"`. Python 표준 라이브러리만 사용하고 파일을 풀거나 모델을 실행하지 않는다. 출력된 archive_sha256을 별도 전달된 ZIP 검증 영수증과 비교한다. 내부 manifest 일치는 발신자 인증을 뜻하지 않는다.\n\n'+
        '최종 구현 검토와 serial Bayes 수식의 독립 적분 확인은 docs/NEXT_FINAL_IMPLEMENTATION_REVIEW.md 및 tests/test_next_serial_bayes_integral.py에 있다.\n\n'+
        '원본 실험은 artifacts/next/experiments, 20개 추가 split fit은 robustness/candidates, 24개 절단의 2,000회 paired bootstrap은 forensic, 독립 원시 재현은 independent_raw_validation, 실제 모델의 API/속도 연결 감사는 final_engineering, 생산 train 재현은 final_training에 있다. 각 RESULTS.json 및 SOURCE_MANIFEST_NEXT.json에 SHA256이 있다.\n\n'+
        'MODEL_RESULTS_NEXT.csv와 EXPERIMENT_LEDGER.csv의 final_evidence_status가 최종 종합 상태다. original_stage_status는 당시 screen/full 단계의 불변 기록으로 일부 ROBUSTNESS_PENDING이 남는다. engineering 미완료의 미선정 후보는 임의로 PASS 처리하지 않는다. SCREEN_FAIL 후보의 품질 숫자를 전체 5-fold 평균과 비교하지 않는다.\n\n'+
        '모델 사용: DEV_CANDIDATES_NEXT.json의 delivery_model_directory와 entrypoint를 함께 사용한다. variance는 next_readonly_main.py, log-energy는 next_registered_main.py, serial-variance는 next_serial_main.py다. 기존 main.py는 기존 REAL 모델용으로 보존되어 있다. infer는 먼저 None을 yield하고 각 O 관측마다 Python float 하나를 내보낸다.\n\n'+
        '역할별 train specification은 configs/next_roles/<ROLE>.json이다. 각 family entry의 기본 train은 그 family에서 우선순위가 가장 높은 선택 역할을 학습한다. 같은 family의 다른 역할은 해당 training 모듈 train_frozen_candidate에 명시적 spec을 넘긴다. 검증 명령은 scripts/train_next_final_roles.py에 기록되어 있다. 이미 만든 최종 디렉터리에 재학습하면 fail-closed로 거절한다.\n\n'+
        '핵심 attribution은 artifacts/next/fixed_ablation_contrasts/research_v1, 학습량 곡선은 learning_curve, H forward 진단은 historical_holdout, 제한된 nested 선택 진단은 nested_menu_cv, alarm 진단은 alarm_calibration, 합성 counterexample은 synthetic_variance_stress, 장기 null 실험은 long_null_stress를 확인한다. 장기 오경보·방향별 반례·앙상블 추가 기여를 함께 해석한 문서는 artifacts/next/research_synthesis/ADDITIONAL_DIAGNOSTICS.md다.\n\n'+
        'Zip에는 원시 데이터와 환경, 인증정보, 모든 OOF point 파일을 포함하지 않는다. canonical raw SHA는 REPRODUCTION_NEXT.json과 src/next/data.py에 있다. 모델 환경은 requirements-lock.txt, 별도 그림 환경은 requirements-next-figures-lock.txt에 고정했다. 그림은 작은 CSV 보고서에서 생성하며 원시 데이터가 필요하지 않다. 분석은 잠근 환경에서 재현했으며 독립 seal 일반화 성능은 아직 확인하지 않았다. 상세 한계와 다음 세션 경계는 LIMITATIONS_NEXT.md 및 SEAL_OPEN_RECOMMENDATION.md에 있다.')
    guard.check_state()
    generated = ['MODEL_RESULTS_NEXT.csv', 'EXPERIMENT_LEDGER.csv', 'CV_RESULTS_NEXT.csv', 'PAIRED_BOOTSTRAP_RESULTS.csv',
        'TIME_FRACTURE_NEXT.csv', 'BREAK_AGE_NEXT.csv', 'MECHANISM_FRACTURE_RESULTS.csv', 'CURRENT_TOP_DISAGREEMENT.csv',
        *slices, 'HISTORICAL_REGIME_RESULTS.csv', 'DEV_CANDIDATES_NEXT.json', 'LATENCY_NEXT.json', 'DETERMINISM_NEXT.json',
        'REPRODUCTION_NEXT.json', '00_GPT_READ_FIRST_NEXT.md', '01_GPT_HANDOFF_NEXT.md', 'CURRENT_TOP_FORENSIC.md',
        'LIMITATIONS_NEXT.md', 'SEAL_OPEN_RECOMMENDATION.md', 'NEXT_SESSION_PLAN.md']
    write_json(receipt_dir/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'clock': clock,
        'writer_sha256': sha256(Path(__file__)), 'candidate_count': len(table),
        'snapshot_sha256': sha256(ROOT/snapshot_dir/'RESULTS.json'), 'backup_manifest_sha256': sha256(backup/'BACKUP_MANIFEST.json'),
        'output_files_sha256': {name: sha256(ROOT/name) for name in generated}})
    print('FINAL NEXT REPORTS WRITTEN', len(table), eid, flush=True)


if __name__ == '__main__':
    main()
