"""Algebraic and observed redundancy of the original bounded confirmation rules."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note, require_guard

EVENT_COLUMNS = ('any_latch', 'first_latch', 'control_any_latch', 'excess_latch_vs_paired_control',
    'prebreak_latch', 'first_detection', 'first_detection_delay', 'scale_only_latch', 'scale_first_latch',
    'scale_first_detection', 'scale_first_detection_delay', 'core_only_latch', 'event_memory_final',
    'scale_event_memory_final', 'event_memory_delta_final', 'event_score_mean',
    'confirmed_episodes_capped', 'first_confirmation_time')


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    require_guard()
    calibration = read_json(BASE / 'SCALE_CALIBRATION_FROZEN.json')
    proof_rows, comparisons, files = [], [], {}
    with activity('Algebraic and observed original confirmation rule equivalence', 'FAILURE_DEDUPLICATION', 'F2_FRESH_CONFIRMATION'):
        for table in calibration['tables']:
            for index, scale in enumerate(table['scale_order']):
                c1, c2 = table['cutoffs'][index][1:]
                proof_rows.append({'H_length': table['H_length'], 'scale': scale,
                    'C1_3OF4_cut': c1, 'C2_2OF4_cut': c2, 'same_cut': c1 == c2,
                    'maximum_strength_when_exactly_two_positive': 2,
                    'confirmed_scale_event_equivalence_proven': c1 == c2 and c1 >= 2,
                    'strict_threshold_comparison': True, 'current_unconfirmed_strength_equivalence_claim': False,
                    'scope': 'Same bounded four-block input and common cut. Applies to confirmed scale events and event-memory updates; does not equate subthreshold current scores.'})
        cases = read_json(BASE / 'NEXT7_PROTOCOL_LOCK.json')['evaluation']['cases']
        for case in cases:
            path = BASE / 'synthetic_g1' / case / 'SERIES.csv'
            done = read_json(path.parent / 'DONE.json')
            assert done['status'] == 'COMPLETE' and digest(path) == done['SERIES_sha256']
            records = rows(path)
            files[path.relative_to(BASE).as_posix()] = digest(path)
            for scale in ('S1_BLOCK_LOG_MAD', 'S2_BLOCK_SCALE_PIT'):
                first = {(int(row['seed']), int(row['horizon'])): row for row in records
                         if row['variant'] == scale + '_C1_SOFT_3OF4'}
                second = {(int(row['seed']), int(row['horizon'])): row for row in records
                          if row['variant'] == scale + '_C2_ACCUMULATED_4BLOCK'}
                assert set(first) == set(second)
                for horizon in done['horizons']:
                    keys = sorted(key for key in first if key[1] == horizon)
                    assert [key[0] for key in keys] == list(range(512))
                    event_differences = {name: sum(first[key][name] != second[key][name] for key in keys) for name in EVENT_COLUMNS}
                    if scale == 'S1_BLOCK_LOG_MAD':
                        assert all(count == 0 for count in event_differences.values()), (case, horizon, event_differences)
                    comparisons.append({'case': case, 'H_length': 2048, 'horizon': horizon, 'scale': scale,
                        'paired_seed_paths': len(keys), 'all_reported_event_summaries_equal': all(count == 0 for count in event_differences.values()),
                        **{name + '_differing_seed_paths': count for name, count in event_differences.items()},
                        'current_score_mean_differing_seed_paths': sum(first[key]['current_score_mean'] != second[key]['current_score_mean'] for key in keys),
                        'confirmation_max_differing_seed_paths': sum(first[key]['confirmation_max'] != second[key]['confirmation_max'] for key in keys),
                        'new_candidate_or_threshold': False})
        csv_write('CONFIRMATION_RULE_ALGEBRA_NEXT7.csv', proof_rows)
        csv_write('CONFIRMATION_RULE_DEDUPLICATION_NEXT7.csv', comparisons)
        save('CONFIRMATION_RULE_EQUIVALENCE_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'source_sha256': digest(__file__), 'calibration_sha256': digest(BASE / 'SCALE_CALIBRATION_FROZEN.json'),
            'input_series_sha256': files, 'algebraic_bound': 'With exactly two positive clipped values in[-4,4], signed four-block mean is at most(4+4+0+0)/4=2. A strict cut>=2 excludes every extra C2-only confirmation.',
            'S1_H2048_all23_cases_event_summary_parity': True,
            'all_point_state_prediction_identity_claim': False, 'new_draws': 0, 'new_fits': 0,
            'output_sha256': {name: digest(BASE / name) for name in
                ('CONFIRMATION_RULE_ALGEBRA_NEXT7.csv', 'CONFIRMATION_RULE_DEDUPLICATION_NEXT7.csv')}})
        save('research_notes/BOUNDED_RULE_EQUIVALENCE_KO.md',
            '# 확인 규칙의 유효한 중복과 그 범위\n\n'
            'C1과 C2는 최근 네 블록의 부호 있는 평균 증거를 사용한다. 각 증거는 -4부터4까지 제한된다. '
            'C1은 같은 방향의 양의 증거가 세 개 이상, C2는 두 개 이상일 때 평균을 허용한다. '
            '두 규칙이 달라질 수 있는 추가 경우는 정확히 두 개만 양의 증거인 경우다. 이때 평균의 최대값은 (4+4+0+0)/4=2다.\n\n'
            '따라서 두 규칙에 같은 기준을 쓰고 그 기준이2 이상이면, 엄격한 초과(>) 조건을 만족하는 추가 C2 사건은 생길 수 없다. '
            '현재 S1의 H=1000,1500,2048 고정 기준은 이 조건을 만족한다. H=3000,5000 또는 다른 기준까지 이 결론을 넓히지 않는다.\n\n'
            '기존 H=2048의23개 시나리오, 각512개 고정 시드와 모든 보고 시점을 대조했다. '
            'S1의 첫 기억 시점, 최종 기억 강도, 사건 기억 평균, 탐지 지연과 발생 횟수 등18개 사건 요약이 모두 정확히 같았다. '
            '확인 기준을 넘기 전의 강도와 현재 상태 점수는 다를 수 있다. 전체 예측기나 모든 상태 좌표가 같다는 주장은 하지 않는다.\n\n'
            '이 결과는 이미 실행한 규칙의 중복을 정리한 것이다. 새 규칙, 새 기준, 새 학습 또는 성능에 따른 규칙 교체를 추가하지 않았다. '
            '수치 근거는 CONFIRMATION_RULE_ALGEBRA_NEXT7.csv와 CONFIRMATION_RULE_DEDUPLICATION_NEXT7.csv에 보존했다.\n')
    note('Bounded confirmation rule redundancy identified', 'S1 C1/C2 confirmed-event equivalence at common cuts>=2; current subthreshold scores remain distinct', 'F2_FRESH_CONFIRMATION')


if __name__ == '__main__':
    main()
