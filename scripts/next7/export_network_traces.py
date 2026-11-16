"""Export reproducible synthetic successes AND failures for the dynamic viewer."""
from pathlib import Path
import csv
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, save, read_json, digest, utc_now, activity, require_guard
from src.next7.null_growth import QualifiedState
from src.next7.calibration import load_calibration, cutoff
from src.next6_extension.fresh_runtime import SafeState
from src.next6_extension.synthetic import generated

COORDINATES = {'current_state': 0, 'current_score': 1, 'ratio': 2, 'raw_event': 3,
    'signed_scale': 9, 'scale_ratio': 11, 'shock': 14, 'block_complete': 15,
    'raw_fresh': 18, 'first_shock': 19, 'first_raw_confirm': 20, 'qualified_event': 30,
    'qualified_first': 35, 'qualified_last': 36, 'qualified_channel': 37,
    'qualified_oldest': 38, 'qualified_same_direction': 39, 'qualified_fresh': 40,
    'admission_cut': 42, 'admission_support': 43, 'qualified_direction': 34}


def cohort(case, horizon=1024):
    with (BASE / f'synthetic_g2_full/{case}/SERIES.csv').open(newline='', encoding='utf-8') as f:
        return sorted((r for r in csv.DictReader(f) if int(r['horizon']) == horizon), key=lambda r: int(r['seed']))


def choose(case, predicate, horizon=1024):
    values = [r for r in cohort(case, horizon) if predicate(r)]
    if not values:
        raise RuntimeError('No qualifying visual example: ' + case)
    return int(values[0]['seed'])


def main():
    require_guard()
    directory = BASE / 'visuals'
    if (directory / 'STATE_NETWORK_TRACES.json').exists():
        raise RuntimeError('Preserve source-locked visual traces')
    selected = [
        {'id': 'scale_down_detected', 'case': 'V4', 'title': '분산 축소를 확인한 경로',
         'seed': choose('V4', lambda r: r['first_detection'] == 'True'), 'horizon': 1024,
         'selection': 'V4, qualified first detection within1024; smallest fixed seed index',
         'explanation': '음의 scale 증거가 새 블록에서 반복되고 H 기준을 통과해 사건 기억이 생깁니다.'},
        {'id': 'recovery_with_memory', 'case': 'RETURN_NORMAL', 'title': '현재는 회복, 과거 사건은 유지',
         'seed': choose('RETURN_NORMAL', lambda r: r['first_detection'] == 'True'
             and int(r['first_latch']) < int(r['tau_offline_only']) + 256
             and r.get('memory_retained_after_current_recovery') == 'True'), 'horizon': 1024,
         'selection': 'First qualified event during the256-point intervention, retained during normal current state; smallest seed',
         'explanation': '일시적 변화가 끝나도 공식 target은 1입니다. 현재 상태와 과거 사건을 따로 읽어야 합니다.'},
        {'id': 'null_false_memory', 'case': 'N0', 'title': '실패 사례: 정상 경로의 오기억',
         'seed': choose('N0', lambda r: r['any_latch'] == 'True'), 'horizon': 1024,
         'selection': 'N0 qualified false latch within1024; smallest seed',
         'explanation': '변화가 없는 경로에서도 확정이 생길 수 있습니다. 확정 기억은 정답 또는 확률 보증을 뜻하지 않습니다.'},
        {'id': 'single_shock_unlatched', 'case': 'O1', 'title': '한 번의 충격을 기억하지 않은 경로',
         'seed': choose('O1', lambda r: r['any_latch'] == 'False' and int(r['first_shock_time']) >= 0), 'horizon': 1024,
         'selection': 'O1 raw shock observed, no qualified latch within1024; smallest seed',
         'explanation': '충격 표시만으로는 사건 기억이 갱신되지 않습니다. 블록 확정과 H 기준 통과가 필요합니다.'},
        {'id': 'burst_false_memory', 'case': 'OB16', 'title': '실패 사례: 16개 충격 관측의 오기억',
         'seed': choose('OB16', lambda r: r['excess_latch_vs_paired_control'] == 'True'), 'horizon': 1024,
         'selection': 'OB16 qualified latch with paired clean control unlatched; smallest seed',
         'explanation': '16개 연속 오염 관측은 확정 조건을 우연히 통과할 수 있습니다. 남아 있는 실패를 함께 표시합니다.'},
        {'id': 'scale_down_missed', 'case': 'V4', 'title': '실패 사례: 분산 축소를 놓친 경로',
         'seed': choose('V4', lambda r: r['any_latch'] == 'False'), 'horizon': 1024,
         'selection': 'V4 with no qualified latch through1024; smallest seed',
         'explanation': '분산 축소가 있어도 정해진 관측 범위 안에서 사건을 확인하지 못할 수 있습니다.'},
        {'id': 'late_persistent', 'case': 'LATE_PERSISTENT', 'title': '늦게 발생한 지속적 변화',
         'seed': choose('LATE_PERSISTENT', lambda r: r['first_detection'] == 'True', 4096), 'horizon': 4096,
         'selection': 'Late persistent change first detected through4096; smallest seed',
         'explanation': '관측 블록 수가 늘면 H 기준도 달라집니다. 총 미래 길이는 예측에 제공되지 않습니다.'},
        {'id': 'insufficient_H', 'case': 'N0', 'title': 'H가 짧아 사건 확정을 보류한 경로',
         'seed': 0, 'horizon': 1024, 'H_suffix': 512,
         'selection': 'Fixed N0 seed0 with most recent512 H observations; explicit support-contract illustration',
         'explanation': 'G2의 H 재분할 기준이 부족해 새 사건 기억을 보류합니다. 현재 상태 출력은 계속 계산합니다.'},
    ]
    save('visuals/TRACE_SELECTION_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'selected': selected, 'selection_is_illustrative_not_population_evidence': True,
        'success_and_failure_both_required': True, 'raw_competition_data_included': False})
    cal = load_calibration()
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    traces = []
    with activity('export actual synthetic state network traces', 'VISUAL_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        for choice in selected:
            h, o, tau, info = generated(choice['case'], choice['seed'], partition=31,
                horizon=choice['horizon'], control=False, lock=suite)
            if 'H_suffix' in choice:
                h = h[-choice['H_suffix']:]
            state = QualifiedState(h, core_cut=cal['core_cut'], scale_cut=cutoff(cal, len(h), 0, 1))
            out = state.replay(o)
            core = SafeState(h, threshold=cal['core_cut']).replay(o)
            blocks = []
            for t in np.flatnonzero(out[:, 15] > 0):
                blocks.append({'time': int(t), 'block': int(out[t, 16]) - 1,
                    'z': np.round(np.r_[core[t, 12:20], out[t, 9]], 6).tolist(),
                    'ratio': round(float(out[t, 2]), 6), 'cut': round(float(out[t, 42]), 6),
                    'raw_confirmed': bool(out[t, 18]), 'qualified': bool(out[t, 40])})
            first = int(out[-1, 35])
            initial = min(len(o) - 1, first + 32 if first >= 0 else info['scheduled'] + 96)
            if choice['id'] == 'recovery_with_memory':
                normal = np.flatnonzero((np.arange(len(o)) >= info['intervention_end'] + 128)
                    & (out[:, 0] == 1) & (out[:, 30] > 0))
                assert len(normal)
                initial = int(normal[0])
            traces.append({**choice, 'tau_offline_only': tau, 'info': info, 'H_length': len(h),
                'initial_time': initial, 'calibration': state.calibration,
                'raw': np.round(o.astype(float), 6).tolist(),
                'values': {name: np.round(out[:, column], 6).tolist() for name, column in COORDINATES.items()},
                'blocks': blocks, 'bounded_mutable_state_bytes': state.bounded_mutable_state_bytes})
        save('visuals/STATE_NETWORK_TRACES.json', {'created_utc': utc_now(), 'traces': traces,
            'precision': 'Six decimals for display; scientific scores use original float64 artifacts',
            'state_names': ['준비', '정상', '충격', '미확정', '지속 확정', '회복 중', 'H 부족'],
            'channel_names': ['rank 평균', 'rank 에너지', 'tail 빈도', 'rank 비대칭', 'lag1 rank', 'lag4 rank', 'lag1 energy', 'lag4 energy', 'signed scale'],
            'graph_semantics': 'Arrows count observed changes of current operational state up to the selected time; not learned causal edges.',
            'event_semantics': 'Qualified historical evidence; not a calibrated probability or known ground truth.',
            'source_sha256': {n: digest(ROOT / n) for n in ('scripts/next7/export_network_traces.py', 'src/next7/null_growth.py')},
            'G2_protocol_sha256': digest(BASE / 'G2_PROTOCOL_LOCK.json'), 'raw_competition_data_included': False})


if __name__ == '__main__':
    main()
