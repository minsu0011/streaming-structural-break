"""Summarize the complete D006 diagnostic without searching individual EWMA cells."""
from pathlib import Path
import importlib.metadata
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now

LABELS = {'LEGACY_AR_FILTER': 'Legacy AR filter', 'LEGACY_ARCH_PREVIOUS': 'Legacy ARCH previous state',
    'LEGACY_LAG_RING': 'Legacy standardized lag ring', 'LEGACY_EWMA': 'Legacy sparse EWMA cells',
    'LEGACY_CUSUM_POSITIVE': 'Positive CUSUM (unused)', 'LEGACY_CUSUM_NEGATIVE': 'Negative CUSUM (unused)',
    'LEGACY_RUNNING_MEAN': 'Running mean'}
ANCHORS = (0, 127, 255, 511)
DIRECTIONS = ('CONTROL_INTO_OUTLIER', 'OUTLIER_INTO_CONTROL')


def main():
    for line in (ROOT/'requirements-next2-figures-lock.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'): continue
        name, version = line.split('=='); assert importlib.metadata.version(name) == version
    source = BASE/'legacy_state_transfer/run_v1'
    receipt = read_json(source/'RESULTS.json'); assert receipt['status'] == 'PASS'
    for name, sha in receipt['files_sha256'].items(): assert digest(source/name) == sha
    columns = read_json(BASE/'legacy_state_transfer/TRAINED_LEGACY_COLUMN_INSPECTION.json')
    assert not columns['legacy_positive_CUSUM_column_used'] and not columns['legacy_negative_CUSUM_column_used']
    assert columns['legacy_running_mean_column_used']
    frame = pd.read_csv(source/'WINDOW_COUNTERFACTUAL_INTERVALS.csv', float_precision='round_trip')
    index = pd.read_csv(source/'PAIR_WINDOW_ARRAY_INDEX.csv', float_precision='round_trip')
    assert len(frame) == len(index) == 630
    with np.load(source/'PAIR_WINDOW_STATISTICS.npz', allow_pickle=False) as arrays:
        for (_, row), (_, key) in zip(frame.iterrows(), index.iterrows()):
            vector = arrays[key['array']]
            assert vector.shape == (row.seed_pairs, 3)
            np.testing.assert_allclose(vector.mean(axis=0), [row.control_mean, row.outlier_mean, row.hybrid_mean], rtol=0, atol=1e-14)
    final = frame[(frame.window_end_exclusive == 512)|(frame.window_end_exclusive == 768)]
    probability = final[final.metric == 'probability_mean']; assert len(probability) == 56
    out = BASE/'figures/legacy_state_transfer_v1'; out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for ax, anchor in zip(axes.flat, ANCHORS):
        for di, direction in enumerate(DIRECTIONS):
            rows = probability[(probability.anchor_offset == anchor)&(probability.direction == direction)].set_index('block').loc[list(LABELS)]
            mean = rows.delta_from_destination.to_numpy()
            ax.errorbar(mean, np.arange(7)+(.12 if di else -.12),
                xerr=np.stack([mean-rows.delta_from_destination_ci_low, rows.delta_from_destination_ci_high-mean]),
                fmt='o', ms=4, capsize=3, color='#176b91' if di == 0 else '#b65b35',
                label='Control into outlier' if di == 0 else 'Outlier into control')
        ax.set_yticks(np.arange(7), list(LABELS.values())); ax.axvline(0, color='#444444', lw=.7)
        ax.grid(axis='x', alpha=.15)
        ax.set_title(f'Anchor +{anchor}; '+('512 original pairs' if anchor < 511 else '342 original pairs'))
        ax.set_xlabel('Mean probability change from destination state')
    axes[0, 0].invert_yaxis(); axes[0, 0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('Legacy EWMA state sustains the registered late-window response', fontsize=14, y=1.01)
    fig.tight_layout(h_pad=2.5)
    fig.text(.03, -.025, 'Anchors 0/127/255: [256,512). Anchor 511: [512,768), a separate availability-defined cohort.\nConditional paired 95% intervals; effects are not additive. CUSUM columns are absent from the frozen trained tree inputs.', fontsize=9)
    files = []
    for extension in ('png', 'pdf'):
        name = '01_legacy_final_window_interventions.'+extension
        fig.savefig(out/name, dpi=190, bbox_inches='tight'); files.append(name)
    plt.close(fig)
    lines = ['# NEXT3 legacy 상태 세분화 결과', '',
        '기존 O1 합성 사례에서 legacy 상태의 후기 반응은 주로 EWMA 상태 개입에 의해 달라졌습니다. '
        '앵커 127/255의 [256,512) 창에서 대조군 EWMA를 이상치 상태에 넣으면 평균 확률이 약 0.114823 감소하고, '
        '반대 방향에서는 약 0.115508 증가했습니다. 원본 미래 관측은 동일하며 개입은 평가용입니다.', '',
        '앵커 0의 AR 필터·lag ring 교환 효과가 후기 창에도 남지만 앵커 127부터는 두 상태의 교환 효과가 0입니다. '
        '반면 EWMA 효과는 유지됩니다. 필터의 초기 차이가 누적 상태로 전달되는 해석과 부합합니다. '
        'EWMA 안의 개별 셀이나 시간척도를 추가 탐색하지 않았으므로 160 척도 하나만 고유 원인이라고 단정하지 않습니다.', '',
        '양/음 CUSUM의 0 효과는 메모리 문제가 해결되었다는 증거가 아닙니다. 동결된 tree-used 모델이 legacy 열 3/4를 '
        '사용하지 않아 해당 CUSUM 상태가 활성 입력에 포함되지 않습니다. Running mean 열 5는 사용되지만 후기 영향은 훨씬 작습니다.', '',
        '모든 원본 J102/C205/BL80 예측 3,145,728개, 복제 재생 3,756회, 7개 상태 전체와 기존 legacy 전체 교환의 비교 '
        '3,756회가 정확히 일치했습니다. D005의 legacy 평균 궤적 2,814개와 90개 창 결과도 정확히 재구성했습니다. '
        '630개 새 세부 창/지표와 9,476,096개 평가용 hybrid 예측을 계산했으며 신규 fit·후보·DEV 관측 접근·seal 접근은 0입니다.', '',
        '| Anchor | Direction | Block | Probability delta | Conditional 95% interval |',
        '|---:|---|---|---:|---|']
    for _, row in probability.iterrows():
        lines.append(f'| {int(row.anchor_offset)} | {row.direction} | {row.block} | {row.delta_from_destination:+.9f} | '
                     f'[{row.delta_from_destination_ci_low:+.9f}, {row.delta_from_destination_ci_high:+.9f}] |')
    lines += ['', '앵커 511은 기존 onset 128/256인 342쌍만 사용합니다. [512,768) 창에서 EWMA 효과는 여전히 남지만 '
        '다른 코호트의 앞 창과 직접 비교해 감소율을 추정하지 않습니다. D005의 같은 늦은 코호트에서는 J102 serial evidence의 '
        '교환 효과도 지속되었습니다.', '',
        '이 결과는 실제 DEV에서 검증된 수정 모델이 아닙니다. 각 상태 효과는 상호작용하고 일반적인 입력 이력에서 '
        '도달하기 어려운 hybrid일 수 있습니다. 2,000개 고정 whole-pair draw의 구간은 적응적 진단 선택·다중 비교를 보정하지 않습니다. '
        '원본 후보와 threshold를 변경하거나 EWMA를 임의로 reset하는 배포 코드는 만들지 않았습니다.', '',
        '전체 자료: `artifacts/next3/legacy_state_transfer/run_v1/`. '
        '정적 입력 열 확인: `artifacts/next3/legacy_state_transfer/TRAINED_LEGACY_COLUMN_INSPECTION.json`.', '']
    report = ROOT/'docs/NEXT3_LEGACY_STATE_TRANSFER_FINDINGS.md'; report.write_text('\n'.join(lines), encoding='utf-8')
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'source_result_sha256': digest(source/'RESULTS.json'), 'aggregate_rows_reconstructed': 630,
        'column_inspection_sha256': digest(BASE/'legacy_state_transfer/TRAINED_LEGACY_COLUMN_INSPECTION.json'),
        'files_sha256': {name: digest(out/name) for name in files}, 'findings_sha256': digest(report),
        'new_fits': 0, 'new_quality_scores': 0, 'visual_review': 'PENDING'}, immutable=True)
    progress('D006 results summarized', 'All seven blocks and both directions; inactive CUSUM columns explicitly distinguished from resolved persistence')
    print('NEXT3 D006 FINDINGS COMPLETE', flush=True)


if __name__ == '__main__': main()
