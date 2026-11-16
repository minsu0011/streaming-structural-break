"""Render every registered D005 block/direction from committed aggregates only."""
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

LABELS = {'LEGACY_ARCH_ABCD': 'Legacy ARCH / ABCD', 'SHARED_AR_FILTER': 'Shared AR filter',
    'CONDITIONAL_VARIANCE': 'Shared variance state', 'J_SERIAL_EVIDENCE': 'J102 serial evidence',
    'C_DEPENDENCE_LAGS': 'C205 lag state', 'C_DEPENDENCE_EVIDENCE': 'C205 evidence'}
DIRECTIONS = ('CONTROL_INTO_OUTLIER', 'OUTLIER_INTO_CONTROL')
ANCHORS = (0, 127, 255, 511)


def main():
    versions = {}
    lock = ROOT/'requirements-next2-figures-lock.txt'
    for line in lock.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'): continue
        name, version = line.split('==')
        assert importlib.metadata.version(name) == version
        versions[name] = version
    source = BASE/'outlier_state_transfer/run_v1'
    result = read_json(source/'RESULTS.json'); assert result['status'] == 'PASS'
    for name, sha in result['files_sha256'].items(): assert digest(source/name) == sha
    table = pd.read_csv(source/'WINDOW_COUNTERFACTUAL_INTERVALS.csv', float_precision='round_trip')
    curves = pd.read_csv(source/'MEAN_COUNTERFACTUAL_CURVES.csv', float_precision='round_trip')
    index = pd.read_csv(source/'PAIR_WINDOW_ARRAY_INDEX.csv', float_precision='round_trip')
    assert len(table) == len(index) == 540
    # Independent aggregate-to-table consistency: no real rows or point predictions.
    with np.load(source/'PAIR_WINDOW_STATISTICS.npz', allow_pickle=False) as arrays:
        for (_, row), (_, key) in zip(table.iterrows(), index.iterrows()):
            values = arrays[key['array']]; assert values.shape == (int(row.seed_pairs), 3)
            np.testing.assert_allclose(values.mean(axis=0), [row.control_mean, row.outlier_mean,
                                       row.hybrid_mean], rtol=0, atol=1e-14)
            destination = values[:, 1] if row.direction == DIRECTIONS[0] else values[:, 0]
            np.testing.assert_allclose((values[:, 2]-destination).mean(),
                                       row.delta_from_destination, rtol=0, atol=1e-14)
    out = BASE/'figures/outlier_state_transfer_v1'; out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    files = []
    def save(fig, name):
        for extension in ('png', 'pdf'):
            filename = name+'.'+extension
            fig.savefig(out/filename, dpi=180, bbox_inches='tight'); files.append(filename)
        plt.close(fig)
    colors = dict(zip(LABELS, plt.get_cmap('tab10').colors))
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharey=True)
    for ai, anchor in enumerate(ANCHORS):
        for di, direction in enumerate(DIRECTIONS):
            ax = axes[ai, di]
            data = curves[(curves.anchor_offset == anchor)&(curves.direction == direction)]
            for block, label in LABELS.items():
                rows = data[data.block == block]
                ax.plot(rows.post_injection_offset, rows.delta_from_destination,
                        label=label, color=colors[block], lw=1.1)
            ax.axhline(0, color='#444444', lw=.6); ax.grid(alpha=.12)
            ax.set_title(f'Anchor +{anchor}; '+('control into outlier' if di == 0 else 'outlier into control'))
            ax.set_xlabel('Points since original outlier'); ax.set_ylabel('Hybrid minus destination probability')
            ax.text(.97, .95, '342 original pairs' if anchor == 511 else '512 original pairs',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc='lower right')
    fig.suptitle('State interventions have different immediate and persistent effects', fontsize=14, y=1.01)
    fig.tight_layout(h_pad=2)
    fig.text(.05, -.018, 'Original O1 inputs and frozen BL80 only. Later anchors copy state after its update. Effects are nonadditive; the late cohort differs.', fontsize=9)
    save(fig, '01_all_anchor_probability_curves')

    final = table[((table.anchor_offset < 511)&(table.window_start_inclusive == 256)&(table.window_end_exclusive == 512))|
                  ((table.anchor_offset == 511)&(table.window_start_inclusive == 512)&(table.window_end_exclusive == 768))].copy()
    probability = final[final.metric == 'probability_mean']; assert len(probability) == 48
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, sharey=True)
    for ai, anchor in enumerate(ANCHORS):
        for di, direction in enumerate(DIRECTIONS):
            ax = axes[ai, di]
            rows = probability[(probability.anchor_offset == anchor)&(probability.direction == direction)].set_index('block').loc[list(LABELS)]
            mean = rows.delta_from_destination.to_numpy()
            ax.errorbar(mean, np.arange(6), xerr=np.stack([mean-rows.delta_from_destination_ci_low,
                rows.delta_from_destination_ci_high-mean]), fmt='o', ms=4, color='#176b91', capsize=3)
            ax.set_yticks(np.arange(6), list(LABELS.values()))
            ax.axvline(0, color='#444444', lw=.7); ax.grid(axis='x', alpha=.15)
            ax.set_title(f'Anchor +{anchor}; '+('control into outlier' if di == 0 else 'outlier into control'))
            ax.set_xlabel('Mean probability difference')
    axes[0, 0].invert_yaxis()
    fig.suptitle('All blocks and both directions in the final registered window', y=1.01, fontsize=14)
    fig.tight_layout(h_pad=2)
    fig.text(.03, -.03, 'Anchors 0/127/255: offsets [256,512), 512 pairs. Anchor 511: [512,768), 342 pairs.\n2,000 original paired draws; conditional 95% intervals do not correct diagnostic choice or multiplicity.', fontsize=9)
    save(fig, '02_final_window_conditional_intervals')

    alarm = final[final.metric == 'any_crossing']; assert len(alarm) == 96
    matrix = np.empty((24, 4)); labels = []
    for ai, anchor in enumerate(ANCHORS):
        for bi, block in enumerate(LABELS):
            row_index = ai*6+bi; labels.append(f'+{anchor}: {LABELS[block]}')
            for di, direction in enumerate(DIRECTIONS):
                for qi, alpha in enumerate((.05, .01)):
                    row = alarm[(alarm.anchor_offset == anchor)&(alarm.block == block)&
                                (alarm.direction == direction)&(alarm.alpha == alpha)]
                    assert len(row) == 1
                    matrix[row_index, 2*di+qi] = row.iloc[0].delta_from_destination
    fig, ax = plt.subplots(figsize=(9, 11))
    bound = max(.01, float(np.abs(matrix).max()))
    plot = ax.imshow(matrix, cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
    ax.set_yticks(np.arange(24), labels)
    ax.set_xticks(np.arange(4), ['Control into outlier\nalpha .05', 'Control into outlier\nalpha .01',
                               'Outlier into control\nalpha .05', 'Outlier into control\nalpha .01'])
    for row in range(24):
        for column in range(4):
            value = matrix[row, column]
            ax.text(column, row, f'{value:+.3f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(value) > .6*bound else '#222222')
    for boundary in (5.5, 11.5, 17.5): ax.axhline(boundary, color='white', lw=2)
    fig.colorbar(plot, ax=ax, label='Any-crossing probability difference from destination')
    ax.set_title('Frozen-threshold alarms: all blocks in the same final windows', pad=18)
    fig.text(.1, -.025, 'Window-local any crossing, not eventual alarm over the full horizon. Thresholds are unchanged.\nFull intervals and every earlier window remain in the committed 540-row source table.', fontsize=9)
    save(fig, '03_final_window_alarm_contrasts')

    lines = ['# NEXT3 O1 상태 개입 결과', '',
        '기존 512개 O1 쌍에서, 이상치의 장기 영향은 기존 ARCH/ABCD와 J102 serial evidence 상태를 통해 남았습니다. '
        'C205의 직접 상태 개입 효과는 후기 구간에서 훨씬 작았습니다. 이는 아래 고정 합성 실험의 결과이며 실제 DEV의 고유 원인이나 수정 모델의 성능을 입증하지 않습니다.', '',
        'D005는 원본 성분/블렌드 예측 3,145,728개, 복제 재생 3,756회, 전체 상태 교환 재생 3,756회를 정확히 재현했습니다. '
        '신규 학습·후보·seal 접근·threshold 재보정·합성 horizon 연장은 모두 0입니다.', '',
        '앵커 0에서는 AR 필터 교환이 후기 예측도 바꿉니다. 앵커 127 이후에는 같은 AR 필터 교환 효과가 0이지만 '
        'legacy 및 serial evidence 교환 효과는 남습니다. 초기에 필터를 거친 입력 차이가 다른 누적 상태로 전달되는 설명과 부합합니다. '
        '단일 블록 효과는 상호작용하므로 더해서 총 기여도로 해석하지 않습니다.', '',
        '아래는 사전 등록된 마지막 창의 모든 블록·앵커·방향입니다. 앵커 0/127/255는 512쌍의 [256,512), '
        '앵커 511은 기존 onset 128/256인 342쌍의 [512,768)입니다. 코호트 차이를 시간 경과 효과로 혼동하지 않습니다.', '',
        '| Anchor | Direction | Block | Probability delta | Conditional 95% interval |',
        '|---:|---|---|---:|---|']
    for _, row in probability.iterrows():
        lines.append(f'| {int(row.anchor_offset)} | {row.direction} | {row.block} | '
                     f'{row.delta_from_destination:+.9f} | [{row.delta_from_destination_ci_low:+.9f}, {row.delta_from_destination_ci_high:+.9f}] |')
    lines += ['', '확률은 모델 출력의 평균이며 TS-AUC가 아닙니다. Alarm 결과는 고정 창 안의 any crossing 및 occupancy이고 '
              '전체 horizon의 eventual false alarm과 다릅니다. 2,000회 whole-pair bootstrap 구간은 고정 모형·관측 코호트에 조건부이며 '
              '적응적 진단 선택이나 다중 비교를 보정하지 않습니다.', '',
              '전체 540개 창/지표 결과: `artifacts/next3/outlier_state_transfer/run_v1/WINDOW_COUNTERFACTUAL_INTERVALS.csv`.',
              '모든 시점 평균 궤적과 세 개 그림: `artifacts/next3/figures/outlier_state_transfer_v1/`.',
              '다음 D006은 같은 원본 O1 안에서 legacy 상태만 세분화합니다. NEXT3 과학 후보 탐색은 계속 닫혀 있습니다.', '']
    report = ROOT/'docs/NEXT3_OUTLIER_STATE_TRANSFER_FINDINGS.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'source_result_sha256': digest(source/'RESULTS.json'), 'source_table_sha256': digest(source/'WINDOW_COUNTERFACTUAL_INTERVALS.csv'),
        'aggregate_rows_reconstructed': 540, 'new_fits': 0, 'new_quality_scores': 0,
        'figure_environment_versions': versions, 'source_sha256': digest(Path(__file__)),
        'files_sha256': {name: digest(out/name) for name in files},
        'findings_sha256': digest(report), 'visual_review': 'PENDING'}, immutable=True)
    progress('D005 complete aggregate report', 'All 540 rows checked; all anchors, blocks and directions rendered without fitting or selecting')
    print('NEXT3 D005 REPORT AND FIGURES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
