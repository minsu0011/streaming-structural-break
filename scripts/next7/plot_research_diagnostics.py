"""Computed research figures with explicit cohort, uncertainty and support labels."""
from pathlib import Path
import argparse
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
from src.next7.common import BASE, read_json, save, digest, utc_now, activity

COLORS = ('#287b8a', '#ba6255', '#d5a043', '#6f83a7')


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def styling():
    font = Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'axes.titlesize': 12,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
        'savefig.facecolor': '#f7f9fa', 'figure.facecolor': '#f7f9fa'})


def finish(fig, directory, name, caption):
    fig.text(.025, .025, caption, fontsize=9, color='#506674', va='bottom')
    fig.subplots_adjust(left=.065, right=.975, top=.9,
        bottom=.25 if name == 'TIME_AGE_RANKING_FRACTURES' else .15, wspace=.28, hspace=.38)
    for extension in ('png', 'svg'):
        fig.savefig(directory / f'{name}.{extension}', dpi=160)
    plt.close(fig)


def conditional(case, length, directory):
    cohort = BASE / f'conditional_statistics/{case}/H_{length}'
    part = BASE / f'support_partition/{case}/H_{length}'
    assert read_json(cohort / 'DONE.json')['status'] == read_json(part / 'DONE.json')['status'] == 'COMPLETE'
    results = rows(cohort / 'SUMMARY.csv')
    risk = [r for r in rows(cohort / 'PER_H_CONDITIONAL_RISK.csv') if r['horizon'] == '16384']
    risk.sort(key=lambda r: (float(r['conditional_G2_latch_rate']), int(r['reference_index'])))
    support = rows(BASE / f'bootstrap_support/{case}/H_{length}/PER_H_SUPPORT.csv')
    partitions = rows(part / 'SUMMARY.csv')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
    fig.suptitle(f'정상 시계열에서 남는 변화 기억  ·  {case} / 과거 길이 {length:,}',
        x=.035, ha='left', fontsize=20, color='#183642', weight='bold')
    ax = axes[0, 0]
    x = np.arange(1, 513)
    y = np.array([float(r['conditional_G2_latch_rate']) for r in risk])
    lo = np.array([float(r['conditional_MC_CI_low']) for r in risk])
    hi = np.array([float(r['conditional_MC_CI_high']) for r in risk])
    ax.fill_between(x, lo, hi, color=COLORS[0], alpha=.17, lw=0)
    ax.plot(x, y, color=COLORS[0], lw=1.8, label='H별 조건부 기억 발생률')
    ax.axhline(.05, ls=':', color='#9a6c38', lw=1, label='5% 비교선 · 보장 수준 아님')
    ax.set(title='같은 정상 분포에서도 H에 따라 위험이 다름',
        xlabel='진단된 위험 순서로 정렬한 과거 구간 512개', ylabel='16,384 시점까지 거짓 기억 발생률', ylim=(0, 1.03), xlim=(1, 512))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax = axes[0, 1]
    horizons = (1024, 4096, 16384)
    for k, (metric, label) in enumerate([('CORE_EVER_LATCH', '기존 hard-4'),
            ('G1_RAW_EVER_LATCH', 'G1 원래 기억'), ('G2_QUALIFIED_EVER_LATCH', 'G2 H 기준 통과 기억')]):
        selected = [next(r for r in results if r['metric'] == metric and int(r['horizon']) == h) for h in horizons]
        middle = np.array([float(r['mean_crossed_rate']) for r in selected])
        lower = middle - np.array([float(r['CI_low_approx_pointwise']) for r in selected])
        upper = np.array([float(r['CI_high_approx_pointwise']) for r in selected]) - middle
        ax.errorbar(np.arange(3) + (k - 1) * .19, middle, yerr=[lower, upper], marker='o',
            ms=5, lw=1.3, capsize=3, color=(COLORS[3], COLORS[1], COLORS[0])[k], label=label)
    ax.set(title='관측이 길어질 때 누적되는 거짓 기억', xlabel='현재까지 관측한 길이',
        ylabel='512×512 교차 경로의 평균 발생률', xticks=range(3), xticklabels=['1,024', '4,096', '16,384'], ylim=(0, 1.03))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 0]
    envelope = np.array([float(r['empirical_support_envelope_with_floor1']) for r in support])
    cut = np.array([float(r['cut_at16384']) for r in support])
    maximum = max(envelope.max(), cut.max()) * 1.06
    ax.plot([1, maximum], [1, maximum], color='#a5b2bb', ls='--', lw=1, label='기준 = 재표본 추출 최대 범위')
    ax.scatter(envelope, cut, s=14, color=COLORS[0], alpha=.45, edgecolors='none')
    ax.set(title=f'최종 기준이 경험적 최대 범위에 도달한 H: {np.sum(envelope == cut)}/512',
        xlabel='가능한 모든 재표본 확인 비율의 최대 U', ylabel='16,384 시점의 원래 G2 기준 T',
        xlim=(.95, maximum), ylim=(.95, maximum))
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax = axes[1, 1]
    names = ('BEYOND_EMPIRICAL_BOOTSTRAP_SUPPORT', 'WITHIN_SUPPORT_ABOVE_TERMINAL_CUT', 'EARLIER_CROSSING_RETAINED_ONLY')
    labels = ('경험적 최대 범위 밖의 초과', '범위 안에서 최종 기준 초과', '초기 초과 기억만 남음')
    bottom = np.zeros(3)
    for name, label, color in zip(names, labels, (COLORS[1], COLORS[2], COLORS[0])):
        values = np.array([float(next(r for r in partitions if r['component'] == name and int(r['horizon']) == h)['crossed_rate']) for h in horizons])
        ax.bar(range(3), values, bottom=bottom, width=.58, color=color, label=label)
        bottom += values
    for x, value in enumerate(bottom):
        ax.text(x, value + .015, f'{value:.1%}', ha='center', fontsize=10)
    ax.set(title='G2 거짓 기억의 정확한 세 부분 분해', xlabel='현재까지 관측한 길이', ylabel='전체 교차 경로 중 비율',
        xticks=range(3), xticklabels=['1,024', '4,096', '16,384'], ylim=(0, max(.5, bottom.max() * 1.33)))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    for ax in axes.flat:
        ax.set_facecolor('white')
        ax.grid(axis='y', color='#dce4e8', alpha=.5, lw=.7)
        ax.set_axisbelow(True)
    finish(fig, directory, f'CONDITIONAL_FALSE_MEMORY_{case}_H{length}',
        '512개의 H × 재사용한 512개 혁신 경로. 셀 262,144개를 독립 표본으로 세지 않음.\n'
        '음영: H를 고정한 혁신 재표본의 근사 구간. 오차 막대: 동일 시드 단위의 2,000회 교차 가중 근사 95% 구간.\n'
        '상태 기억을 분석한 그림이며 최종 분류기의 경보율과 다름. 조건부 5%·무한 시간 보장 없음. 기준값 변경 없음.')
    return [cohort / 'DONE.json', part / 'DONE.json']


def fractures(directory):
    paths = [BASE / 'FRACTURE_RESULTS_NEXT7_G1_G2.csv', BASE / 'FRACTURE_RESULTS_NEXT7_G3.csv']
    data = [row for path in paths for row in rows(path) if row['reference'] == 'M0']
    methods = ('I4_EVENT_SCALE_TREE', 'G2_FULL', 'G3_PAIRWISE_FULL')
    panels = [(['all_time'], ['전체']),
        (['time_0_9', 'time_10_24', 'time_25_49', 'time_50_99', 'time_100_199', 'time_200_399', 'time_400_plus'],
         ['0–9', '10–24', '25–49', '50–99', '100–199', '200–399', '400+']),
        (['age_0_4', 'age_5_9', 'age_10_19', 'age_20_49', 'age_50_plus'], ['0–4', '5–9', '10–19', '20–49', '50+'])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7.8), gridspec_kw={'width_ratios': [1, 2.2, 1.7]})
    fig.suptitle('시간대와 변화 직후 구간을 분리한 순위 성능', x=.035, ha='left', fontsize=20, color='#183642', weight='bold')
    titles = ('전체 TS-AUC', '같은 관측 시점끼리 비교', '양성의 변화 후 경과 시간별')
    for ax, (scopes, labels), title in zip(axes, panels, titles):
        for k, (method, color) in enumerate(zip(methods, (COLORS[3], COLORS[0], COLORS[1]))):
            selected = [next(r for r in data if r['candidate'] == method and r['scope'] == scope) for scope in scopes]
            middle = np.array([float(r['estimate_mean_matched_delta']) for r in selected])
            lower = middle - np.array([float(r['mean_delta_ci_2_5']) for r in selected])
            upper = np.array([float(r['mean_delta_ci_97_5']) for r in selected]) - middle
            ax.errorbar(middle, np.arange(len(scopes)) + (k - 1) * .19, xerr=[lower, upper],
                fmt='o', ms=4, color=color, capsize=3, lw=1.1, label=method)
        ax.axvline(0, color='#526672', lw=1)
        ax.set(title=title, xlabel='BL80 대비 짝지은 평균 ΔTS-AUC', yticks=range(len(labels)), yticklabels=labels)
        ax.invert_yaxis()
        ax.set_facecolor('white')
        ax.grid(axis='x', color='#dce4e8', lw=.7)
    axes[0].set(yticks=[-.19, 0, .19], yticklabels=['I4', 'G2', 'G3'], ylim=(.4, -.4))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, ncol=3, loc='lower center', bbox_to_anchor=(.5, .135))
    finish(fig, directory, 'TIME_AGE_RANKING_FRACTURES',
        '원래 RP 6,400개·그룹 5폴드. 전체 시계열 단위의 동일한 2,000회 짝지은 재표본.\n'
        '양성 경과 시간 분해에서도 같은 시점의 음성 비교 대상을 모두 유지함. 구간별 95% 구간이며 동시 보장·선택 보정 아님.\n'
        '특정 구간의 이득을 전체 성능 개선으로 해석하지 않음. 사후 분석의 정답 구간은 모델 입력으로 사용하지 않음.')
    return paths


def main(snapshot):
    assert snapshot and all(c.isalnum() or c in '_-' for c in snapshot)
    directory = BASE / 'visuals/diagnostics' / snapshot
    assert not directory.exists(), 'Preserve every prior figure snapshot'
    directory.mkdir(parents=True)
    styling()
    sources = []
    with activity('Computed conditional-memory and ranking diagnostic figures', 'SCIENTIFIC_VISUALIZATION', 'F4_NULL_GROWTH_CORRECTION'):
        for path in sorted((BASE / 'support_partition').glob('*/*/DONE.json')):
            case, length = path.parent.parent.name, int(path.parent.name.split('_')[1])
            sources += conditional(case, length, directory)
        sources += fractures(directory)
        save(str((directory / 'MANIFEST.json').relative_to(BASE)), {'status': 'COMPLETE', 'created_utc': utc_now(),
            'snapshot': snapshot, 'source_sha256': digest(__file__), 'generated_from_measured_data': True,
            'input_sha256': {p.relative_to(BASE).as_posix(): digest(p) for p in sources},
            'files_sha256': {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
            'uncompleted_cohorts_not_drawn': True, 'new_candidate_or_cutoff': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    main(args.snapshot)
