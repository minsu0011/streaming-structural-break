"""Export numerical research figures for the H-reference transport forensic."""
from pathlib import Path
import argparse
import csv
import re
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from src.next7.common import BASE, read_json, save, digest, utc_now, activity, note

plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
    'figure.facecolor': '#f6f9fb', 'axes.facecolor': 'white', 'savefig.facecolor': '#f6f9fb',
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
CHANNEL_NAMES = ('rank 평균', 'rank 에너지', '꼬리 빈도', 'rank 비대칭', 'lag1 rank',
                 'lag4 rank', 'lag1 에너지', 'lag4 에너지', 'signed scale')


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main(label):
    if not re.fullmatch('[a-z0-9_]+', label):
        raise ValueError('Simple unique figure snapshot name required')
    relative = 'visuals/normalizer_transport/' + label
    destination = BASE / relative
    assert not destination.exists()
    destination.mkdir(parents=True)
    names = read_json(BASE / 'NORMALIZER_TRANSPORT_FORENSIC_LOCK.json')['covariates']
    complete = sorted((BASE / 'normalizer_transport/correlations').glob('*/*/DONE.json'))
    assert complete
    inputs, outputs = {}, []
    geometry_path = BASE / 'RANK_REFERENCE_GEOMETRY_AUDIT_DONE.json'
    assert read_json(geometry_path)['rank_moment_formula_parity']
    inputs[geometry_path.relative_to(BASE).as_posix()] = digest(geometry_path)
    def export(figure, name):
        for suffix in ('png', 'svg'):
            path = destination / (name + '.' + suffix)
            figure.savefig(path, dpi=160)
            outputs.append(path)
        plt.close(figure)
    with activity('Draw historical reference transport forensic ' + label, 'RESEARCH_VISUALIZATION', 'F4_NULL_GROWTH_CORRECTION'):
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.set_xlim(-.08, 1.50)
        ax.set_ylim(-.35, 2.85)
        ax.axis('off')
        fig.suptitle('같은 H에서도 보정용 기준과 실제 관측용 기준은 다릅니다', x=.045, ha='left', fontsize=20, weight='bold', color='#203e49')
        h, prefix, prefix_fit = 2048, int(.6 * 2048), int(.6 * int(.6 * 2048))
        a, b = prefix_fit / h, prefix / h
        colors = ('#267b87', '#89b9bc', '#e1b76f')
        for y, widths, labels in (
            (1.9, (a, b - a, 1 - b), ('기준 분포\nH 0-735', '가상 기준의 보정\nH 736-1227\n완성 블록30개', 'H 안의 가상 online\nH 1228-2047\n완성 블록51개')),
            (.65, (b, 1 - b), ('실제 관측용 기준 분포\nH 0-1227', '실제 관측용 기준의 보정\nH 1228-2047\n완성 블록51개'))):
            left = 0
            for j, (width, text) in enumerate(zip(widths, labels)):
                color = colors[j] if len(widths) == 3 else colors[j]
                ax.add_patch(Rectangle((left, y), width, .55, facecolor=color, edgecolor='white', linewidth=2))
                ax.text(left + width / 2, y + .275, text, ha='center', va='center', color='white' if j == 0 else '#233e47', fontsize=10)
                left += width
        ax.text(0, 2.62, 'G2의 가상 경로: 앞60%로 상태 초기화 → 뒤40%의 증거 재표본', fontsize=12, weight='bold')
        ax.text(0, 1.37, '실제 online: 전체 H로 상태 초기화 → 이후 새 관측을 처리', fontsize=12, weight='bold')
        ax.annotate('재표본 최대값에서\nH별 허용 기준 계산', xy=(1, 2.17), xytext=(1.13, 2.17),
            arrowprops={'arrowstyle': '<-', 'color': '#56737c'}, va='center', color='#735a32')
        ax.annotate('실제 확인 증거와\n위의 H 기준을 비교', xy=(1, .925), xytext=(1.13, .925),
            arrowprops={'arrowstyle': '<-', 'color': '#56737c'}, va='center', color='#235a62')
        ax.text(0, .22, '예: H=2048. 실제 구현의 정수 내림과 16개 단위 블록 수를 표시했습니다.', color='#5f7680')
        ax.text(0, -.06, '기준의 차이를 추적하는 연구 그림입니다. 미래 관측은 보정에 사용하지 않으며, 이 그림은 개선 효과의 증명이 아닙니다.', color='#5f7680')
        fig.subplots_adjust(left=.045, right=.98, top=.87, bottom=.08)
        export(fig, 'H_REFERENCE_CONSTRUCTION')
        cohort_rows = []
        for path in complete:
            receipt = read_json(path)
            assert receipt['status'] == 'COMPLETE'
            table = path.parent / 'CORRELATIONS.csv'
            assert digest(table) == receipt['files_sha256']['CORRELATIONS.csv']
            inputs[table.relative_to(BASE).as_posix()] = digest(table)
            selected = {row['covariate']: row for row in rows(table) if row['horizon'] == '16384'}
            assert set(selected) == set(names)
            cohort_rows.append((receipt['case'], receipt['H_length'], selected))
        matrix = np.array([[float(selected[name]['Pearson_conditional_risk']) if selected[name]['Pearson_conditional_risk'] else np.nan
                           for _, _, selected in cohort_rows] for name in names])
        labels = [name + ' | 중심 차이' for name in CHANNEL_NAMES] + [name + ' | 폭 비의 로그' for name in CHANNEL_NAMES] + [
            'H 위 rank 기준 차이 | 최대', 'H 위 rank 기준 차이 | RMS', 'rank 2차 모멘트 | 동률·수치 구조', '가상 기준의 core 하한 채널 수', '실제 기준의 core 하한 채널 수']
        fig, ax = plt.subplots(figsize=(max(8, 4 + len(cohort_rows) * .95), 11))
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#dfe5e8')
        plotted = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_xticks(np.arange(len(cohort_rows)), [f'{case}\nH {length}' for case, length, _ in cohort_rows])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, f'{value:+.2f}' if np.isfinite(value) else '정의 안 됨', ha='center', va='center',
                        color='white' if np.isfinite(value) and abs(value) > .65 else '#263b45', fontsize=9)
        fig.colorbar(plotted, ax=ax, shrink=.6, label='H별 오기억률과의 Pearson 관련')
        ax.set_title('정규화 기준의 차이와 장기 오기억률\n완료된 코호트의 사후 진단 · 전체23개 지표', fontsize=15, pad=20)
        fig.text(.035, .02, '시점16384 · H 512개 × 기존 혁신 경로512개. 교차 셀을 독립 표본으로 세지 않습니다.\n'
            '인과효과나 새 입력 특징이 아닙니다. 구간 추정과 정의 불가 표본 수는 원본 CSV에 보존했습니다.\n'
            '고정 길이의 rank 2차 모멘트는 동률과 반올림으로만 미세하게 달라집니다. 독립적인 분포 위험 지표로 해석하지 않습니다.', color='#637c86', fontsize=9)
        fig.subplots_adjust(left=.30, right=.96, top=.89, bottom=.095)
        export(fig, 'NORMALIZER_RISK_ASSOCIATIONS')
        selected_covariates = ('log_prefix_spread_over_full__signed_log_MAD',
                              'center_shift_full_units__signed_log_MAD', 'max_rank_reference_gap_on_H')
        selected_labels = ('signed scale: 가상 기준 폭 / 실제 기준 폭의 로그',
                           'signed scale: 실제 기준 단위로 본 중심 차이', 'H 위 두 rank 기준의 최대 차이')
        for background in sorted({case for case, _, _ in cohort_rows}):
            subset = [(case, length) for case, length, _ in cohort_rows if case == background]
            fig, axes = plt.subplots(3, len(subset), figsize=(5 * len(subset), 10), squeeze=False)
            for column, (case, length) in enumerate(subset):
                parameter_path = BASE / f'normalizer_transport/H_parameters/{case}/H_{length}/COVARIATES.npz'
                risk_path = BASE / f'conditional_statistics/{case}/H_{length}/GRID_FOR_FIGURES.npz'
                inputs[parameter_path.relative_to(BASE).as_posix()] = digest(parameter_path)
                inputs[risk_path.relative_to(BASE).as_posix()] = digest(risk_path)
                with np.load(parameter_path, allow_pickle=False) as data:
                    values = data['values'].copy()
                with np.load(risk_path, allow_pickle=False) as data:
                    risk = data['qualified'][:, :, 2].mean(axis=1)
                for row, (name, label_text) in enumerate(zip(selected_covariates, selected_labels)):
                    ax = axes[row, column]
                    ax.scatter(values[:, names.index(name)], risk, s=12, alpha=.40, color='#277b89', edgecolors='none')
                    ax.set_ylim(-.025, 1.025)
                    ax.set_xlabel(label_text, fontsize=9)
                    ax.set_ylabel('H별 조건부 오기억률')
                    ax.grid(alpha=.13)
                    if row == 0:
                        ax.set_title(f'{case} · H {length}')
            fig.suptitle('개별 H에서 보이는 기준 차이와 오기억', fontsize=17, x=.04, ha='left', weight='bold')
            fig.text(.04, .02, '점 하나는 원래 H 하나입니다. 위험은 같은512개 혁신 경로를 교차해 측정했습니다.\n'
                '사후 관련 분석이며 회귀 모형, 새 탐지 기준, 인과효과를 추가하지 않았습니다.', fontsize=9, color='#637c86')
            fig.subplots_adjust(left=.09, right=.97, top=.90, bottom=.10, hspace=.40, wspace=.27)
            export(fig, 'PER_H_REFERENCE_RISK_' + background)
        save(relative + '/MANIFEST.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'source_sha256': digest(__file__), 'protocol_sha256': digest(BASE / 'NORMALIZER_TRANSPORT_FORENSIC_LOCK.json'),
            'completed_cohorts': [{'case': case, 'H_length': length} for case, length, _ in cohort_rows],
            'input_sha256': inputs, 'files_sha256': {path.name: digest(path) for path in outputs},
            'only_complete_cohort_results_used': True, 'new_fits_or_inference_features': 0})
    note('H reference forensic figures created', label + ': construction, all diagnostics and fixed scale/rank scatter plots', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    main(args.snapshot)
