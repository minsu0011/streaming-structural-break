"""Publication figures from complete synthetic scalar evidence only."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/next5'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = {}
    def receipt(path):
        result = read(path)
        assert result['status'] == 'PASS'
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return result
    def table(path, proof):
        assert sha(path) == proof['files_sha256'][path.name]
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    stress = BASE / 'synthetic/fitted_summary/complete_v1'
    proof = receipt(stress / 'RESULTS.json')
    assert proof['fitted_artifacts'] == 97
    repair = receipt(stress / 'COHORT_PROVENANCE_REPAIR.json')
    assert repair['original_numeric_summary_result_sha256'] == sha(stress / 'RESULTS.json')
    scores = table(stress / 'FIXED_ARTIFACT_MEAN_TS_AUC.csv', proof)
    alarms = table(stress / 'LONG_NULL_RESULTS.csv', proof)
    impulse_path = BASE / 'impulse_decomposition/complete_v1'
    impulse_proof = receipt(impulse_path / 'RESULTS.json')
    impulses = table(impulse_path / 'IMPULSE_EXCESS_DECOMPOSITION.csv', impulse_proof)
    out = BASE / 'analysis_figures/stress_v1'
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
        'font.size': 10, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False})
    figures = []
    scenarios = ['M1', 'M2', 'V1', 'V2', 'V3', 'V4', 'A1', 'A2', 'T1', 'T2', 'D1', 'D2', 'X1', 'CB_N3_070', 'CB_N3_050']
    selected = [r for r in scores if r['scope'] == 'full' and r['reference'] == 'BL80' and r['scenario'] in scenarios]
    candidates = sorted({r['candidate'] for r in selected})
    assert len(candidates) == 19 and len(selected) == 19 * 15
    matrix = np.empty((19, 15))
    uncertain = np.zeros((19, 15), dtype=bool)
    for i, cid in enumerate(candidates):
        for j, scenario in enumerate(scenarios):
            row = next(r for r in selected if (r['candidate'], r['scenario']) == (cid, scenario))
            assert int(row['fitted_artifacts']) == 5
            matrix[i, j] = 1000 * float(row['estimate_mean_delta'])
            uncertain[i, j] = float(row['ci_2_5']) <= 0 <= float(row['ci_97_5'])
    fig, ax = plt.subplots(figsize=(16, 10))
    bound = max(1., float(np.abs(matrix).max()))
    picture = ax.imshow(matrix, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
    for i in range(19):
        for j in range(15):
            ax.text(j, i, f'{matrix[i,j]:+.1f}' + ('*' if uncertain[i,j] else ''), ha='center', va='center',
                fontsize=8, color='white' if abs(matrix[i, j]) > .6 * bound else '#111111')
    ax.set_yticks(range(19), candidates)
    ax.set_xticks(range(15), scenarios, rotation=30, ha='right')
    ax.set_title('고정된 5개 학습 모델의 평균 합성 TS-AUC 차이', loc='left', fontweight='bold', pad=14)
    fig.colorbar(picture, ax=ax, label='BL80 대비 Δ × 1,000')
    fig.text(.02, .02, '각 셀은 5개 CV artifact의 점수 평균이며 앙상블 예측 점수가 아닙니다. * 기존 paired-seed 95% 구간이 0을 포함합니다.\n학습 fold는 겹치며 seed 구간은 후보 선택·학습·calibration 불확실성을 포함하지 않습니다. 무변화와 outlier의 관례상 TS-AUC=0.5는 제외했습니다.', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, 1])
    figures.append(('ALL_FOLD_COUNTEREXAMPLES', fig))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    horizons = [1024, 4096, 16384]
    for ax, scenario in zip(axes.flat, ['N0', 'N1', 'N2', 'N3']):
        for model, color, label in [('BL80', '#286ca8', 'matched BL80'), ('candidate', '#ce6b27', 'X2 scale × tail')]:
            values = np.empty((5, 3))
            for fold in range(5):
                for j, horizon in enumerate(horizons):
                    records = [r for r in alarms if r['candidate'] == 'X2_SCALE_TAIL' and r['scope'] == 'full'
                        and r['scenario'] == scenario and r['model'] == model and int(r['fold']) == fold
                        and int(r['horizon']) == horizon and float(r['alpha']) == .05]
                    assert len(records) == 1
                    values[fold, j] = float(records[0]['null_eventual_alarm_rate'])
            x = np.arange(3)
            ax.plot(x, values.mean(axis=0), '-o', color=color, label=label)
            ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=color, alpha=.12)
            for fold in range(5):
                ax.scatter(x + (fold - 2) * .025, values[fold], color=color, alpha=.6, s=12)
        ax.axhline(.05, ls='--', color='#888888', lw=1)
        ax.set_xticks(range(3), ['1,024', '4,096', '16,384'])
        ax.set_ylim(0, 1)
        ax.set_title(scenario)
        ax.set_xlabel('평가 horizon')
        ax.set_ylabel('한 번 이상 알람이 발생한 비율')
        ax.grid(axis='y', alpha=.18)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('1,024시점에서 보정한 임계값을 긴 무변화 구간에 적용', x=.04, ha='left', fontweight='bold')
    fig.text(.03, .02, '선: 5개 고정 artifact의 평균. 점: 각 fold. 음영: 최소–최대 범위이며 신뢰구간이 아닙니다. 각 코호트는 동일 256개 생성 seed입니다.\n점선 0.05는 1,024 calibration 설정이며 16,384까지의 anytime false-alarm 보장이 아닙니다.', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .95])
    figures.append(('X2_ALL_FOLD_LONG_NULL', fig))
    render_impulse(figures, impulses)
    with PdfPages(out / 'COMPLETE_FITTED_STRESS_FIGURES.pdf') as pdf:
        for name, figure in figures:
            figure.savefig(out / (name + '.png'), dpi=180, bbox_inches='tight')
            pdf.savefig(figure, bbox_inches='tight')
            plt.close(figure)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
        'figures': len(figures), 'source_sha256': sha(Path(__file__)), 'input_sha256': sources,
        'files_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        'matplotlib_version': matplotlib.__version__, 'raw_or_Shadow_data_loaded': False,
        'fitted_coverage': 97, 'new_fits': 0, 'new_predictions': 0, 'new_seed_draws': 0,
        'scope': 'Scalar evidence visualizations; fixed artifact means/ranges with explicit conditional uncertainty boundaries'}
    (out / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('NEXT5 COMPLETE FITTED STRESS FIGURES', len(figures), flush=True)


def render_impulse(figures, impulses):
    lags = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    candidates = ['P1_TWO_TIMESCALE_CONFIRMATION', 'R2_CONDITIONAL_SCALE_PIT', 'X2_SCALE_TAIL']
    metrics = [('total_candidate_delta_vs_BL80', '#343a45', '전체 후보 Δ'),
        ('amplitude_only_delta_vs_BL80', '#286ca8', 'BL80 비중 축소'),
        ('new_expert_excess_above_affine_control', '#ce6b27', '추가 expert 응답')]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for i, cid in enumerate(candidates):
        for j, scenario in enumerate(['O1', 'OB16']):
            ax = axes[i, j]
            for metric, color, label in metrics:
                means = []
                for lag in lags:
                    records = [r for r in impulses if r['candidate'] == cid and r['scope'] == 'full'
                        and r['scenario'] == scenario and r['metric_kind'] == 'mean_at_impulse_lag'
                        and int(r['metric_or_lag']) == lag]
                    assert len(records) == 5 and {int(r['fold']) for r in records} == set(range(5))
                    means.append(np.mean([float(r[metric]) for r in records]))
                ax.plot(lags, means, '-o', color=color, ms=3, label=label)
            ax.axhline(0, color='#888888', lw=1)
            ax.set_xscale('symlog', linthresh=1, base=2)
            ax.set_xticks(lags, [str(lag) for lag in lags], fontsize=8)
            ax.set_title(cid.split('_')[0] + ' / ' + scenario)
            ax.set_ylabel('paired score excess의 BL80 대비 차이')
            ax.grid(axis='y', alpha=.18)
    for ax in axes[-1]:
        ax.set_xlabel('첫 충격부터의 lag')
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('낮은 충격 잔류 점수의 선형 분해', x=.04, ha='left', fontweight='bold')
    fig.text(.03, .02, '각 선은 5개 고정 artifact의 기술적 평균입니다. 전체 Δ = BL80 비중 축소 효과 + 추가 expert 응답.\nOB16의 lag 0–15는 아직 충격 구간입니다. 진폭 감소는 탐지 개선·영구 회복을 뜻하지 않으며 비선형 회복시간을 이 식으로 분해하지 않습니다.', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .95])
    figures.append(('ALL_FOLD_IMPULSE_DECOMPOSITION', fig))


if __name__ == '__main__':
    main()
