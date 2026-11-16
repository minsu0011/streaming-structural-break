"""Standalone figures of the controlled mechanism and null-regime findings."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


if __name__ == '__main__':
    folder = ROOT/'artifacts/next/synthetic_variance_stress'
    report = json.loads((folder/'RESULTS.json').read_text(encoding='utf-8'))
    if report['status'] != 'PASS':
        raise RuntimeError('Controlled diagnostic is incomplete')
    for name, digest in report['files_sha256'].items():
        if sha256(folder/name) != digest:
            raise RuntimeError('Controlled diagnostic report changed')
    out = ROOT/'artifacts/next/figures/controlled_stress_v1'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'MANIFEST.json').exists():
        raise RuntimeError('Retain the existing immutable figure snapshot')
    with (folder/'SYNTHETIC_AUC.csv').open(newline='', encoding='utf-8') as handle:
        auc = list(csv.DictReader(handle))
    with (folder/'ALARM_SUMMARY.csv').open(newline='', encoding='utf-8') as handle:
        alarm = list(csv.DictReader(handle))
    regimes = ['IID_GAUSSIAN', 'AR06_GAUSSIAN', 'AR09_STUDENT5', 'AR03_ARCH065_GAUSSIAN', 'AR_MINUS04_ARCH04_STUDENT5']
    labels = ['IID Gaussian', 'AR 0.6 / Gaussian', 'AR 0.9 / Student t5', 'AR 0.3 / ARCH 0.65', 'AR -0.4 / ARCH 0.4 / t5']
    scenarios = ['MEAN_SHIFT', 'SCALE_SHIFT', 'DEPENDENCE_SHIFT', 'TAIL_SHIFT', 'VARIANCE_PERSISTENCE_SHIFT', 'MIXED_SHIFT']
    short = ['Mean', 'Scale', 'Dependence', 'Tail', 'Variance\npersistence', 'Mixed']
    lookup = {(row['model'], row['regime'], row['scenario']): float(row['synthetic_ts_auc']) for row in auc}
    roles = ['PRIMARY', 'HIGHEST_MEAN', 'V007_VARIANCE_FORECAST']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    matrices = [np.array([[lookup[roles[2], regime, scenario]-lookup[baseline, regime, scenario] for scenario in scenarios] for regime in regimes]) for baseline in roles[:2]]
    limit = max(abs(matrix).max() for matrix in matrices)
    for ax, matrix, title in zip(axes, matrices, ['V007 minus frozen primary', 'V007 minus matched 51-feature baseline']):
        picture = ax.imshow(matrix, cmap='RdBu', norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect='auto')
        ax.set(title=title, xticks=range(6), xticklabels=short, yticks=range(5), yticklabels=labels)
        ax.tick_params(axis='x', labelsize=9)
        for row in range(5):
            for col in range(6):
                value = matrix[row, col]
                ax.text(col, row, f'{value:+.3f}', ha='center', va='center', color='white' if abs(value) > .65*limit else '#111')
    fig.subplots_adjust(left=.2, right=.9, bottom=.20, top=.89, wspace=.10)
    color_ax = fig.add_axes([.92, .25, .015, .57])
    fig.colorbar(picture, cax=color_ax, label='Synthetic TS-AUC difference')
    fig.text(.015, .035, '1,000 changed futures and their paired null futures per cell. Fixed CV-fold-0 models; no synthetic refitting or model selection.\nThese chosen AR/ARCH/Student processes are diagnostics, not an inferred competition generator. Values are point estimates.', fontsize=9)
    fig.savefig(out/'controlled_mechanism_differences.png', dpi=190)
    fig.savefig(out/'controlled_mechanism_differences.svg')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = ['#697586', '#a07633', '#007f78']
    model_labels = ['Frozen primary', 'Matched baseline', 'V007 variance pair']
    for r, role in enumerate(roles):
        selected = {row['regime']: row for row in alarm if row['model'] == role and row['scenario'] == 'NO_CHANGE' and float(row['nominal_alpha']) == .05}
        values = np.array([float(selected[regime]['null_false_alarm_fraction']) for regime in regimes])*100
        low = np.array([float(selected[regime]['null_false_alarm_wilson_95_low']) for regime in regimes])*100
        high = np.array([float(selected[regime]['null_false_alarm_wilson_95_high']) for regime in regimes])*100
        ax.bar(np.arange(5)+(r-1)*.25, values, .23, color=colors[r], label=model_labels[r], yerr=np.stack([values-low, high-values]), capsize=3, error_kw={'linewidth': .8})
    ax.axhline(5, color='#333', lw=1.2, ls='--', label='5% pooled calibration target')
    ax.set(title='Stationary null alarms vary by historical regime', ylabel='Series with at least one false alarm (%)', xticks=range(5), xticklabels=['IID\nGaussian', 'AR 0.6\nGaussian', 'AR 0.9\nStudent t5', 'AR 0.3\nARCH 0.65', 'AR -0.4 / ARCH 0.4\nStudent t5'])
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, .14, 1, 1))
    fig.text(.015, .025, 'Each model uses one threshold from 5,000 separate 1,024-point calibration nulls, pooled equally across regimes.\nEvaluation: 1,000 new nulls per regime. Error bars: descriptive 95% Wilson intervals; no per-regime threshold tuning.', fontsize=9)
    fig.savefig(out/'controlled_null_alarm_regimes.png', dpi=190)
    fig.savefig(out/'controlled_null_alarm_regimes.svg')
    plt.close(fig)
    files = list(out.glob('*.png'))+list(out.glob('*.svg'))
    write_json(out/'MANIFEST.json', {'status': 'PASS', 'created_utc': utc_now(), 'parent_result_sha256': sha256(folder/'RESULTS.json'),
        'script_sha256': sha256(Path(__file__)), 'files_sha256': {path.name: sha256(path) for path in files},
        'purpose': 'Intermediate standalone scientific figures; no final handoff or session-completion claim.'})
    print('EXPORTED CONTROLLED FIGURES', len(files), flush=True)
