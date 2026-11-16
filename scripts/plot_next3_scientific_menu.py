"""Standalone figures for the completed finite menu, using only aggregate evidence."""
from pathlib import Path
import sys
import json
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress

LABELS = {
    'I2_HISTORICAL_RANK_PRODUCTS': 'I2 H-rank products', 'O1_LINEAR_RESIDUAL': 'O1 nested residual',
    'P1_LAGS_1_2': 'P1 lags 1 / 2', 'P2_LAGS_4_8': 'P2 lags 4 / 8',
    'P3_LOCAL_ONLY': 'P3 local only', 'P4_BAYES_ONLY': 'P4 Bayes only',
    'R1_SCORE_GAP': 'R1 score gap', 'R1_EVIDENCE_RATIO': 'R1 evidence ratio',
    'R2_LOGISTIC_GATE': 'R2 nested logistic', 'R3_DEPTH2_GATE': 'R3 nested depth 2',
    'S1_LOWER_SEMISCALE': 'S1 asymmetric H', 'S2_LOWER_QUANTILE': 'S2 lower quartile',
    'S3_NEGATIVE_BAYES': 'S3 negative Bayes',
}


def main():
    lock = ROOT/'requirements-next2-figures-lock.txt'; versions = {}
    for line in lock.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'): continue
        name, version = line.split('=='); assert importlib.metadata.version(name) == version
        versions[name] = version
    out = BASE/'figures/scientific_menu_v1'; out.mkdir(parents=True, exist_ok=False)
    sources = {}; figures = []
    def read(path):
        path = ROOT/path; sources[path.relative_to(ROOT).as_posix()] = digest(path)
        return pd.read_csv(path, float_precision='round_trip')
    def save(fig, name, caption):
        fig.savefig(out/(name+'.png'), dpi=200, bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig); figures.append({'name': name, 'caption': caption})
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.titleweight': 'bold', 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42})
    joint_path = 'artifacts/next3/selection_uncertainty/CONDITIONAL_POOL_ENVELOPE.csv'
    joint_receipt = read_json(BASE/'selection_uncertainty/RESULTS.json')
    assert joint_receipt['status'] == 'PASS'
    joint = read(joint_path).sort_values('estimate_delta', ascending=False)
    assert len(joint) == 12 and not joint.selection_adjusted.any()
    fig, ax = plt.subplots(figsize=(10, 7.8))
    y = np.arange(len(joint)); p = joint.estimate_delta.to_numpy()
    ax.errorbar(p, y+.12, xerr=np.stack([p-joint.joint_conditional_envelope_lower,
        joint.joint_conditional_envelope_upper-p]), fmt='o', color='#8a949e', capsize=3,
        markersize=3, label='Joint conditional pool envelope')
    ax.errorbar(p, y-.12, xerr=np.stack([p-joint.conditional_percentile_ci_2_5,
        joint.conditional_percentile_ci_97_5-p]), fmt='o', color='#176b91', capsize=3,
        markersize=4, label='Individual conditional 95% interval')
    ax.axvline(0, color='#333333', lw=1)
    ax.axvline(.0015, color='#b95620', ls='--', lw=1.3, label='Required mean effect: +0.0015')
    ax.set_yticks(y, [LABELS[k] for k in joint.candidate]); ax.invert_yaxis()
    ax.set_xlabel('Mean original 5-fold DEV TS-AUC difference from frozen BL80')
    ax.set_title('No completed full-CV candidate reaches the effect-size gate', pad=18)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.), fontsize=8, frameon=False); ax.grid(axis='x', alpha=.15)
    ax.set_xlim(-.0026, .00185)
    fig.text(.12, -.025, '2,000 paired whole-series draws, conditional on fixed fitted predictions.\nNeither interval accounts for adaptive hypothesis creation, screening or earlier DEV reuse.', fontsize=8)
    save(fig, '01_full_menu_conditional_uncertainty', 'Twelve completed full candidates, including rejected full-admission recipes. Joint envelope is descriptive and not a selective-coverage guarantee.')

    scores = read('MODEL_RESULTS_NEXT3.csv'); full = scores[scores.full_mean.notna()].copy(); assert len(full) == 12
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(full.screen_delta, full.delta_vs_frozen_performance, s=48, color='#176b91', zorder=3)
    for kind, offset in [('S1_LOWER_SEMISCALE', (-125, -30)), ('S3_NEGATIVE_BAYES', (-30, 28)),
                         ('I2_HISTORICAL_RANK_PRODUCTS', (15, 13)), ('R2_LOGISTIC_GATE', (10, -26)),
                         ('P2_LAGS_4_8', (-90, -24))]:
        r = full[full.experiment_id == kind].iloc[0]
        ax.annotate(LABELS[kind], (r.screen_delta, r.delta_vs_frozen_performance), xytext=offset,
                    textcoords='offset points', fontsize=9, arrowprops={'arrowstyle': '-', 'color': '#777777'})
    ax.axhline(0, color='#777777', lw=.8); ax.axvline(0, color='#777777', lw=.8)
    ax.axhline(.0015, color='#b95620', ls='--', label='Required full mean effect')
    ax.set_xlabel('Screen mean delta (fixed 3,000-series subset; folds 0 and 4)')
    ax.set_ylabel('Full mean delta (original 8,000-series five folds)')
    ax.set_title('Promising screen changes did not establish full-DEV gains', pad=16)
    ax.set_ylim(-.0016, .00185); ax.set_xlim(-.00135, .00315); ax.grid(alpha=.15)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.text(.12, -.045, 'Each stage compares with its own matched frozen reference. Stages differ in training and validation populations.\nS2 failed screening (-0.003843) and has no full result; no full evaluation was added for this plot.', fontsize=8)
    save(fig, '02_screen_to_full_effects', 'Stage-specific paired reference deltas. This scatter is not a new selection rule or a matched estimate of generalization bias.')

    paired = read('PAIRED_BOOTSTRAP_NEXT3.csv')
    candidates = [k for k in joint.candidate if k in set(paired.candidate)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), sharey=True)
    for ax, scope, title in zip(axes, ['mechanism_SCALE_DOMINANT', 'mechanism_DEPENDENCE_DOMINANT'],
                               ['SCALE_DOMINANT', 'DEPENDENCE_DOMINANT']):
        rows = paired[paired.scope == scope].set_index('candidate').loc[candidates]
        values = rows.estimate_mean_delta.to_numpy()
        ax.errorbar(values, np.arange(len(candidates)), xerr=np.stack([values-rows.ci_2_5, rows.ci_97_5-values]),
                    fmt='o', color='#176b91', capsize=3, markersize=4)
        ax.axvline(0, color='#333333', lw=1); ax.grid(axis='x', alpha=.15)
        ax.set_title(title); ax.set_xlabel('Conditional mechanism TS-AUC difference')
    axes[0].set_yticks(np.arange(len(candidates)), [LABELS[k] for k in candidates]); axes[0].invert_yaxis()
    fig.suptitle('Partial scale improvements do not resolve the joint tradeoff', y=1.02)
    fig.text(.12, -.04, 'Original-OOF diagnostic slices with their locked controls; only admitted forensic candidates are shown.\nThese are conditional intervals, not independent confirmation or new promotion decisions.', fontsize=8)
    save(fig, '03_scale_dependence_tradeoff', 'Mechanism-level effects are not additive overall contributions; pair masses differ. Show all eligible completed forensic candidates without changing their rejection decisions.')
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'figures': figures,
        'script_sha256': digest(Path(__file__)), 'source_sha256': sources, 'package_versions': versions,
        'figure_requirements_sha256': digest(lock), 'new_fits': 0, 'new_quality_scores': 0, 'seal_rows': 0,
        'files_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}, immutable=True)
    progress('Finite scientific evidence figures', 'Three standalone PNG/PDF figures generated from aggregate results; visual review remains')
    print('NEXT3 SCIENTIFIC FIGURES', out, flush=True)


if __name__ == '__main__': main()
