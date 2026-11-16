"""Plot measured quality/cost tradeoffs for the eight engineered candidates."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    out = ROOT/'artifacts/next/figures/actual_artifact_tradeoffs_v2'
    if out.exists():
        raise RuntimeError('Preserve the first actual-artifact tradeoff figure')
    sources, rows = {}, []
    def read(relative):
        path = ROOT/relative
        sources[path.relative_to(ROOT).as_posix()] = sha256(path)
        return json.loads(path.read_text(encoding='utf-8'))
    reference = read('artifacts/next/REFERENCE_RESTORATION.json')['references']['PRIMARY']
    labels = {'J102_SERIAL_VARIANCE_512': 'J102  Serial correction, ages 512',
        'Y001_LONG_VARIANCE_ARCH1': 'Y001  Variance, ages 512',
        'B104_LONG_VARIANCE_NO_GLR_MEMORY': 'B104  Ages 512, no GLR / memory',
        'Q005_VARIANCE_NO_MEMORY': 'Q005  Ages 128, no memory',
        'V007_VARIANCE_FORECAST': 'V007  Variance, ages 128',
        'Z001_LOG_ENERGY_CURRENT': 'Z001  Current log energy',
        'F201_LONG_VARIANCE_TREES50': 'F201  Ages 512, 50 trees',
        'F101_VARIANCE_TREES50': 'F101  Ages 128, 50 trees',
        'PRIMARY': 'Frozen PRIMARY reference'}
    for eid in labels:
        latency = read(f'artifacts/next/latency/all_dev_v1/{eid}.json')
        if latency['status'] != 'PASS' or latency['identity']['model_kind'] != 'ALL_DEV_REFIT_ARTIFACT':
            raise RuntimeError('Exact actual-artifact timing required')
        if eid == 'PRIMARY':
            scores = reference['fold_scores']
        else:
            original = read(f'artifacts/next/experiments/{eid}/RESULTS.json')
            engineering = read(f'artifacts/next/final_engineering/{eid}/RESULTS.json')
            if original['full']['status'] != 'COMPLETE' or engineering['status'] != 'PASS' or not all(engineering['checks'].values()):
                raise RuntimeError('Complete candidate quality and engineering evidence required')
            if engineering['model_sha256'] != latency['identity']['model_sha256']:
                raise RuntimeError('Timing and engineering model bytes differ')
            scores = original['full']['fold_scores']
        rows.append({'candidate': eid, 'label': labels[eid], 'mean': float(np.mean(scores)),
            'median': float(np.median(scores)), 'worst': float(min(scores)), 'best': float(max(scores)),
            'point_median_us': latency['point_distribution_us']['median'],
            'history_amortized_us': latency['real_history_initialization']['amortized_us_per_point'],
            'continuous_10m_seconds': latency['continuous_stream']['measured_prediction_seconds'],
            'fixed_state_array_bytes': latency['continuous_stream']['state_array_bytes_after']})
    rows.sort(key=lambda row: -row['mean'])
    out.mkdir(parents=True)
    with (out/'MEASURED_TRADEOFFS.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7), sharey=True,
        gridspec_kw={'width_ratios': [1.3, 1., 1.3]})
    for index, row in enumerate(rows):
        color = '#748094' if row['candidate'] == 'PRIMARY' else '#088479'
        axes[0].plot([row['worst'], row['best']], [index, index], color=color, alpha=.5, lw=2)
        axes[0].scatter(row['mean'], index, s=45, color=color, zorder=3)
        axes[1].scatter(row['worst'], index, s=45, color=color)
        axes[2].plot([row['point_median_us'], row['history_amortized_us']], [index, index], color='#bcc5cd', lw=1)
        axes[2].scatter(row['point_median_us'], index, color='#36689b', marker='o', s=38,
            label='Point median' if index == 0 else None)
        axes[2].scatter(row['history_amortized_us'], index, color='#af6e36', marker='s', s=38,
            label='H + observed O, amortized' if index == 0 else None)
        axes[2].annotate(f"{row['point_median_us']:.1f}", (row['point_median_us'], index), xytext=(-7, 0),
            textcoords='offset points', ha='right', va='center', fontsize=8, color='#36689b')
        axes[2].annotate(f"{row['history_amortized_us']:.2f}", (row['history_amortized_us'], index), xytext=(6, 0),
            textcoords='offset points', ha='left', va='center', fontsize=8, color='#af6e36')
    axes[0].set(title='Original CV mean and five-fold range', xlabel='TS-AUC',
        yticks=np.arange(len(rows)), yticklabels=[row['label'] for row in rows])
    axes[0].invert_yaxis()
    axes[0].axvline(reference['mean'], color='#748094', linestyle='--', lw=.8)
    axes[1].set(title='Worst original fold', xlabel='TS-AUC')
    axes[1].axvline(reference['worst'], color='#748094', linestyle='--', lw=.8)
    axes[2].set(title='Actual all-DEV artifact latency', xlabel='Microseconds per point', xlim=(0, 42))
    handles, legend_labels = axes[2].get_legend_handles_labels()
    for ax in axes:
        ax.grid(axis='x', alpha=.15)
        ax.tick_params(axis='y', length=0)
    fig.suptitle('Quality, worst-fold behavior and measured inference cost', fontsize=15)
    fig.tight_layout(rect=(0, .25, 1, .94), w_pad=2.5)
    fig.legend(handles, legend_labels, loc='upper right', bbox_to_anchor=(.985, .205), frameon=False, fontsize=9)
    fig.text(.02, .02, 'Eight candidates passed all nine engineering checks. This research snapshot does not freeze final roles.\n'
        'Left whiskers show the five observed CV folds, not a confidence interval. J102 and Y001 are effectively tied in quality.\n'
        'Latency uses exact all-DEV weights and the same fixed 128-series scalar protocol. It excludes generator, file/socket and cloud overhead.\n'
        'One retained measurement per artifact; ordinary OS load/power variation remains. H amortization assumes the measured history/online-length mixture.',
        fontsize=9, color='#4b5563')
    images = [out/'actual_tradeoffs.png', out/'actual_tradeoffs.svg']
    for path in images:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    write_json(out/'MANIFEST.json', {'status': 'PASS', 'created_utc': utc_now(), 'script_sha256': sha256(Path(__file__)),
        'source_files_sha256': sources, 'models': len(rows),
        'files_sha256': {path.name: sha256(path) for path in [*images, out/'MEASURED_TRADEOFFS.csv']},
        'new_model_fits': 0, 'new_quality_evaluations': 0, 'final_role_selection': False})
    print('ACTUAL ARTIFACT TRADEOFF FIGURE PASS', len(rows), flush=True)


if __name__ == '__main__':
    main()
