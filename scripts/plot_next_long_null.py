"""Export horizon false-alarm figures only from the completed fixed study."""
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
    folder = ROOT/'artifacts/next/long_null_stress'
    result = json.loads((folder/'RESULTS.json').read_text(encoding='utf-8'))
    if result['status'] != 'PASS' or result['long_scalar_predictions'] != 327680000:
        raise RuntimeError('The complete fixed long-null study is required')
    for name, digest in result['files_sha256'].items():
        if sha256(folder/name) != digest:
            raise RuntimeError('Long-null summary changed')
    with (folder/'HORIZON_FALSE_ALARMS.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    out = ROOT/'artifacts/next/figures/long_null_v2'
    if out.exists():
        raise RuntimeError('Preserve the first completed long-null plot snapshot')
    out.mkdir(parents=True)
    roles = ['PRIMARY', 'V007_VARIANCE_FORECAST', 'Q005_VARIANCE_NO_MEMORY', 'Y001_LONG_VARIANCE_ARCH1']
    labels = ['Frozen primary', 'V007, ages 128', 'Q005, no memory', 'Y001, ages 512']
    colors = ['#68788e', '#098477', '#925ea1', '#bb7133']
    regimes = list(dict.fromkeys(row['regime'] for row in rows))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.fonttype': 'none'})
    files = []
    for alpha in (.01, .05):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
        upper = min(100., 5*np.ceil(100*max(float(row['wilson_95_high']) for row in rows
            if float(row['nominal_alpha_at_1024']) == alpha)/5)+5)
        for ax, regime in zip(axes.ravel(), regimes):
            for role, label, color in zip(roles, labels, colors):
                selected = sorted((row for row in rows if row['model'] == role and row['regime'] == regime and float(row['nominal_alpha_at_1024']) == alpha), key=lambda row: int(row['horizon']))
                horizon = np.array([int(row['horizon']) for row in selected])
                rates = 100*np.array([float(row['false_alarm_fraction']) for row in selected])
                low = 100*np.array([float(row['wilson_95_low']) for row in selected])
                high = 100*np.array([float(row['wilson_95_high']) for row in selected])
                ax.plot(horizon, rates, 'o-', color=color, linewidth=1.8, markersize=4, label=label)
                ax.fill_between(horizon, low, high, color=color, alpha=.09, linewidth=0)
            ax.axhline(100*alpha, color='#444', linestyle='--', linewidth=.8)
            ax.axvline(1024, color='#777', linestyle=':', linewidth=.8)
            ax.set_xscale('log', base=2)
            ax.set(title=regime, xlabel='Observed online points', ylabel='At least one false alarm (%)',
                xticks=[128, 512, 1024, 4096, 16384], xticklabels=['128', '512', '1,024', '4,096', '16,384'])
            ax.tick_params(axis='x', rotation=30)
            ax.set_ylim(0, upper)
            ax.grid(axis='y', alpha=.18)
        axes.ravel()[-1].axis('off')
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        axes.ravel()[-1].legend(handles, legend_labels, loc='center', frameon=False, fontsize=11)
        fig.suptitle(f'Long stationary streams after a fixed {100*alpha:g}% calibration at 1,024 points', fontsize=15)
        fig.tight_layout(rect=(0, .11, 1, .955))
        fig.text(.02, .025, 'One threshold per model from 5,000 separate calibration nulls, held fixed for all horizons. 1,000 new nulls per historical regime.\nShading: descriptive Wilson 95% intervals. These synthetic regimes are not the competition generator; no production threshold or promotion decision is changed.', fontsize=9, color='#4b5563')
        base = out/('null_horizon_alpha_'+str(int(100*alpha)))
        for extension in ('.png', '.svg'):
            path = base.with_suffix(extension)
            fig.savefig(path, dpi=190)
            files.append(path)
        plt.close(fig)
    write_json(out/'MANIFEST.json', {'status': 'PASS', 'created_utc': utc_now(), 'parent_result_sha256': sha256(folder/'RESULTS.json'),
        'script_sha256': sha256(Path(__file__)), 'files_sha256': {p.name: sha256(p) for p in files}})
    print('LONG NULL FIGURES COMPLETE', len(files), flush=True)


if __name__ == '__main__':
    main()
