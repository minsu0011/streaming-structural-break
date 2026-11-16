"""Standalone paired intervention intervals from completed fixed OOF only."""
from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    source = ROOT/'artifacts/next/fixed_ablation_contrasts/research_v1/PAIRED_INTERVENTIONS.csv'
    out = ROOT/'artifacts/next/figures/fixed_interventions_v2'
    if out.exists():
        raise RuntimeError('Preserve the original exported figure snapshot')
    out.mkdir(parents=True)
    with source.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    labels = {'Y001_LONG_VARIANCE_ARCH1': 'Long ages: 512 vs 128',
        'Y002_LONG_VARIANCE_ARCH2': 'ARCH2 vs ARCH1 (long)', 'Y003_LONG_VARIANCE_EWMA': 'EWMA vs ARCH1 (long)',
        'Q002_VARIANCE_FORECAST_ONLY': 'Forecast channel only (short)', 'Q004_VARIANCE_NO_BAYES': 'Remove Bayes (short)',
        'Q005_VARIANCE_NO_MEMORY': 'Remove GLR memory (short)', 'Q009_VARIANCE_ARCH2_PAIR': 'ARCH2 vs ARCH1 (short)',
        'Q010_VARIANCE_EWMA_PAIR': 'EWMA vs ARCH1 (short)', 'Z001_LOG_ENERGY_CURRENT': 'Current log energy vs forecast',
        'F101_VARIANCE_TREES50': '50 vs 100 trees (short)', 'B101_LONG_VARIANCE_NO_MEMORY': 'Remove GLR memory (long)',
        'B102_LONG_VARIANCE_NO_BAYES': 'Remove Bayes (long)', 'B104_LONG_VARIANCE_NO_GLR_MEMORY': 'Remove explicit GLR + memory (long)',
        'F201_LONG_VARIANCE_TREES50': '50 vs 100 trees (long)', 'J101_SERIAL_VARIANCE_128': 'AR1 sum-variance correction (short)',
        'J102_SERIAL_VARIANCE_512': 'AR1 sum-variance correction (long)'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(12, 8.8))
    for j, row in enumerate(rows):
        point, low, high = [1000*float(row[key]) for key in ('estimate_mean_delta', 'ci_2_5', 'ci_97_5')]
        color = '#087f75' if low > 0 else '#bd463b' if high < 0 else '#68788e'
        ax.plot([low, high], [j, j], color=color, lw=2.5, solid_capstyle='round')
        ax.scatter([point], [j], s=43, color=color, zorder=3)
        ax.text(4.8, j, f"{point:+.3f}", va='center', fontsize=9, color=color)
    ax.axvline(0, color='#343a46', linestyle='--', linewidth=1)
    ax.set(yticks=np.arange(len(rows)), yticklabels=[labels[r['candidate']] for r in rows],
        xlabel='Mean TS-AUC difference × 1,000 (candidate minus matched parent)',
        title='Which additions have a measurable DEV contribution?', xlim=(-7, 5.8))
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.15)
    ax.text(4.8, -.8, 'Estimate', fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, 1))
    fig.text(.02, .024, '2,000 paired whole-series bootstrap draws per original fold. Intervals condition on fitted OOF and selected contrasts; no selection/multiplicity correction.\nShort = maximum age 128; long = 512. J102 vs Y001: +0.0000002804, CI includes zero, P(gain) = 0.5015. No evidence of a serial-correction gain.', fontsize=9, color='#4b5563')
    files = [out/'paired_interventions.png', out/'paired_interventions.svg']
    fig.savefig(files[0], dpi=190)
    fig.savefig(files[1])
    plt.close(fig)
    write_json(out/'MANIFEST.json', {'status': 'PASS', 'created_utc': utc_now(), 'script_sha256': sha256(Path(__file__)),
        'source_csv_sha256': sha256(source), 'contrasts': len(rows), 'files_sha256': {p.name: sha256(p) for p in files},
        'purpose': 'Research diagnostic figure, not final roles or an independent selection-adjusted estimate.'})
    print('PAIRED INTERVENTION FIGURE COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
