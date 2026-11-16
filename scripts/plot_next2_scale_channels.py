"""Export interpretable calibration/channel figures with the two cohorts explicit."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    out = ROOT/'artifacts/next2/figures'
    a = ROOT/'artifacts/next2/h_calibration_asymmetry/SYNTHETIC_H_PARAMETERS.csv'
    b = ROOT/'artifacts/next2/channel_response/PAIRED_CHANNEL_RESPONSE.csv'
    h = pd.read_csv(a); response = pd.read_csv(b); backgrounds = ['N0', 'N1', 'N2', 'N3']
    labels = ['Gaussian', 'AR', 'ARCH', 'Tail + ARCH']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].boxplot([h[h.background == bg].energy_H_sd for bg in backgrounds], tick_labels=labels,
        showfliers=False, whis=(5, 95))
    axes[0].set_ylabel('H raw-energy calibration SD')
    axes[0].set_title('Fixed H calibration: 1,000 seeds per background')
    for ax, factor in zip(axes[1:], (.7, .5)):
        d = response[(response.post_window == 512) & (response.variance_factor == factor)]
        for j, (channel, color, name) in enumerate([('raw_energy', '#267cad', 'Raw energy'),
                ('log_variance_forecast', '#c17445', 'Log forecast')]):
            r = d[d.channel == channel].set_index('background').loc[backgrounds]
            means = r.mean_paired_clipped_current_delta.to_numpy(); x = np.arange(4)+(j-.5)*.3
            ax.errorbar(x, means, yerr=np.vstack([means-r.ci_2_5.to_numpy(), r.ci_97_5.to_numpy()-means]),
                fmt='o', capsize=3, color=color, label=name)
        ax.axhline(0, color='#888888', linewidth=.8); ax.set_xticks(range(4), labels)
        ax.set_ylim(-.4, .02); ax.set_ylabel('Paired mean clipped-current difference')
        ax.set_title('Variance x'+str(factor)+': first 512 post-change points'); ax.legend(frameon=False, fontsize=8)
    for ax in axes: ax.tick_params(axis='x', rotation=15)
    fig.suptitle('Energy attenuation coexists with a signed forecast response', fontsize=14)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.text(.03, .005, 'Left: partition-0 H distributions; box IQR, whiskers 5th/95th percentiles. Right: partition-3 matched pairs, 2,000 conditional bootstrap draws.\nDifferent cohorts; channels precede evidence summaries and trees. These are explanatory diagnostics, not model-performance estimates.', fontsize=8)
    name = '09_scale_calibration_and_channel_response'
    for suffix in ('png', 'pdf'): fig.savefig(out/(name+'.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    write_json(out/'SCALE_CHANNEL_FIGURE_MANIFEST.json', {'status': 'PASS', 'created_utc': utc_now(),
        'script_sha256': sha256(Path(__file__)), 'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in (a, b)},
        'files_sha256': {name+'.'+suffix: sha256(out/(name+'.'+suffix)) for suffix in ('png', 'pdf')},
        'caption': 'Fixed H calibration and paired current-channel response in distinct synthetic cohorts; before evidence summaries and trees. Conditional percentile intervals do not include adaptive selection uncertainty.'})
    print('NEXT2 SCALE CHANNEL FIGURE PASS', flush=True)


if __name__ == '__main__': main()
