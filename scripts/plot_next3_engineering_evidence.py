"""Plot the existing E003 cost trials with their actual aggregation definitions."""
from pathlib import Path
import importlib.metadata
import statistics
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now


def main():
    versions = {}
    for line in (ROOT/'requirements-next2-figures-lock.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'): continue
        name, version = line.split('=='); assert importlib.metadata.version(name) == version
        versions[name] = version
    cold_path = BASE/'engineering/lean_shared/cold/RESULTS.json'
    direct_path = BASE/'engineering/lean_shared/direct/RESULTS.json'
    cold, direct = read_json(cold_path), read_json(direct_path)
    assert cold['status'] == direct['status'] == 'PASS'
    assert cold['cold_cost_gate_pass'] and direct['direct_cost_gate_pass']
    for path, record in ((cold_path, cold), (direct_path, direct)):
        for name, sha in record['files_sha256'].items(): assert digest(path.parent/name) == sha
    rss = np.array([[p['packed_RSS_bytes'], p['lean_RSS_bytes']] for p in cold['pairs']], dtype=np.float64)/2**20
    ready = np.array([[p['packed_first_none_seconds'], p['lean_first_none_seconds']] for p in cold['pairs']])
    ratios = np.array([p['direct_10M_ratio'] for p in direct['pairs']])
    assert rss.shape == ready.shape == (3, 2) and ratios.shape == (3,)
    assert abs(np.median(rss[:, 1])/np.median(rss[:, 0])-cold['median_RSS_ratio']) < 1e-14
    assert float(np.median(ratios)) == direct['median_direct_10M_ratio']
    out = BASE/'figures/engineering_evidence_v1'; out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.5))
    colors = ['#BA785B', '#287C9C']
    for axis, values, title, ylabel in ((axes[0], rss, 'Fresh-process post-warm RSS', 'MiB'),
                                       (axes[1], ready, 'Import to first ready None', 'Seconds')):
        medians = np.median(values, axis=0)
        axis.bar([0, 1], medians, width=.57, color=colors, alpha=.83)
        for pair in range(3):
            offset = (pair-1)*.095
            axis.scatter(np.array([0, 1])+offset, values[pair], color='#182936', s=28, zorder=4)
        for index, value in enumerate(medians):
            axis.text(index, value*1.035, f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        axis.set_xticks([0, 1], ['E002 packed', 'E003 lean'])
        axis.set_ylim(0, float(values.max())*1.24); axis.set_ylabel(ylabel); axis.set_title(title, pad=12)
        axis.yaxis.grid(True, alpha=.18); axis.set_axisbelow(True)
    axes[0].text(.5, .97, f'{(1-cold["median_RSS_ratio"])*100:.2f}% lower median RSS',
                 transform=axes[0].transAxes, ha='center', va='top', color='#135970', fontweight='bold')
    threshold = .8*statistics.median(p['packed_first_none_seconds'] for p in cold['pairs'])
    axes[1].axhline(threshold, linestyle=':', color='#963F33', linewidth=1.5)
    axes[1].text(.03, .97, f'{(1-cold["median_first_none_ratio"])*100:.2f}% lower median ready time',
                 transform=axes[1].transAxes, ha='left', va='top', fontsize=9)
    axes[1].text(.02, threshold/axes[1].get_ylim()[1]-.05, '20% startup-only target: not reached',
                 transform=axes[1].transAxes, fontsize=8, color='#963F33')
    axis = axes[2]; axis.axhline(1., color='#6D7377', linewidth=1)
    axis.axhline(1.05, color='#963F33', linestyle='--', linewidth=1.5)
    axis.scatter(np.arange(3), ratios, color=colors[1], s=60, zorder=3)
    for index, value in enumerate(ratios):
        axis.text(index, value+.003, f'{value:.6f}', ha='center', fontsize=9)
    axis.set_xticks(np.arange(3), ['Pair 0', 'Pair 1', 'Pair 2'])
    axis.set_ylim(.98, 1.062); axis.set_xlim(-.5, 2.5)
    axis.set_ylabel('E003 / E002 measured seconds'); axis.set_title('Actual direct 10M time ratio', pad=12)
    axis.text(.02, .96, 'Fixed upper bound: 1.05', transform=axis.transAxes, fontsize=9, color='#963F33', va='top')
    axis.text(.02, .03, f'Median paired ratio: {direct["median_direct_10M_ratio"]:.6f}\nNo throughput improvement claimed',
              transform=axis.transAxes, fontsize=9)
    axis.yaxis.grid(True, alpha=.18); axis.set_axisbelow(True)
    figure.suptitle('NEXT3 engineering: identical predictions, lower inference process memory', fontsize=16, y=.99)
    figure.subplots_adjust(left=.06, right=.985, top=.79, bottom=.20, wspace=.29)
    figure.text(.06, .085,
        'Three paired trials on the same pinned Windows host. Dots are observed trials, not confidence intervals.\n'
        'Cold panels compare ratios of medians; the direct panel reports the median of paired ratios. '
        'Retained NumPy state, model-file size and process RSS are distinct quantities.', fontsize=9, color='#41474C')
    for suffix in ('png', 'pdf'): figure.savefig(out/('01_runtime_cost_evidence.'+suffix), dpi=180, facecolor='white')
    plt.close(figure)
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'data_sources_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in (cold_path, direct_path)},
        'files_sha256': {path.name: digest(path) for path in out.iterdir() if path.suffix in ('.png', '.pdf')},
        'figure_environment_versions': versions, 'script_sha256': digest(Path(__file__)),
        'new_measurements': 0, 'new_fits': 0, 'new_quality_scores': 0, 'confidence_intervals_shown': False,
        'visual_review': 'PENDING_SEPARATE_IMAGE_INSPECTION'}, immutable=True)
    progress('Engineering cost evidence figure prepared', 'Existing cold RSS, ready-None and three paired actual 10M trials; no new measurement or speedup claim')
    print('NEXT3 ENGINEERING FIGURE PASS; VISUAL REVIEW REQUIRED', flush=True)


if __name__ == '__main__': main()
