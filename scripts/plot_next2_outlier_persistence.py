"""Show fixed all-DEV and matched CV0 outlier persistence in separate panels."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = ROOT/'artifacts/next2/figures'; out.mkdir(parents=True, exist_ok=True)
    sources = {}; frames = []
    for directory in ('outlier_age_expiry', 'reused_outlier_age_expiry'):
        folder = ROOT/'artifacts/next2'/directory
        receipt = folder/'RESULTS.json'; r = json.loads(receipt.read_text()); assert r['status'] == 'PASS'
        path = folder/'MEAN_CURVES.csv'; assert digest(path) == r['files_sha256'][path.name]
        frames.append(pd.read_csv(path, float_precision='round_trip'))
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
        sources[receipt.relative_to(ROOT).as_posix()] = digest(receipt)
    data = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    panels = [('all_DEV', 'All-DEV frozen models', [
        ('J102_SERIAL_VARIANCE_512', 'J102', '#536878'),
        ('B104_LONG_VARIANCE_NO_GLR_MEMORY', 'B104', '#b66b3b'),
        ('C205_DEPENDENCE', 'C205', '#43936b'),
        ('BL2_C205_DEPENDENCE_80', '80/20 blend', '#168397')]),
        ('original_CV_fold_0', 'Matched original fold-0 ablations', [
        ('Y001_LONG_VARIANCE_ARCH1', 'Y001 full', '#536878'),
        ('B101_LONG_VARIANCE_NO_MEMORY', 'No GLR memory', '#267cad'),
        ('B102_LONG_VARIANCE_NO_BAYES', 'No Bayes', '#ac5260'),
        ('V007_VARIANCE_FORECAST', 'Maximum age 128', '#b98b35')])]
    for ax, (fitted, title, models) in zip(axes, panels):
        for eid, label, color in models:
            values = data[(data.candidate == eid) & (data.fitted_model == fitted)].sort_values('post_injection_offset')
            assert len(values) == 768 and values.seed_pairs.eq(342).all()
            ax.plot(values.post_injection_offset, values.paired_probability_delta, label=label, color=color, linewidth=1.8)
        for boundary in (128, 256, 512): ax.axvline(boundary, color='#cccccc', linewidth=.8, linestyle=':')
        ax.axhline(0, color='#999999', linewidth=.7)
        ax.set_title(title); ax.set_xlabel('Observations after the single injected outlier')
        ax.set_xlim(0, 768); ax.set_xticks([0, 128, 256, 512, 768])
        ax.legend(frameon=False, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Mean probability difference: outlier minus paired control')
    fig.suptitle('Persistence after the original age windows: the observations differ at one point only')
    name = '08_outlier_persistence_and_memory'
    for suffix in ('png', 'pdf'): fig.savefig(out/(name+'.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)
    report = {'status': 'PASS', 'script_sha256': digest(Path(__file__)), 'source_sha256': sources,
        'files_sha256': {name+'.'+suffix: digest(out/(name+'.'+suffix)) for suffix in ('png', 'pdf')},
        'caption': 'Fixed original O1 trajectories, availability-defined subgroup of 342 seed pairs with injection onset 128/256. Left and right use different fitted-model populations and are deliberately separate. Mean curves are descriptive; fixed-offset conditional bootstrap intervals and boundary contrasts are in the source CSVs. Vertical guides indicate existing age-grid windows. No new fits, no synthetic selection and no single-feature causal claim.'}
    (out/'OUTLIER_FIGURE_MANIFEST.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('NEXT2 OUTLIER FIGURE PASS', flush=True)


if __name__ == '__main__': main()
