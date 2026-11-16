"""Attribute the fixed blend delta to positive-side mechanism and break age."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next3.forensic import positive_pair_credit
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.scoring.ts_auc import ts_auc

MODELS = {'J102':'J102_SERIAL_VARIANCE_512', 'C205':'C205_DEPENDENCE', 'BL80':'BL2_C205_DEPENDENCE_80'}


def main():
    assert read_json(BASE/'preflight/NEXT3_PREFLIGHT.json')['status'] == 'PASS'
    out = BASE/'pair_contributions'
    hypothesis = read_json(BASE/'hypotheses/F001_PAIR_CONTRIBUTIONS.json')
    if (out/'RESULTS.json').exists():
        raise RuntimeError('Do not overwrite complete diagnostic')
    data = prepare_dev(ROOT)
    meta_path = ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    lock = read_json(meta_path.parent/'REGIME_ASSIGNMENT_LOCK.json')
    assert digest(meta_path) == lock['files_sha256'][meta_path.name]
    series = pd.read_parquet(meta_path)
    data.guard.admit(series.dataset_id.to_numpy(), purpose='Evaluation-only pair attribution')
    categories = sorted(series.signature.unique().tolist())
    lookup = np.full(max(data.guard.dev_ids)+1, -1, dtype=np.int16)
    taus = np.full(len(lookup), -1, dtype=np.int32)
    lookup[series.dataset_id] = [categories.index(value) for value in series.signature]
    taus[series.dataset_id] = series.tau
    predictions, sources, reference_scores = {}, {}, {}
    metadata = None
    for role, eid in MODELS.items():
        candidate, report, md, predictions[role], sources[role] = load_full_oof(data, eid)
        reference_scores[role] = report['full']['fold_scores']
        if metadata is not None:
            for key in metadata:
                np.testing.assert_array_equal(md[key], metadata[key])
        metadata = md
        progress('B1 fixed-model pair attribution', f'Validated original five-fold OOF: {role}')
    groups = lookup[metadata['dataset_id']]
    ages = metadata['time_online']-taus[metadata['dataset_id']]
    age_edges = [0, 5, 10, 20, 50, 100, 100000]
    rows = []
    for fold in range(5):
        subset = np.flatnonzero(metadata['fold'] == fold)
        y, t = metadata['target'][subset], metadata['time_online'][subset]
        credits = {}
        for role, pred in predictions.items():
            credits[role], mass = positive_pair_credit(y, pred[subset], t)
            np.testing.assert_allclose(credits[role].sum()/mass.sum(), reference_scores[role][fold], rtol=0, atol=1e-14)
        total = int(mass.sum())
        masks = [('mechanism', category, groups[subset] == k) for k, category in enumerate(categories)]
        masks += [('break_age', f'{a}-{b-1}' if b < 100000 else f'{a}+',
                   (ages[subset] >= a) & (ages[subset] < b)) for a, b in zip(age_edges[:-1], age_edges[1:])]
        for scope, name, mask in masks:
            positive = mask & (y == 1)
            pairs = int(mass[mask].sum())
            item = {'fold':fold, 'scope':scope, 'group':name,
                'positive_rows':int(positive.sum()), 'positive_series':len(np.unique(metadata['dataset_id'][subset][positive])),
                'pairs':pairs, 'fold_total_pairs':total, 'fraction_of_official_pair_mass':pairs/total}
            for role in predictions:
                wins = float(credits[role][mask].sum())
                item[role+'_contribution'] = wins/total
                item[role+'_conditional_pair_auc'] = wins/pairs if pairs else None
            item['blend_minus_J102_contribution'] = item['BL80_contribution']-item['J102_contribution']
            item['perfect_pair_ranking_headroom_contribution'] = pairs/total-item['BL80_contribution']
            rows.append(item)
        progress('B1 fixed-model pair attribution', f'Fold {fold}: mechanism/age pair mass and exact score contributions')
        print('NEXT3 PAIR ATTRIBUTION', fold, total, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'FOLD_PAIR_CONTRIBUTIONS.csv', index=False)
    numeric = ['fraction_of_official_pair_mass','J102_contribution','C205_contribution','BL80_contribution',
        'blend_minus_J102_contribution','perfect_pair_ranking_headroom_contribution']
    summary = frame.groupby(['scope','group'], as_index=False)[numeric].mean()
    support = frame.groupby(['scope','group'], as_index=False)[['positive_series','positive_rows','pairs']].sum()
    summary = summary.merge(support, on=['scope','group'], validate='one_to_one')
    summary['supported_at_least_100_break_series'] = summary.positive_series >= 100
    summary.to_csv(out/'SUMMARY.csv', index=False)
    expected_delta = np.mean(reference_scores['BL80'])-np.mean(reference_scores['J102'])
    for scope in ('mechanism','break_age'):
        part = summary[summary.scope == scope]
        np.testing.assert_allclose(part.blend_minus_J102_contribution.sum(), expected_delta, rtol=0, atol=1e-14)
        np.testing.assert_allclose(part.fraction_of_official_pair_mass.sum(), 1., rtol=0, atol=1e-14)
    atomic_json(out/'RESULTS.json', {'status':'PASS','created_utc':utc_now(),
        'hypothesis_sha256':digest(BASE/'hypotheses/F001_PAIR_CONTRIBUTIONS.json'), 'sources':sources,
        'model_folds':reference_scores, 'exact_blend_mean_delta':float(expected_delta),
        'signature_sha256':digest(meta_path), 'script_sha256':digest(Path(__file__)),
        'implementation_sha256':digest(ROOT/'src/next3/forensic.py'),
        'new_fits':0, 'new_candidates':0, 'seal_rows':0,
        'files_sha256':{p.name:digest(p) for p in out.glob('*.csv')},
        'interpretation':'Additive official pair contributions allocate every pair by its positive series. Negative scores from all series participate. These differ from mechanism-plus-null subset AUCs.',
        'headroom_limitation':'Perfect pair ordering is an unattainable descriptive ceiling, not evidence that a causal model can obtain it. No oracle scores enter model selection.',
        'mechanism_and_tau_are_evaluation_only':True}, immutable=True)
    print('NEXT3 PAIR ATTRIBUTION PASS', expected_delta, flush=True)


if __name__ == '__main__':
    main()
