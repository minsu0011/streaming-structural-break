"""Evaluate fixed held-out router outputs against their probability-loss surrogate."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.evaluation import load_full_oof as load_new_oof
from src.next2.evaluation import load_full_oof as load_frozen_oof
from src.next3.gate_serialization import descriptor_predict, descriptor_weights
from src.next3.forensic import positive_pair_credit
from src.next.fitting import weights_from_training
from src.next.data import prepare_dev
from src.scoring.ts_auc import ts_auc

J = 'J102_SERIAL_VARIANCE_512'; C = 'C205_DEPENDENCE'; B = 'BL2_C205_DEPENDENCE_80'
ROUTERS = ('R1_SCORE_GAP', 'R1_EVIDENCE_RATIO', 'R2_LOGISTIC_GATE', 'R3_DEPTH2_GATE')


def log_loss_rows(y, p):
    probability = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.-1e-7)
    return -(y*np.log(probability)+(1-y)*np.log1p(-probability))


def main():
    hyp_path = BASE/'hypotheses/D004_ROUTER_METRIC_ALIGNMENT.json'; hyp = read_json(hyp_path)
    assert hyp['script_sha256'] == digest(Path(__file__))
    out = BASE/'router_metric_alignment'; out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'POLICY.json', {'created_utc': utc_now(), 'hypothesis_sha256': digest(hyp_path),
        'script_sha256': digest(Path(__file__)), 'scope': 'Existing original held-out OOF only; no training-row scoring or new fit'}, immutable=True)
    data = prepare_dev(ROOT); metadata = data.rows(); predictions = {}; sources = {}; expected_scores = {}
    for kind in (J, C, B):
        _, record, meta, prediction, proof = load_frozen_oof(data, kind)
        predictions[kind] = prediction; sources[kind] = proof
    for kind in ROUTERS:
        _, record, meta, prediction, proof = load_new_oof(data, kind)
        predictions[kind] = prediction; sources[kind] = proof; expected_scores[kind] = record['fold_scores']
    rows = []; weights = []
    for fold in range(5):
        take = np.flatnonzero(metadata['fold'] == fold)
        y = metadata['target'][take]; t = metadata['time_online'][take]; ids = metadata['dataset_id'][take]
        w = weights_from_training(y, t, ids, 'W3')
        baseline = predictions[B][take]; baseline_loss = log_loss_rows(y, baseline)
        baseline_score = ts_auc(y, baseline, t)
        baseline_credit, pair_exposure = positive_pair_credit(y, baseline, t)
        pair_count = int(pair_exposure.sum()); assert pair_count > 0
        for kind, prediction in predictions.items():
            p = prediction[take]; loss = log_loss_rows(y, p); score = ts_auc(y, p, t)
            if kind in expected_scores: np.testing.assert_allclose(score, expected_scores[kind][fold], rtol=0, atol=1e-14)
            credit, pairs = positive_pair_credit(y, p, t); np.testing.assert_array_equal(pair_exposure, pairs)
            delta_credit = credit-baseline_credit
            np.testing.assert_allclose(delta_credit.sum()/pair_count, score-baseline_score, rtol=0, atol=1e-14)
            loss_delta = loss-baseline_loss
            rows.append({'candidate': kind, 'fold': fold, 'ts_auc': score, 'delta_ts_auc': score-baseline_score,
                'W3_weighted_log_loss': float(np.average(loss, weights=w)),
                'delta_W3_weighted_log_loss': float(np.average(loss_delta, weights=w)),
                'unweighted_log_loss': float(loss.mean()), 'delta_unweighted_log_loss': float(loss_delta.mean()),
                'W3_weighted_brier': float(np.average((p-y)**2, weights=w)),
                'pair_credit_delta_exact': float(delta_credit.sum()/pair_count),
                'positive_pair_exposure_LL_better_rank_worse': float(pair_exposure[(loss_delta < 0)&(delta_credit < 0)].sum()/pair_count),
                'positive_pair_exposure_LL_worse_rank_better': float(pair_exposure[(loss_delta > 0)&(delta_credit > 0)].sum()/pair_count),
                'evaluation_kind': 'ORIGINAL_HELD_OUT_OOF_DIAGNOSTIC_ONLY', 'new_fit': False})
        nested = BASE/'nested/full'/f'outer_{fold}'; receipt = read_json(nested/'RESULTS.json')
        assert receipt['status'] == 'PASS' and receipt['outer_validation_never_used_for_inner_training']
        for name in ('GATE_VALID.npy', 'VALID_GLOBAL_ROWS.npy'):
            assert digest(nested/name) == receipt['files_sha256'][name]
        np.testing.assert_array_equal(np.load(nested/'VALID_GLOBAL_ROWS.npy', allow_pickle=False), take)
        xvalid = np.load(nested/'GATE_VALID.npy', mmap_mode='r', allow_pickle=False)
        for kind in ('R2_LOGISTIC_GATE', 'R3_DEPTH2_GATE'):
            path = BASE/'experiments'/kind/'full'/f'fold_{fold}_gate.json'
            outer_receipt = read_json(path.parent/f'fold_{fold}.json')
            assert digest(path) == outer_receipt['files_sha256'][path.name]
            descriptor = read_json(path); actual = descriptor_predict(descriptor, xvalid)
            np.testing.assert_array_equal(actual, predictions[kind][take])
            weight = descriptor_weights(descriptor, xvalid)
            primary_loss = log_loss_rows(y, xvalid[:, 0]); complement_loss = log_loss_rows(y, xvalid[:, 1])
            weights.append({'candidate': kind, 'fold': fold, 'weight_mean': float(weight.mean()),
                'W3_weighted_weight_mean': float(np.average(weight, weights=w)),
                'weight_min': float(weight.min()), 'weight_max': float(weight.max()),
                'fraction_weight_below_frozen_020': float(np.mean(weight < .2)),
                'fraction_weight_at_frozen_020': float(np.mean(weight == .2)),
                'W3_fraction_C_standalone_logloss_better_than_J': float(np.average(complement_loss < primary_loss, weights=w)),
                'gate_json_sha256': digest(path), 'no_training_rows_scored': True})
        progress('Router objective alignment forensic', f'Outer fold {fold}: fixed OOF log-loss and exact pair credits; no training or retuning', experiment='D004_ROUTER_METRIC_ALIGNMENT')
        print('NEXT3 ROUTER METRIC ALIGNMENT', fold, flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(out/'FOLD_METRIC_ALIGNMENT.csv', index=False)
    pd.DataFrame(weights).to_csv(out/'HELDOUT_GATE_WEIGHTS.csv', index=False)
    numeric = ['ts_auc', 'delta_ts_auc', 'W3_weighted_log_loss', 'delta_W3_weighted_log_loss',
               'unweighted_log_loss', 'delta_unweighted_log_loss', 'W3_weighted_brier',
               'positive_pair_exposure_LL_better_rank_worse', 'positive_pair_exposure_LL_worse_rank_better']
    summary = frame.groupby('candidate')[numeric].mean().reset_index()
    summary['LL_improves_while_TS_AUC_drops'] = (summary.delta_W3_weighted_log_loss < 0)&(summary.delta_ts_auc < 0)
    summary.to_csv(out/'SUMMARY.csv', index=False); data.guard.check_state()
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'hypothesis_sha256': digest(hyp_path),
        'source_bindings': sources, 'models': len(predictions), 'folds': 5, 'new_fits': 0, 'seal_rows': 0,
        'new_candidate_selection': False, 'training_rows_scored': 0,
        'files_sha256': {p.name: digest(p) for p in out.glob('*.csv')},
        'limitation': 'Descriptive existing held-out outputs. Log loss treats scores as probabilities; TS-AUC depends only on same-time rankings. Direction disagreement supports a surrogate mismatch interpretation but does not identify a causal repair or establish independent validation.'}, immutable=True)


if __name__ == '__main__': main()
