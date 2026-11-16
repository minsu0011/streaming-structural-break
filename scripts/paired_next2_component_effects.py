"""Conditional paired attribution for completed full-CV component controls.

Failed promotion gates remain failed. This analysis trains no models and uses
the same whole-series bootstrap draws as the original NEXT2 forensic policy.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from src.next.bootstrap import PreparedAUC, paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from src.next2.session import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now

OUT = ROOT / 'artifacts/next2/component_effects'
PRIMARY = 'J102_SERIAL_VARIANCE_512'


def existing_vectors(eid):
    for generation in ('next2', 'next'):
        folder = ROOT / f'artifacts/{generation}/forensic' / eid
        receipt = folder / 'RESULTS.json'
        if not receipt.exists():
            continue
        report = json.loads(receipt.read_text(encoding='utf-8'))
        vectors, digests = [], {}
        for fold in range(5):
            name = f'fold_{fold}_all_time.npy'
            if sha256(folder / name) != report['files_sha256'][name]:
                raise RuntimeError('Reused bootstrap changed: ' + eid)
            values = np.load(folder / name, allow_pickle=False)
            assert values.shape == (2001,) and np.isfinite(values).all()
            vectors.append(values)
            digests[(folder / name).relative_to(ROOT).as_posix()] = sha256(folder / name)
        return np.asarray(vectors).T, digests
    return None


def main():
    set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    data = prepare_dev(ROOT)
    rows, receipts = [], {}
    refs = {}
    for path in sorted((ROOT / 'artifacts/next2/experiments').glob('*/RESULTS.json')):
        original = json.loads(path.read_text(encoding='utf-8'))
        candidate = original['candidate']
        eid = candidate['experiment_id']
        if original.get('full', {}).get('status') != 'COMPLETE':
            continue
        if candidate.get('blend') or candidate.get('parity_reference') or eid.startswith('C2'):
            continue
        parent = candidate['parent_id']
        for reference in {PRIMARY, parent}:
            if reference not in refs:
                refs[reference] = existing_vectors(reference)
                if refs[reference] is None:
                    raise RuntimeError('Exact parent reference bootstrap missing: ' + reference)
        folder = OUT / eid
        folder.mkdir(exist_ok=True)
        identity = {'candidate_results_sha256': sha256(path), 'script_sha256': sha256(Path(__file__)),
                    'policy_sha256': sha256(ROOT / 'configs/next2/RESEARCH_POLICY_LOCK.json'),
                    'replicates': 2000, 'seed_base': 202609094, 'candidate': eid,
                    'parent': parent, 'original_status': original['status']}
        policy = folder / 'POLICY.json'
        if policy.exists():
            assert json.loads(policy.read_text()) == identity
        else:
            write_json(policy, identity)
        reused = existing_vectors(eid)
        if reused is not None:
            values, hashes = reused
            mode = 'EXACT_EXISTING_FORENSIC_REUSE'
        else:
            c, _, metadata, prediction, sources = load_full_oof(data, eid)
            vectors, hashes = [], {}
            for fold in range(5):
                target = folder / f'fold_{fold}_all_time.npy'
                receipt = target.with_suffix('.json')
                if target.exists() and receipt.exists():
                    r = json.loads(receipt.read_text())
                    assert r['sha256'] == sha256(target) and r['oof_sources'] == sources
                    vector = np.load(target, allow_pickle=False)
                else:
                    mask = metadata['fold'] == fold
                    ids, y, t = (metadata[k][mask] for k in ('dataset_id', 'target', 'time_online'))
                    code, counts = draw_group_multiplicities(ids, 2000, 202609094 + fold)
                    vector = PreparedAUC.create(y, prediction[mask], t, code, counts.shape[1]).evaluate(counts)
                    np.save(target, vector, allow_pickle=False)
                    write_json(receipt, {'sha256': sha256(target), 'oof_sources': sources})
                assert vector.shape == (2001,) and np.isfinite(vector).all()
                np.testing.assert_allclose(vector[0], original['full']['fold_scores'][fold], rtol=0, atol=1e-14)
                vectors.append(vector)
                hashes[target.relative_to(ROOT).as_posix()] = sha256(target)
            values = np.asarray(vectors).T
            mode = 'NEW_BOOTSTRAP_EXISTING_OOF_ZERO_FITS'
        for reference in sorted({PRIMARY, parent}):
            ref, refhash = refs[reference]
            rows.append({'candidate': eid, 'reference': reference, 'is_direct_parent': reference == parent,
                         'original_status': original['status'], 'mode': mode,
                         'candidate_mean': float(values[0].mean()), 'reference_mean': float(ref[0].mean()),
                         'candidate_median': float(np.median(values[0])), 'candidate_worst': float(values[0].min()),
                         **paired_summary(ref, values),
                         'probability_delta_above_minus_0005': float(np.mean((values.mean(1)-ref.mean(1))[1:] > -.0005))})
        write_json(folder / 'RESULTS.json', {'status': 'PASS', **identity, 'mode': mode,
            'files_sha256': hashes, 'reference_files_sha256': {r: refs[r][1] for r in {PRIMARY, parent}},
            'new_model_fits': 0, 'candidate_promotion_status_unchanged': True})
        receipts[(folder / 'RESULTS.json').relative_to(ROOT).as_posix()] = sha256(folder / 'RESULTS.json')
        pd.DataFrame(rows).to_csv(OUT / 'PAIRED_EFFECTS.csv', index=False)
        checkpoint('Paired component attribution ' + eid, ['Continue fixed synthetic stress', 'Consolidate component gates'])
        print('NEXT2 COMPONENT EFFECT', eid, rows[-1]['estimate_mean_delta'], flush=True)
    write_json(OUT / 'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'comparisons': len(rows),
        'candidate_count': len(receipts), 'new_model_fits': 0, 'files_sha256': receipts,
        'table_sha256': sha256(OUT / 'PAIRED_EFFECTS.csv'),
        'limitations': 'Conditional fixed-OOF bootstrap. No training, adaptive selection or multiplicity correction. Failed gates remain failed.'})


if __name__ == '__main__':
    main()
