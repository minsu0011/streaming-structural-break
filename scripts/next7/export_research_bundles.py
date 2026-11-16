"""Copy exact fitted bytes into portable research bundles; no refitting or role change."""
from pathlib import Path
import hashlib
import shutil
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next7.common import BASE, read_json, save, digest, ids_hash, utc_now, require_guard, activity, note
from src.next7.bundle import RECIPES, load


def main():
    guard = require_guard()
    sources = ('src/next7/bundle.py', 'src/next7/integrated_runtime.py', 'src/next7/model.py',
        'src/next7/pairwise.py', 'src/next7/features.py', 'src/next7/event.py', 'src/next7/scale.py',
        'src/next7/null_growth.py', 'src/next7/calibration.py', 'src/next5/runtime.py',
        'src/next5/model.py', 'src/next6_extension/fresh_state.py', 'src/next6_extension/fresh_runtime.py')
    records = []
    with activity('Export mode-safe research-only streaming bundles', 'REPRODUCTION_ENGINEERING', 'F5_CAUSAL_NESTED_INTEGRATION'):
        for fold in range(5):
            relative = f'research_bundles/fold_{fold}'
            directory = BASE / relative
            if (directory / 'MANIFEST.json').exists():
                loaded = load(directory)
                records.append({'fold': fold, 'manifest_sha256': digest(directory / 'MANIFEST.json')})
                continue
            contract = read_json(BASE / f'nested_base/fold_{fold}/CONTRACT.json')
            train = np.array(contract['training_ids'], dtype=np.int32)
            guard.partition(train, contract['outer_validation_ids'])
            originals = {kind + '.joblib': ROOT / f'artifacts/next5/references/fits/PRIMARY/full/{kind}/fold_{fold}/model.joblib'
                for kind in ('J102', 'C205')}
            originals.update({'I4_EVENT_SCALE_TREE.joblib': BASE / f'g1/fold_{fold}/I4_EVENT_SCALE_TREE/model.joblib',
                'G2_FULL.joblib': BASE / f'g2/fold_{fold}/G2_FULL/model.joblib',
                'G3_PAIRWISE_FULL.joblib': BASE / f'g3/fold_{fold}/model.joblib',
                'SCALE_CALIBRATION.json': BASE / 'SCALE_CALIBRATION_FROZEN.json'})
            directory.mkdir(parents=True, exist_ok=True)
            hashes, origins = {}, {}
            for name, original in originals.items():
                target = directory / name
                if not target.exists():
                    with original.open('rb') as source, target.open('xb') as output:
                        shutil.copyfileobj(source, output)
                assert digest(target) == digest(original)
                hashes[name] = digest(target)
                origins[name] = original.relative_to(ROOT).as_posix()
            save(relative + '/MANIFEST.json', {'format': 'NEXT7_RESEARCH_BUNDLE_V1', 'status': 'FROZEN',
                'created_utc': utc_now(), 'role': 'RESEARCH_FOLD_ONLY_NOT_PROMOTED', 'default_method': 'M0',
                'fold': fold, 'qualified_state_by_recipe': RECIPES, 'training_ids_sha256': ids_hash(train),
                'base_training_int32_sha256': hashlib.sha256(np.sort(train).tobytes()).hexdigest(),
                'training_series': len(train), 'new_fits': 0, 'copies_match_original_model_bytes': True,
                'files_sha256': hashes, 'original_relative_paths': origins,
                'source_sha256': {n: digest(ROOT / n) for n in sources},
                'scope_limit': 'Frozen outerTRAIN fitted artifacts for research replay. No all-RP deployment refit or promotion.',
                'input_contract': 'new_state(H), then predict_one(current_scalar); no dataset ID, tau or future length.',
                'score_semantics': 'Ranking/evidence outputs, not calibrated break probabilities.'})
            load(directory)
            records.append({'fold': fold, 'manifest_sha256': digest(directory / 'MANIFEST.json')})
        save('RESEARCH_BUNDLES_EXPORTED.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'folds': records, 'new_fits': 0, 'primary_role_changed': False, 'source_sha256': digest(__file__)})
    note('Research streaming bundles exported', 'Exact fitted bytes, mode-safe raw versus qualified mapping, BL80 remains default', 'F5_CAUSAL_NESTED_INTEGRATION')


if __name__ == '__main__':
    main()
