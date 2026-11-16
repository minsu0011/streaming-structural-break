"""Local synthetic-only bridge to the unchanged official execute() function."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import json
import shutil
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import next_registered_main as entry
from src.validation.protocol import official_runner
from src.utils.artifacts import write_json, sha256


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source-model', required=True)
    parser.add_argument('--parameters', default='{}')
    args = parser.parse_args()
    def stage_synthetic_fixture(datasets, model_directory_path):
        import numpy as np
        count = 0
        for sid, historical, online, tau in datasets:
            assert isinstance(sid, (int, np.integer))
            h, o = np.asarray(historical), np.asarray(online)
            assert h.ndim == 1 and o.ndim == 1 and np.isfinite(h).all() and np.isfinite(o).all()
            count += 1
        if count != 20:
            raise RuntimeError('Unexpected synthetic official train iterator coverage')
        source, target = Path(args.source_model), Path(model_directory_path)
        if (target/'model.joblib').exists() or (target/'model.json').exists():
            raise RuntimeError('Synthetic stager will not overwrite an existing model')
        target.mkdir(parents=True, exist_ok=True)
        for name in ('model.joblib', 'model.json'):
            shutil.copy2(source/name, target/name)
        write_json(target/'SYNTHETIC_STAGING.json', {'status': 'PASS', 'synthetic_training_tuples_consumed': count,
            'training_function': 'Engineering fixture staging only; no fit on synthetic IDs outside the frozen DEV universe.',
            'source_model_sha256': sha256(source/'model.joblib')})
    class Module:
        def get_function(self, name):
            return stage_synthetic_fixture if name == 'train' else getattr(entry, name)
        def get_value(self, name, default=None):
            return getattr(entry, name, default)
    commands = official_runner().execute(SimpleNamespace(trip_data_fuse=lambda: None), Module(), args.data, args.model)
    commands[args.command](**json.loads(args.parameters))
