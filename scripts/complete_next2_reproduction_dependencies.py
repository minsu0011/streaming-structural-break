"""Complete import-time configuration dependencies before any independent fitting starts."""
from pathlib import Path
import sys
import json
import shutil
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now
from src.next2.session import clock


def main():
    assert clock()['audit_started']
    out = ROOT/'artifacts/next2/independent_reproduction'
    preparation = json.loads((out/'PREPARATION.json').read_text()); payload = Path(preparation['payload'])
    assert not (out/'PREPARATION_READY.json').exists()
    assert not (payload/'artifacts/independent').exists()
    manifest_path = payload/'COPIED_SOURCE_MANIFEST.json'
    assert sha256(manifest_path) == preparation['copied_manifest_sha256']
    original = json.loads(manifest_path.read_text())
    shutil.copyfile(manifest_path, payload/'COPIED_SOURCE_MANIFEST_BEFORE_CONFIG_COMPLETION.json')
    additions = ['configs/'+name+'.json' for name in ('features', 'arch_input', 'ar_residual_input',
        'historical_context', 'historical_quantiles', 'rank_gaussian', 'magnitude_dependence')]
    additions += ['ROBUSTNESS_SPLIT_LOCK.json']
    for relative in additions:
        source, destination = ROOT/relative, payload/relative
        assert not destination.exists(); destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination); assert sha256(source) == sha256(destination)
        original['files_sha256'][relative] = sha256(source)
    original['dependency_completion_utc'] = utc_now(); write_json(manifest_path, original)
    report = {**preparation, 'status': 'PREPARED', 'created_utc': utc_now(),
        'copied_manifest_sha256': sha256(manifest_path), 'copied_files': len(original['files_sha256']),
        'initial_preparation_receipt_sha256': sha256(out/'PREPARATION.json'),
        'completion_script_sha256': sha256(Path(__file__)), 'added_dependencies': additions,
        'preflight_finding': 'Initial copy omitted frozen JSON defaults imported by legacy feature modules. Static dependency review found this before any child process or learner fit.',
        'training_fits_before_completion': 0, 'training_or_model_source_changed': False,
        'initial_manifest_preserved': 'COPIED_SOURCE_MANIFEST_BEFORE_CONFIG_COMPLETION.json'}
    write_json(out/'PREPARATION_READY.json', report)
    print('NEXT2 INDEPENDENT CONFIG DEPENDENCIES COMPLETE', len(additions), flush=True)


if __name__ == '__main__': main()
