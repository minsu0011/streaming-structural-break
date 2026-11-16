"""Read-only, recipe-safe research bundles; default method remains frozen BL80."""
from pathlib import Path
import numpy as np
from src.next7.common import ROOT, read_json, digest
from src.next7.integrated_runtime import Prepared
from src.next7.model import Correction
from src.next5.model import FeatureModel
from src.next5.runtime import SharedResearchBaseline

RECIPES = {'M_EVENT': False, 'G2_EVENT': True, 'I4_EVENT_SCALE_TREE': False,
    'G2_FULL': True, 'G3_PAIRWISE_FULL': True}


def load(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'MANIFEST.json')
    if manifest.get('format') != 'NEXT7_RESEARCH_BUNDLE_V1' or manifest.get('status') != 'FROZEN':
        raise ValueError('Frozen NEXT7 research bundle required')
    if manifest.get('role') != 'RESEARCH_FOLD_ONLY_NOT_PROMOTED' or manifest.get('default_method') != 'M0':
        raise ValueError('This bundle cannot assign a new primary role')
    if manifest['qualified_state_by_recipe'] != RECIPES:
        raise ValueError('Registered raw/qualified recipe mapping changed')
    for relative, sha in manifest['source_sha256'].items():
        if digest(ROOT / relative) != sha:
            raise RuntimeError('Research bundle runtime source changed: ' + relative)
    for relative, sha in manifest['files_sha256'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or digest(path) != sha:
            raise RuntimeError('Research bundle file identity changed')
    refs = [FeatureModel.load(directory / name) for name in ('J102.joblib', 'C205.joblib')]
    if any(m.training['training_ids_sha256'] != manifest['base_training_int32_sha256'] for m in refs):
        raise ValueError('Baseline model training membership mismatch')
    heads = {}
    for method in ('I4_EVENT_SCALE_TREE', 'G2_FULL', 'G3_PAIRWISE_FULL'):
        model = Correction.load(directory / (method + '.joblib'))
        if model.training['fold'] != manifest['fold'] or model.training['training_ids_sha256'] != manifest['training_ids_sha256']:
            raise ValueError('Correction and baseline training membership mismatch')
        if not model.training['strict_nested_OOF'] or not model.training['outer_validation_excluded']:
            raise ValueError('Strict nested correction provenance required')
        if method == 'I4_EVENT_SCALE_TREE' and model.method != method:
            raise ValueError('Wrong I4 head')
        if method != 'I4_EVENT_SCALE_TREE' and model.training.get('candidate') != method:
            raise ValueError('Wrong qualified-memory head')
        heads[method] = model
    calibration = read_json(directory / 'SCALE_CALIBRATION.json')
    return ResearchBundle(manifest, refs, heads, calibration)


class ResearchBundle:
    def __init__(self, manifest, references, heads, calibration):
        self.manifest = manifest
        self.references, self.heads, self.calibration = references, heads, calibration
        self.baseline = SharedResearchBaseline(*references)

    def prepare(self, method='M0', *, resident=True):
        if method == 'M0':
            return self.baseline
        if method not in RECIPES:
            raise ValueError('Only named frozen research recipes are available')
        return Prepared(*self.references, self.heads.get(method), self.calibration,
            qualified=RECIPES[method], resident=resident)

    def new_state(self, historical, method='M0', *, resident=True):
        return self.prepare(method, resident=resident).new_state(historical)
