"""Extract a finite set of unchanged numerical definitions before E003 locking."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import zlib
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, read_json, utc_now, progress

DEFINITIONS = {
    'src/next/fast_variance_initialization.py': ['conditional_replay_only', 'fast_variance_parameters'],
    'src/next2/pruned_runtime.py': ['pair_channel_update', 'historical_pair'],
    'src/next2/dependence_runtime.py': ['historical_z', 'evidence'],
    'src/next2/used_feature_runtime.py': ['dependencies', 'used_old_update', 'replay_old', 'UsedOldState', 'used_arch_step'],
    'src/next2/shared_blend_runtime.py': ['shared_step', 'SharedChannels', 'shared_feature_step'],
    'src/next3/packed_shared_runtime.py': ['packed_model_step', 'packed_kernel'],
    'src/next/fused_runtime.py': ['_resident_class', 'unique_array_bytes'],
}
HEADER = '''"""Generated frozen arithmetic for E003; see LEAN_NUMERIC_BUILD.json."""
from functools import lru_cache
import numpy as np
from numba import njit, typeof
from numba.experimental import jitclass
from src.features.config import CONFIG
from src.features.historical_calibration import CalibrationPolicy
from src.features.ar_order_input import historical_filter
from src.features.arch_input import historical_variance_filter, arch_one
from src.features.ar_residual_input import innovation_one
from src.features.streaming import StreamingFeatureState, feature_names
from src.next.whitening import fit_historical, innovation_update
from src.next.channel_evidence import make_channel_args, _logadd
from src.next.serial_channel_evidence import historical_correlations, sum_variances, serial_channel_step
from src.next2.complement_bank import normalized_one, historical_raw, raw_complement
from src.next3.packed_trees import packed_tree_margin

OLD_CONFIG = CONFIG.variant(normalization='median_mad', scales=(5,20,160))
OLD_POLICY = CalibrationPolicy(method='historical_median_mad')
'''


def main():
    if (BASE/'engineering/lean_shared/IMPLEMENTATION_LOCK.json').exists():
        raise RuntimeError('E003 implementation is already frozen')
    destination = ROOT/'src/next3/lean_numeric.py'
    if destination.exists(): raise RuntimeError('Preserve the first extracted source')
    pieces = [HEADER]; definitions = []
    for relative, selected in DEFINITIONS.items():
        path = ROOT/relative; source = path.read_text(encoding='utf-8'); lines = source.splitlines(keepends=True)
        nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for name in selected:
            node = nodes[name]; start = min([node.lineno]+[d.lineno for d in node.decorator_list])-1
            original = ''.join(lines[start:node.end_lineno]).rstrip()
            generated = original.replace('@njit(cache=True)', '@njit(cache=False)')
            pieces.append('\n\n# Extracted from '+relative+' :: '+name+'\n'+generated+'\n')
            definitions.append({'source': relative, 'name': name, 'original_source_sha256': digest(path),
                'original_definition_sha256': hashlib.sha256(original.encode()).hexdigest(),
                'generated_definition_sha256': hashlib.sha256(generated.encode()).hexdigest(),
                'change': 'disk-cache decorator only' if generated != original else 'none'})
    content = ''.join(pieces); ast.parse(content)
    destination.write_text(content, encoding='utf-8', newline='\n')
    reference = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2'
    blob = reference.read_bytes(); offset = len(b'SB_NEXT2_MODEL_V1\n')
    envelope = json.loads(zlib.decompress(blob[offset+40:])); record = envelope['model']
    metadata = {k: v for k, v in record.items() if k not in ('primary', 'complement')}
    for role in ('primary', 'complement'):
        metadata[role] = {k: v for k, v in record[role].items() if k not in ('tree', 'columns')}
    frozen = read_json(BASE/'preflight/FROZEN_REFERENCES.json')['protected_sha256']
    source_contract = {p: h for p, h in frozen.items() if p.startswith('src/') and p.endswith('.py')}
    source_contract['src/next3/packed_trees.py'] = digest(ROOT/'src/next3/packed_trees.py')
    atomic_json(ROOT/'configs/next3/LEAN_FEATURE_CONTRACT.json', {
        'format': 'NEXT3_LEAN_FROZEN_80_20_V1', 'model_metadata': metadata,
        'scientific_sources_sha256': source_contract,
        'reference_model_sha256': digest(reference),
        'reference_is_metadata_only': 'Trees and selected columns are decoded from each supplied canonical model, never copied here',
    }, immutable=True)
    atomic_json(ROOT/'configs/next3/LEAN_NUMERIC_BUILD.json', {
        'created_utc': utc_now(), 'generator_sha256': digest(Path(__file__)),
        'generated_source_sha256': digest(destination), 'definitions': definitions,
        'permitted_transformation': 'Exact source extraction; cache=True decorator becomes cache=False only',
    }, immutable=True)
    progress('E003 numerical source extraction completed', 'No new candidate inference or cost measurements yet')
    print('EXTRACTED', len(definitions), 'definitions; bound', len(source_contract), 'scientific sources', flush=True)


if __name__ == '__main__': main()
