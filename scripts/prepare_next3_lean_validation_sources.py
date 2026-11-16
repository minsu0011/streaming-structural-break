"""Derive E003 parity harnesses from reviewed E002 harnesses before measurements."""
from pathlib import Path
import ast
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress


def transformed(name, output, substitutions):
    text = (ROOT/name).read_text(encoding='utf-8')
    for before, after in substitutions:
        if before not in text: raise RuntimeError('Missing expected source segment: '+before)
        text = text.replace(before, after)
    ast.parse(text)
    target = ROOT/output
    if target.exists(): raise RuntimeError('Preserve earlier generated validation source')
    target.write_text(text, encoding='utf-8', newline='\n')
    return {'original': name, 'original_sha256': digest(ROOT/name), 'generated': output, 'generated_sha256': digest(target)}


def main():
    if (BASE/'engineering/lean_shared/IMPLEMENTATION_LOCK.json').exists(): raise RuntimeError('E003 frozen')
    paths = ['configs/features.json', 'configs/arch_input.json', 'configs/ar_residual_input.json',
        'configs/historical_context.json', 'configs/historical_quantiles.json', 'configs/rank_gaussian.json',
        'configs/magnitude_dependence.json', 'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json']
    atomic_json(ROOT/'configs/next3/LEAN_RUNTIME_INPUTS.json', {
        'files_sha256': {p: digest(ROOT/p) for p in paths},
        'scope': 'Frozen configuration bytes required by pure numeric inference and its training recipe'}, immutable=True)
    adapter = '''from src.next3.lean_shared_runtime import PreparedLeanSharedPredictor, source_hash, accounting
from src.next3.lean_model import loads as lean_loads, feature_contract
VARIANTS = ('LEAN_PACKED_SHARED',)


def PreparedPackedSharedPredictor(model, *, variant):
    if variant != 'LEAN_PACKED_SHARED': raise ValueError('One fixed E003 variant')
    contract = feature_contract()
    return PreparedLeanSharedPredictor(lean_loads(dumps(model), contract=contract), contract=contract)
'''
    reports = [transformed('scripts/verify_next3_packed_shared.py', 'scripts/verify_next3_lean_shared.py', [
        ('from src.next3.packed_shared_runtime import PreparedPackedSharedPredictor, VARIANTS, source_hash, accounting', adapter),
        ('engineering/packed_shared', 'engineering/lean_shared'), ('E002_PACKED_SHARED', 'E003_LEAN_SHARED'),
        ('state.base.output', 'state.output'), ('state.base.shared.dependence_output', 'state.shared.dependence_output'),
        ('both locked packed variants', 'the locked lean variant'), ('both fixed variants', 'one fixed lean variant'),
        ('NEXT3 PACKED', 'NEXT3 LEAN'),
    ])]
    reports.append(transformed('scripts/verify_next3_packed_public_api.py', 'scripts/verify_next3_lean_public_api.py', [
        ('from src.next3.packed_shared_runtime import accounting', 'from src.next3.lean_shared_runtime import accounting'),
        ('PERFORMANCE_packed_shared_v1', 'PERFORMANCE_lean_shared_v1'),
        ('PACKED_RUNTIME_SELECTION.json', 'LEAN_RUNTIME_SELECTION.json'),
        ("assert selection['status'] == 'PROVISIONAL_ENGINEERING_ELIGIBLE'", "assert selection['status'] in ('PROVISIONAL_ENGINEERING_ELIGIBLE', 'VALIDATION_ONLY')"),
        ('packed_public_api_all_dev', 'lean_public_api_all_dev'),
        ('next3_packed_performance_main', 'next3_lean_performance_main'),
        ('E002_PACKED_SHARED', 'E003_LEAN_SHARED'), ('NEXT3 PACKED', 'NEXT3 LEAN'),
    ]))
    atomic_json(BASE/'engineering/lean_shared/VALIDATION_SOURCE_DERIVATION.json', {
        'created_utc': utc_now(), 'generator_sha256': digest(Path(__file__)), 'sources': reports,
        'scope': 'Harness adaptation only; canonical frozen arithmetic and expected OOF predictions unchanged'}, immutable=True)
    progress('E003 validation harnesses prepared', 'Classless decoder and exact arithmetic remain unmeasured until implementation lock')


if __name__ == '__main__': main()
