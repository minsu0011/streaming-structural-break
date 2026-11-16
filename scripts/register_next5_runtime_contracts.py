from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,digest,utc_now,progress
from src.next5.scheduler import register_job


def main():
    atomic_json(BASE/'runtime_validation/CONTRACT_POLICY.json',{'created_utc':utc_now(),
        'runtime_source_sha256':digest(ROOT/'src/next5/runtime.py'),'test_source_sha256':digest(ROOT/'tests/test_next5_runtime.py'),
        'method':'Existing shared J102/C205 recurrences with float64 probabilities; frozen candidate kernels and trees in a resident scalar adapter.',
        'feature_parity':'Exact float32 feature coordinates; probability tolerance 1e-15 for expit implementation roundoff.',
        'recipe_changes':False,'new_fits':0,'public_deployment':False},immutable=True)
    register_job({'id':'RT000_ADAPTER_CONTRACTS','kind':'validation','priority':35,
        'argv':['-m','pytest','-q','tests/test_next5_runtime.py','--junitxml=artifacts/next5/runtime_validation/CONTRACTS.xml'],
        'description':'All 20 frozen candidates: scalar feature parity, matched float64 shared baseline, exact prefix, bounded state and invalid current-point rejection',
        'retry_safe':False})
    progress('Resident runtime adapter contracts registered','Existing fitted recipes unchanged; verify exact coordinates and float64 composite parity before public-stream use')


if __name__=='__main__':main()
