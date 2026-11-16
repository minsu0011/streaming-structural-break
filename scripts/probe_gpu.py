"""Confirm CUDA training availability with a small synthetic hardware probe."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,ledger

def main():
    import xgboost as xgb
    data=np.random.default_rng(20260908).normal(size=(4096,12)).astype(np.float32)
    model=xgb.XGBClassifier(n_estimators=8,max_depth=2,device='cuda',tree_method='hist',n_jobs=4,random_state=20260908)
    try:
        model.fit(data,(data[:,0]>0).astype(np.uint8))
        config=json.loads(model.get_booster().save_config())
        device=config['learner']['generic_param']['device']
        report={'status':'PASS' if device.startswith('cuda') else 'CPU_FALLBACK','actual_device':device}
    except Exception as exc:
        report={'status':'CPU_FALLBACK','exception_type':type(exc).__name__}
    report.update({'xgboost':xgb.__version__,'scope':'synthetic hardware probe only','competition_training_performed':False})
    write_json(ROOT/'artifacts/audits/GPU_SMOKE.json',report)
    ledger(experiment_id='hardware_cuda_probe',model='xgboost',status=report['status'],notes='Synthetic hardware validation only')
    print(report)

if __name__=='__main__':main()
