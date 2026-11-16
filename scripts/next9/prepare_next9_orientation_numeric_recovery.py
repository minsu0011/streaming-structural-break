"""Recover a frozen panel after float32 upper-bound clipping overflow."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now

def main():
    old=ROOT/'src/next9/inference.py'
    source=old.read_text(encoding='utf-8')
    source=source.replace('prob = np.clip(p,1e-8,1-1e-8)', '''prob = np.clip(p,1e-8,1-1e-8)
        # Preserve previously finite float32 computations exactly. Only the
        # unrepresentable upper bound requires float64 evaluation.
        if np.any(prob >= 1):
            prob = np.clip(np.asarray(p,dtype=np.float64),1e-8,1-1e-8)''')
    atomic(ROOT/'src/next9/inference_numeric_recovery.py',source)
    script=(ROOT/'scripts/next9/select_next9_orientation.py').read_text(encoding='utf-8')
    script=script.replace('from src.next9.inference import compare','from src.next9.inference_numeric_recovery import compare')
    script=script.replace("if out.exists() and any(out.iterdir()):", "if out.exists() and any(out.iterdir()) and not (out/'PREDICTIONS_FROZEN.json').exists():")
    start=script.index("                    for method in ['O0'")
    end=script.index("                    vy=labels.get",start)
    block=script[start:end]
    replacement="""                    if (out/'PREDICTIONS_FROZEN.json').exists():
                        frozen=read_json(out/'PREDICTIONS_FROZEN.json')
                        assert frozen['scope']==scope.provenance()
                        assert frozen['training_domains']==[d.record() for d in domains]
                        files=dict(frozen['files_sha256'])
                        for name,sha in files.items(): assert digest(out/name)==sha
                        with np.load(out/'PREDICTIONS.npz') as z:
                            np.testing.assert_array_equal(z['global_row'],rows[va])
                            predictions={k:z[k] for k in z.files if k!='global_row'}
                        models={Path(k).stem:joblib.load(out/k) for k in files if k.endswith('.joblib')}
                    else:
"""+''.join('    '+line+'\n' for line in block.splitlines())
    script=script[:start]+replacement+script[end:]
    script=script.replace("        atomic(root/mode/'SELECTION_FROZEN.json',mode_record)","""        if (root/mode/'SELECTION_FROZEN.json').exists():
            assert read_json(root/mode/'SELECTION_FROZEN.json')==mode_record
        else:
            atomic(root/mode/'SELECTION_FROZEN.json',mode_record)""")
    path='scripts/next9/select_next9_orientation_numeric_recovery.py'
    compile(script,path,'exec');atomic(ROOT/path,script)
    atomic(BASE/'repairs/ORIENTATION_FLOAT32_LOGLOSS_RECOVERY_REGISTERED.json',dict(
        status='FROZEN',timestamp=utc_now(),failed_task='ORIENTATION_V2_outer_4_inner_0',
        cause='float32 1-1e-8 rounds to1, causing infinite logloss; JSON rejected it',
        scope='Reuse hash-verified frozen models and predictions; preserve all completed finite panels bytewise',
        numerical_policy='Use float64 clipping for probability vectors whose float32 clipped upper bound reaches1; finite prior calculations unchanged',
        no_threshold_or_fit_recipe_change=True,failed_artifacts_preserved=True,
        original_inference_sha256=digest(old),source_sha256={p:digest(ROOT/p) for p in [path,'src/next9/inference_numeric_recovery.py']}))

if __name__=='__main__':main()
