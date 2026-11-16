"""All-five-fold model results, with remaining scientific/runtime gates explicit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now


with activity('Read complete five-fold frozen M comparison before separate A B runtime completion','MODEL_RESULT_INTERPRETATION','B2_NO_REPLICATED_OPPOSITE_ORIENTATION'):
    import numpy as np
    from src.next9.science_summary import aggregate_contrast
    output={};sources={}
    for mode in ['NORMAL','EARLY']:
        records=[];samples=[]
        for fold in range(5):
            root=BASE/'outer_model_evaluation'/f'outer_{fold}'
            assert read_json(root/'DONE.json')['status']=='COMPLETE'
            directory=root/mode/'ALL';r=read_json(directory/'DONE.json')
            assert digest(directory/'BOOTSTRAP.npz')==r['bootstrap_sha256']
            with np.load(directory/'BOOTSTRAP.npz') as z:samples.append({k:z[k] for k in z.files})
            records.append(r);sources[(directory/'DONE.json').relative_to(BASE).as_posix()]=digest(directory/'DONE.json')
        means={method:float(np.mean([r['points'][method]['TS_AUC'] for r in records])) for method in records[0]['points']}
        output[mode]=dict(mean_TS_AUC=means,contrasts={a+'__vs__'+b:aggregate_contrast(records,samples,a,b)
            for a,b in [('M7','M0'),('M3','M2'),('M6','TABULAR_SAME_INPUT')]})
    atomic(BASE/'MODEL_ONLY_FIVE_FOLDS_NEXT9.json',dict(status='COMPLETE',timestamp=utc_now(),results=output,source_sha256=sources,
        all_other_science_runtime_and_final_audit_gates_pending=True,not_promotion_authorization=True,
        five_original_folds_equal_weight=True,conditional_group_bootstrap_not_hidden_confirmation=True))
    print(output,flush=True)
