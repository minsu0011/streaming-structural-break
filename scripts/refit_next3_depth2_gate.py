"""Final gate training uses honest base OOF; no in-sample quality score."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.learned_router import fit_gate
from src.next3.gate_serialization import describe_gate,descriptor_predict
from src.next.data import prepare_dev


def main():
    kind='R3_DEPTH2_GATE';hyp=BASE/'hypotheses/R3_DEPTH2_GATE_SYNTHETIC.json'
    registration=read_json(hyp)
    for name,sha in registration['sources_sha256'].items():assert digest(ROOT/name)==sha
    out=BASE/'refits'/kind/'research_v1'
    if out.exists():raise RuntimeError('Preserve prior final-gate fit')
    out.mkdir(parents=True)
    data=prepare_dev(ROOT);metadata=data.rows();features=np.full((len(metadata['target']),6),np.nan,dtype=np.float32)
    filled=np.zeros(len(features),dtype=bool);sources={}
    for fold in range(5):
        source=BASE/'nested/full'/f'outer_{fold}'
        receipt=read_json(source/'RESULTS.json')
        assert receipt['outer_validation_never_used_for_inner_training']
        for name in ('GATE_VALID.npy','VALID_GLOBAL_ROWS.npy'):
            assert digest(source/name)==receipt['files_sha256'][name]
            sources[(source/name).relative_to(ROOT).as_posix()]=digest(source/name)
        rows=np.load(source/'VALID_GLOBAL_ROWS.npy',allow_pickle=False)
        assert not filled[rows].any() and np.all(metadata['fold'][rows]==fold)
        features[rows]=np.load(source/'GATE_VALID.npy',allow_pickle=False);filled[rows]=True
    assert filled.all() and np.isfinite(features).all()
    admitted=data.guard.admit(np.unique(metadata['dataset_id']),purpose='NEXT3 final gate honest base OOF training')
    assert set(admitted)==set(data.guard.dev_ids)
    atomic_json(out/'FIT_START.json',{'created_utc':utc_now(),'intended_gate_fits':1,'base_fits':0,'hypothesis_sha256':digest(hyp)},immutable=True)
    progress('Final depth2 gate fitting','All DEV honest base OOF; frozen full-DEV J/C reused only for inference',experiment=kind)
    gate=fit_gate(kind,features,metadata);descriptor=describe_gate(gate)
    atomic_json(out/'gate.json',descriptor,immutable=True)
    np.testing.assert_array_equal(descriptor_predict(read_json(out/'gate.json'),features),gate.predict(features))
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'new_gate_fits':1,'new_base_fits':0,
        'training_series':len(admitted),'training_rows':len(features),'base_inputs_honest_OOF':True,
        'quality_score_on_refit_rows':None,'seal_rows':0,'not_promoted':True,
        'hypothesis_sha256':digest(hyp),'sources_sha256':sources,'files_sha256':{'gate.json':digest(out/'gate.json')},
        'training_detail':gate.training_detail},immutable=True)
    print('NEXT3 DEPTH2 FINAL GATE FIT PASS',gate.training_detail,flush=True)


if __name__=='__main__':main()
