"""Raw synthetic feature/state/schema replay before any live runtime audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main():
    violations=install_workspace_boundary()
    with target_free_npz(),activity('Independent synthetic raw-feature and complete state-conditioned input scalar replay','RUNTIME_QUALIFICATION','B9_NORMALIZATION_STATE_AVAILABILITY'):
        import numpy as np
        import pickle
        from src.next8.features import FeatureState
        from src.next9.streaming import PrefixBank
        from src.next9.replay_checks import ArrayBank
        from src.next9.joint_design import assemble
        from src.next9.transitions import SelfLaw
        candidates=[72,81,89,59,85,77,64,51,1];maps={j:np.arange(10,dtype=int) for j in candidates}
        rules={j:dict(pair=['S1_DIRECTION',2,0],low=-1.,high=1.) for j in candidates[:6]}
        law=SelfLaw('T1_SELF_MARKOV',np.array([[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.25]*4]),None,{})
        laws={j:law for j in rules};records=[]
        for H_length in [0,159,512,2048]:
            rng=np.random.default_rng(9101604+H_length);h=rng.normal(size=H_length).astype(np.float32);online=rng.normal(size=257).astype(np.float32)
            fs=FeatureState(h);seed=fs.history.copy();epsilon=fs.epsilon.copy();context=fs.context.copy();values=fs.replay(online)[0]
            batch=ArrayBank(seed,values,epsilon,context,H_length)
            expected,names=assemble(batch,np.arange(len(online)),candidates,rules,maps,laws,'M5')
            scalar=FeatureState(h);bank=PrefixBank(seed,epsilon,context,H_length);actual=[];scalars=[]
            for t,point in enumerate(online):
                value=scalar.update(float(point))[0];scalars.append(value.copy());bank.stage(value)
                x,nn=assemble(bank,np.array([t]),candidates,rules,maps,laws,'M5');assert nn==names;actual.append(x[0]);bank.commit()
                if t==73:scalar,bank=pickle.loads(pickle.dumps((scalar,bank),protocol=5))
            np.testing.assert_array_equal(np.asarray(scalars),values)
            np.testing.assert_array_equal(np.asarray(actual),expected)
            records.append(dict(H_length=H_length,points=len(online),input_columns=len(names),
                feature_scalar_batch_bitwise=True,complete_M5_numeric_input_bitwise=True,save_restore_at74=True,
                short_H_unknown=H_length<160,labels_or_tau_used=False))
        assert not violations
        files=['src/next9/streaming.py','src/next9/replay_checks.py','scripts/next9/qualify_next9_streaming.py']
        atomic(BASE/'tests/RAW_STREAMING_SCHEMA_QUALIFICATION_NEXT9.json',dict(status='PASS',timestamp=utc_now(),records=records,
            source_sha256={p:digest(ROOT/p) for p in files},includes_all_four_state_families=True,
            includes_current_F_interactions_and_previous_issued_self_law=True))


if __name__=='__main__':main()
