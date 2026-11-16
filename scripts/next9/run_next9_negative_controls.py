from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(part):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'NEGATIVE_CONTROL_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items(): assert digest(ROOT/name)==sha,name
    with target_free_npz() as accesses:
        import pickle
        import numpy as np
        from src.next8.features import FeatureState
        from src.next9.negative_controls import BACKGROUNDS,INTERVENTIONS,SEEDS,H_LENGTHS,generate
        from src.next9.states import MovementState,describe_history
        from src.next9.io import save_npz
        frozen=read_json(BASE/'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json')
        selected=sorted(set(j for r in frozen['candidates'] for j in r['normal_candidates']+r['strict_early_candidates']))
        background=BACKGROUNDS[part];receipts=[]
        for h_length in H_LENGTHS:
            for seed in SEEDS:
                for intervention in INTERVENTIONS:
                    case=f'{background}_{h_length}_{seed}_{intervention}'
                    out=BASE/'negative_controls'/case
                    if (out/'DONE.json').exists():
                        receipts.append(dict(case=case,sha256=digest(out/'DONE.json')));continue
                    with activity('Synthetic paired raw observation/control '+case,'NEGATIVE_CONTROLS','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                        h,online,meta=generate(background,intervention,seed,h_length)
                        engine=FeatureState(h);initial=pickle.dumps(engine)
                        seeds=engine.history.copy();epsilon=np.asarray(engine.epsilon[0],dtype=np.float32)
                        values,_=engine.replay(online)
                        scalar=pickle.loads(initial)
                        scalar_values=np.stack([scalar.update(float(v))[0] for v in online[:32]])
                        np.testing.assert_array_equal(scalar_values,values[:32])
                        restored=pickle.loads(initial)
                        first,_=restored.replay(online[:193]);restored=pickle.loads(pickle.dumps(restored))
                        latter,_=restored.replay(online[193:])
                        np.testing.assert_array_equal(np.concatenate([first,latter]),values)
                        changed=online.copy();changed[256:]=np.float32(17.)
                        prefix,_=pickle.loads(initial).replay(changed)
                        np.testing.assert_array_equal(prefix[:256],values[:256])
                        extended=np.concatenate([seeds,values])
                        states={name:np.empty((512,153),dtype=np.uint8) for name in ['S1_DIRECTION','S2_DIRECTION_RUN','S3_DIRECTION_SLOPE','S4_REVERSAL']}
                        for t in range(512):
                            before=describe_history(extended[t:t+16].T,epsilon,True)
                            for name in states:states[name][t]=before[name]
                        scalar_comparisons=0
                        for j in selected:
                            movement=MovementState(seeds[:,j],float(epsilon[j]))
                            for t,f in enumerate(values[:,j]):
                                before=movement.peek()
                                for name in states:
                                    assert before[name]==states[name][t,j]
                                scalar_comparisons+=1
                                movement.observe(float(f))
                                if t==192:movement=pickle.loads(pickle.dumps(movement))
                        save_npz(out/'SYNTHETIC_TRAJECTORY.npz',values=values,**states)
                        counts={str(j):np.bincount(states['S1_DIRECTION'][:,j],minlength=4).tolist() for j in range(153)}
                        done=dict(status='PASS',timestamp=utc_now(),case=meta,raw_scalar_batch_max_error=0.,
                            raw_save_restore_max_error=0.,raw_future_suffix_prefix_max_error=0.,
                            movement_scalar_vector_comparisons=scalar_comparisons,selected_feature_ids=selected,
                            all153_movement_state_counts=counts,synthetic_trajectory_sha256=digest(out/'SYNTHETIC_TRAJECTORY.npz'),
                            synthetic_case_not_an_independent_real_series=True,RP_target_or_score_used=False,
                            known_generator_onset_not_passed_to_feature_or_state_API=True,
                            implementation_sha256=digest(BASE/'NEGATIVE_CONTROL_IMPLEMENTATION_LOCK_NEXT9.json'))
                        atomic(out/'DONE.json',done);receipts.append(dict(case=case,sha256=digest(out/'DONE.json')))
        assert not violations
        atomic(BASE/'negative_controls'/f'PART_{part}_DONE.json',dict(status='PASS',timestamp=utc_now(),background=background,
            cases=receipts,no_real_labels_tau_used=True,NPZ_members=sorted(set(accesses))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',required=True,type=int,choices=range(4));main(p.parse_args().part)
