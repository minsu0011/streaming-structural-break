"""All-point actual frozen self-law replay and diagnostic full-cadence losses."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(fold,mode,part,qualify=False):
    denied=install_workspace_boundary()
    import numpy as np
    import joblib
    from src.next9.data import FeatureBank,outer_scope
    from src.next9.self_cadence import replay_series,VARIANTS
    from src.next9.io import save_npz
    with target_free_npz() as accesses:
        bank=FeatureBank();scope=outer_scope(bank,fold)
        selection=read_json(BASE/'self_transitions'/scope.name/mode/'DONE.json')
        assert selection['scope']==scope.provenance()
        laws={};chosen={};source_hashes={}
        for record in selection['records']:
            j=record['feature'];chosen[j]=record['chosen_variant']
            directory=BASE/'outer_self_transition_evaluation'/scope.name/mode/f'F{j:03d}'
            frozen=read_json(directory/'PREDICTIONS_FROZEN.json')
            assert frozen['scope']==scope.provenance() and frozen['chosen_variant']==chosen[j]
            laws[j]={}
            for variant in VARIANTS:
                p=directory/(variant+'.joblib');assert digest(p)==frozen['files_sha256'][p.name]
                laws[j][variant]=joblib.load(p);source_hashes[str(p.relative_to(BASE))]=digest(p)
                assert laws[j][variant].training['training_ids_sha256']==scope.provenance()['training_ids_sha256']
        selected=np.flatnonzero(np.isin(bank.ids,list(scope.validation_ids)))[part::2]
        if qualify:
            # Fixed covariate criteria, not any prediction outcome: first short
            # and first >=512-point series in original order.
            lengths=bank.offsets[selected+1]-bank.offsets[selected]
            a=selected[np.flatnonzero(lengths<200)[0]]
            b=selected[np.flatnonzero(lengths>=512)[0]]
            selected=np.asarray([a,b])
        root=BASE/('self_cadence_qualification' if qualify else 'self_cadence')/scope.name/mode/f'part_{part}'
        if (root/'DONE.json').exists():raise FileExistsError('Preserve completed self cadence evidence')
        records=[];batch_receipts=[]
        for offset in range(0,len(selected),8):
            out=root/f'b{offset//8:04d}'
            if (out/'DONE.json').exists():
                r=read_json(out/'DONE.json');assert digest(out/'GROUP_LOSSES.npz')==r['group_losses_sha256']
                records+=r['records'];batch_receipts.append(str((out/'DONE.json').relative_to(BASE)));continue
            with activity(f'{scope.name} {mode} full-cadence selected self-law actual replay part{part} batch{offset//8}',
                          'SELF_LAW_FULL_CADENCE','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                batch=[];losses=[];ids=[]
                for s in selected[offset:offset+8]:
                    sid=int(bank.ids[s]);scope.admit_validation([sid]);assert bank.guard.split.groups[sid]==sid
                    lo,hi=map(int,bank.offsets[s:s+2])
                    loss,r=replay_series(bank.series['seed_values'][s],np.asarray(bank.values[lo:hi]),
                        bank.series['epsilon'][s],bank.series['context'][s],int(bank.series['H_length'][s]),laws,chosen,mode)
                    r.update(dataset_id=sid,fold=fold,mode=mode);batch.append(r);losses.append(loss);ids.append(sid)
                save_npz(out/'GROUP_LOSSES.npz',group_id=np.asarray(ids),loss=np.asarray(losses),features=np.asarray(sorted(laws)))
                receipt=dict(status='PASS',timestamp=utc_now(),records=batch,group_losses_sha256=digest(out/'GROUP_LOSSES.npz'))
                atomic(out/'DONE.json',receipt);records+=batch;batch_receipts.append(str((out/'DONE.json').relative_to(BASE)))
        assert not denied
        atomic(root/'DONE.json',dict(status='PASS',timestamp=utc_now(),fold=fold,mode=mode,part=part,scope=scope.provenance(),
            qualification=qualify,features=sorted(laws),variants=list(VARIANTS),chosen=chosen,
            series_ids=[r['dataset_id'] for r in records],series=len(records),points=sum(r['points'] for r in records),
            maximum_probability_error=max(r['maximum_probability_error'] for r in records),
            maximum_path_error=max(r['maximum_path_error'] for r in records),
            all_states_and_momentum_bitwise=True,save_restore_all_series=True,current_feature_not_exposed_at_issuance=True,
            model_files_sha256=source_hashes,batch_receipts_sha256={p:digest(BASE/p) for p in batch_receipts},
            no_break_labels_or_fits=True,does_not_activate_B_or_M5_branches=True,NPZ_members=sorted(set(accesses))))
        if qualify:
            atomic(BASE/'SELF_CADENCE_QUALIFICATION_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
                receipt_sha256=digest(root/'DONE.json'),source_sha256={p:digest(ROOT/p) for p in
                    ['src/next9/self_cadence.py','scripts/next9/replay_next9_self_cadence.py']},
                phase0_and_all_point_same_group_diagnostic_only=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5));p.add_argument('--mode',required=True,choices=['NORMAL','EARLY'])
    p.add_argument('--part',type=int,required=True,choices=[0,1]);p.add_argument('--qualify',action='store_true')
    a=p.parse_args();main(a.fold,a.mode,a.part,a.qualify)
