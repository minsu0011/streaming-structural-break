"""One global closure before any NEXT9 outer break score is released."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main():
    violations=install_workspace_boundary()
    lock=read_json(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    if (BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json').exists():raise RuntimeError('Outer predictions already globally frozen')
    with target_free_npz() as accesses,activity('Globally freeze every M0 toM7 and matched tabular probability before outerlabels','PREDICTION_FREEZE','B6_BL80_REDUNDANCY'):
        import numpy as np
        from src.next9.data import FeatureBank,outer_scope
        from src.next9.domains import fit_domains
        from src.next9.io import save_npz
        bank=FeatureBank();records=[];sources={}
        for fold in range(5):
            scope=outer_scope(bank,fold);domain=fit_domains(bank,scope)
            for stage in ['outer_orientation_refits','outer_movement_refits','outer_cross_controls']:
                p=BASE/stage/scope.name/'DONE.json';r=read_json(p)
                assert r['status']=='COMPLETE' and r['scope']==scope.provenance()
                sources[p.relative_to(BASE).as_posix()]=digest(p)
            for mode in ['NORMAL','EARLY']:
                hd=BASE/'joint_heads'/scope.name/mode;cd=BASE/'cross_and_tabular'/scope.name/mode;md=BASE/'meta_correction'/scope.name/mode
                h,c,m=[read_json(p/'DONE.json') for p in [hd,cd,md]]
                assert h['scope']==c['scope']==m['scope']==scope.provenance()
                for directory,done in [(hd,h),(cd,c)]:
                    for name,sha in done['files_sha256'].items():assert digest(directory/name)==sha
                assert digest(md/'PREDICTIONS.npz')==m['prediction_sha256']
                with np.load(md/'PREDICTIONS.npz') as z:rows=z['global_row'];ids=z['dataset_id'];pred={k:z[k] for k in ['M0','M7']}
                np.testing.assert_array_equal(rows,bank.sampled_rows(scope.validation_ids,start=400 if mode=='EARLY' else 0,stride=1))
                np.testing.assert_array_equal(ids,bank.row_ids[rows])
                scope.admit_validation(ids)
                with np.load(hd/'PREDICTIONS.npz') as z:
                    hr=z['global_row'];index=np.searchsorted(hr,rows);np.testing.assert_array_equal(hr[index],rows)
                    pred.update({k:z[k][index] for k in ['M1','M2','M3','M4','M5']})
                with np.load(cd/'PREDICTIONS.npz') as z:
                    np.testing.assert_array_equal(z['global_row'],rows)
                    pred.update({k:z[k] for k in ['M6','TABULAR_SAME_INPUT']})
                for name,p in pred.items():
                    assert p.shape==rows.shape and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),name
                if m['M7_alias']=='M0':np.testing.assert_array_equal(pred['M7'],pred['M0'])
                if c['M6_alias']=='M5':np.testing.assert_array_equal(pred['M6'],pred['M5'])
                out=BASE/'frozen_predictions'/scope.name/mode
                save_npz(out/'PREDICTIONS.npz',global_row=rows,dataset_id=ids,**pred)
                record=dict(status='FROZEN',timestamp=utc_now(),fold=fold,mode=mode,scope=scope.provenance(),rows=len(rows),
                    prediction_sha256=digest(out/'PREDICTIONS.npz'),domain_spec=domain,
                    head_aliases=h['aliases'],M6_alias=c['M6_alias'],M7_alias=m['M7_alias'],M7_qualified=m['eligible'],
                    outer_labels_used=False,methods=list(pred),original_outer_full_point_coverage=True,
                    source_receipts_sha256={str(p.relative_to(BASE))+'/DONE.json':digest(p/'DONE.json') for p in [hd,cd,md]})
                atomic(out/'FROZEN.json',record)
                records.append(dict(fold=fold,mode=mode,path=(out/'FROZEN.json').relative_to(BASE).as_posix(),sha256=digest(out/'FROZEN.json')))
                sources.update(record['source_receipts_sha256'])
        assert not violations
        atomic(BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),records=records,
            source_receipts_sha256=sources,all_five_original_folds_and_both_modes=True,all_M0_to_M7_and_matched_tabular=True,
            candidate_freeze_sha256=digest(BASE/'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json'),
            implementation_sha256=digest(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_NEXT9.json'),
            outer_label_quality_read=False,NPZ_members=sorted(set(accesses)),no_future_selection_or_parameter_changes=True))
        atomic(BASE/'OUTER_RELEASE_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
            semantic_state='OUTER_EVALUATION_AUTHORIZED_AFTER_GLOBAL_FREEZE',
            global_prediction_freeze_sha256=digest(BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json'),
            all_A_B_C_predictors_frozen=True,outer_score_values_read_at_authorization=False))


if __name__=='__main__':main()
