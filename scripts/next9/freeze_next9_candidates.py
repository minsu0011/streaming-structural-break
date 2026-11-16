"""Pool all target-free panels and freeze separate normal/strict-early lists."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,ROOT,atomic,read_json,digest,activity,utc_now,csv_write
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main():
    violations=install_workspace_boundary()
    lock=read_json(BASE/'CANDIDATE_FREEZE_IMPLEMENTATION_LOCK_NEXT9.json')
    for n,s in lock['source_sha256'].items():assert digest(ROOT/n)==s,n
    for i in range(12):assert read_json(BASE/'movement_screen'/f'PART_{i}_DONE.json')['status']=='COMPLETE'
    for i in range(4):assert read_json(BASE/'early_selection'/f'PART_{i}_DONE.json')['status']=='COMPLETE'
    with target_free_npz() as reads,activity('Pool all153 multihorizon evidence and freeze all20procedure candidate lists before labels','CANDIDATE_FREEZE','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
        import numpy as np
        from src.next9.data import FeatureBank,outer_scope,inner_scope
        from src.next9.movement_screen import paired_group_summary
        from src.next9.eligibility import holm,select,mode_pass
        bank=FeatureBank();freezes=[];summary=[];contrasts=[]
        for fold in range(5):
            for inner in [None,0,1,2]:
                parent=outer_scope(bank,fold) if inner is None else inner_scope(bank,fold,inner)
                path=BASE/'candidates'/parent.name
                if (path/'FROZEN.json').exists():
                    raise RuntimeError('Prior partial candidate freeze retained; explicit recovery required')
                records=[];p_family={};p_early={}
                for j in range(153):
                    record=dict(feature=j,panels={},pooled={})
                    for mode in ['NORMAL','EARLY_LATE','EARLY_SELECT']:
                        panels=[];losses=[];group_ids=[]
                        for split in range(3):
                            name=f'outer_{fold}_'+('main' if inner is None else f'inner_{inner}')+f'_selection_{split}_'+('NORMAL' if mode=='NORMAL' else 'EARLY')
                            directory=BASE/('early_selection' if mode=='EARLY_SELECT' else 'movement_screen')/f'F{j:03d}'/name
                            done=read_json(directory/'DONE.json')
                            assert digest(directory/'RESULTS.json')==done['files_sha256']['RESULTS.json']
                            assert digest(directory/'GROUP_LOSSES.npz')==done['files_sha256']['GROUP_LOSSES.npz']
                            result=read_json(directory/'RESULTS.json')
                            if mode=='EARLY_SELECT':assert result['validation_auxiliary_time_maximum']<=199
                            panels.append(result['records'])
                            with np.load(directory/'GROUP_LOSSES.npz') as z:
                                losses.append(z['loss']);group_ids.append(z['group_id'])
                        loss=np.concatenate(losses);groups=np.concatenate(group_ids)
                        assert len(np.unique(groups))==len(groups),'Original group crosses internalvalidationpanels'
                        pooled=[]
                        for k,h in enumerate([1,4,16]):
                            r=dict(horizon=h,groups=len(groups),D0A_MSE=float(loss[:,k,0].mean()),
                                   H_center_MSE=float(loss[:,k,1].mean()),D1_MSE=float(loss[:,k,2].mean()))
                            for column,label in [(0,'persistence_contrast'),(1,'H_center_contrast')]:
                                r[label]=paired_group_summary(loss[:,k,column],loss[:,k,2],groups)
                                key=f'{j}|{mode}|{h}|{label}'
                                (p_early if mode=='EARLY_SELECT' else p_family)[key]=r[label]['p_one_sided']
                            pooled.append(r)
                        record['panels'][mode]=panels;record['pooled'][mode]=pooled
                    records.append(record)
                assert len(p_family)==1836 and len(p_early)==918
                adjusted={**holm(p_family),**holm(p_early)}
                for record in records:
                    j=record['feature']
                    for mode,pooled in record['pooled'].items():
                        for r in pooled:
                            for label in ['persistence_contrast','H_center_contrast']:
                                r[label]['holm_adjusted_p']=adjusted[f'{j}|{mode}|{r["horizon"]}|{label}']
                                contrasts.append(dict(scope=parent.name,fold=fold,inner=inner,feature=j,mode=mode,
                                      horizon=r['horizon'],reference=label,D0A_MSE=r['D0A_MSE'],H_center_MSE=r['H_center_MSE'],D1_MSE=r['D1_MSE'],**r[label]))
                normal,total_normal=select(records,early_only=False)
                early,total_early=select(records,early_only=True)
                for r in records:
                    passes={m:mode_pass(r['panels'][m],r['pooled'][m]) for m in r['pooled']}
                    summary.append(dict(scope=parent.name,fold=fold,inner=inner,feature=r['feature'],
                          normal_supported=passes['NORMAL'],early_late_supported=passes['EARLY_LATE'],
                          strict_early_supported=passes['EARLY_SELECT'],
                          normal_eligible=passes['NORMAL'] and passes['EARLY_LATE'],
                          selected_normal=r['feature'] in normal,selected_early=r['feature'] in early,
                          eligibility_not_variable_orientation_certification=True))
                atomic(path/'ALL_FEATURE_EVIDENCE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=parent.provenance(),records=records))
                frozen=dict(status='FROZEN',timestamp=utc_now(),scope=parent.provenance(),fold=fold,inner=inner,
                      normal_candidates=normal,strict_early_candidates=early,normal_eligible_before_cap=total_normal,
                      strict_early_eligible_before_cap=total_early,detailed_cap=12,finalist_cap=6,
                      evidence_sha256=digest(path/'ALL_FEATURE_EVIDENCE.json'),
                      normal_Holm_family_size=1836,strict_early_Holm_family_size=918,
                      break_labels_tau_or_outer_scores_used=False,strict_early_candidates_use_no_late_validation_outcome=True,
                      original_normal_early_late_stress_not_redefined=True,
                      protocol_sha256=digest(BASE/'NEXT9_PROTOCOL_LOCK.json'),
                      early_supplement_sha256=digest(BASE/'EARLY_SELECTION_PROTOCOL_SUPPLEMENT_NEXT9.json'),
                      implementation_lock_sha256=digest(BASE/'CANDIDATE_FREEZE_IMPLEMENTATION_LOCK_NEXT9.json'))
                atomic(path/'FROZEN.json',frozen)
                freezes.append(dict(path=(path/'FROZEN.json').relative_to(BASE).as_posix(),sha256=digest(path/'FROZEN.json'),
                                    scope=parent.name,normal_candidates=normal,strict_early_candidates=early))
                print('FROZEN_TARGET_FREE_CANDIDATES',parent.name,normal,early,flush=True)
        csv_write('TARGET_FREE_MOVEMENT_SCREEN.csv',contrasts)
        csv_write('MOVEMENT_ELIGIBILITY.csv',summary)
        csv_write('H_CENTER_COMPARISON.csv',[r for r in contrasts if r['reference']=='H_center_contrast'])
        csv_write('EARLY_LATE_MOVEMENT_RESULTS.csv',[r for r in contrasts if r['mode'] in ('EARLY_LATE','EARLY_SELECT')])
        assert not violations
        atomic(BASE/'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),
              procedure_count=20,candidates=freezes,all153_all3horizons_retained=True,
              actual_ridge_estimator_fit_calls=18360,new_early_selection_fits=0,break_labels_tau_read=0,
              NPZ_members=sorted(set(reads)),boundary_violations=violations,
              label_phase_may_start_only_after_all20_complete=True,
              no_candidate_reranking_from_subsequent_break_orientation_or_outer_score=True,
              result_sha256={n:digest(BASE/n) for n in ['TARGET_FREE_MOVEMENT_SCREEN.csv','MOVEMENT_ELIGIBILITY.csv','H_CENTER_COMPARISON.csv','EARLY_LATE_MOVEMENT_RESULTS.csv']}))


if __name__=='__main__':main()
