"""One outer-score release after every G1 prediction has been frozen."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.next7.common import BASE,save,read_json,digest,csv_write,utc_now,activity,require_guard,note
from src.next6.data import source
from src.scoring.ts_auc import ts_auc


def main():
    guard=require_guard()
    frozen=read_json(BASE/'G1_ALL_OUTER_PREDICTIONS_FROZEN.json')
    assert frozen['status']=='FROZEN' and frozen['all_five_folds_completed_before_outer_score_release']
    reservation=read_json(BASE/'G1_CANDIDATE_RESERVATION.json')
    if not (BASE/'G1_BOOTSTRAP_SELECTION_POLICY.json').exists():
        save('G1_BOOTSTRAP_SELECTION_POLICY.json',{'status':'FROZEN','created_utc':utc_now(),
            'before_outer_score_release':not (BASE/'G1_OUTER_SCORE_RELEASE.json').exists(),
            'finalist_rule':'Top3 non-standalone BL80 integration candidates by mean matched delta; add M_EVENT and M_CURRENT; fill to5 by remaining integration mean delta order, ties lexical',
            'max_finalists':5,'standalone_engine_scores':'diagnostic only, already counted in full scientific budget',
            'common_paired_bootstrap_draws':'5000 if any selected candidate mean delta>=.0025, otherwise2000; choose before drawing',
            'uncertainty':'conditional on these fitted OOF predictions; adaptive selection not corrected'})
    if (BASE/'G1_OUTER_SCORES.json').exists():
        raise RuntimeError('Preserve already released G1 scores')
    for record in frozen['folds']:
        directory=BASE/f'g1/fold_{record["fold"]}'
        for name,sha in record['prediction_sha256'].items():
            assert digest(directory/name)==sha
    save('G1_OUTER_SCORE_RELEASE.json',{'status':'FROZEN_PREDICTIONS_RELEASED','created_utc':utc_now(),
        'prediction_freeze_sha256':digest(BASE/'G1_ALL_OUTER_PREDICTIONS_FROZEN.json'),
        'bootstrap_policy_sha256':digest(BASE/'G1_BOOTSTRAP_SELECTION_POLICY.json'),
        'no_G1_refit_or_threshold_or_weight_changes_after_release':True,
        'adaptive_new_generation_requires_separate_preregistration_and_budget':True})
    with activity('G1 single outer score release','SCIENTIFIC_EVALUATION','F5_CAUSAL_NESTED_INTEGRATION'):
        data=source()
        metadata=data.rows()
        results=[]
        scores={}
        for fold in range(5):
            directory=BASE/f'g1/fold_{fold}'
            files=[directory/'fixed_predictions.npz']+[directory/m/'predictions.npz' for m in ('I3_CURRENT_EVENT','M_FULL','I4_EVENT_SCALE_TREE')]
            for path in files:
                with np.load(path,allow_pickle=False) as values:
                    rows=values['global_row']
                    guard.admit(np.unique(values['dataset_id']),purpose='G1 outer score admission')
                    np.testing.assert_array_equal(metadata['dataset_id'][rows],values['dataset_id'])
                    assert set(int(guard.split.assignment(int(i))[1]) for i in np.unique(values['dataset_id']))=={fold}
                    methods=[n for n in values.files if n not in ('global_row','dataset_id')]
                    for key in methods:
                        method=path.parent.name if key=='prediction' else key
                        value=float(ts_auc(metadata['target'][rows],values[key],metadata['time_online'][rows]))
                        scores.setdefault(method,{})[str(fold)]=value
                        results.append({'method':method,'fold':fold,'TS_AUC':value,
                            'series':len(np.unique(values['dataset_id'])),'rows':len(rows),
                            'strict_nested_correction':method in ('I3_CURRENT_EVENT','M_FULL','I4_EVENT_SCALE_TREE'),
                            'no_outer_label_fit':True})
            print(utc_now(),'G1 OUTER SCORED',fold,flush=True)
        old=read_json(ROOT/'artifacts/next6/OUTER_SCORES.json')['M0']
        baseline=np.array([scores['M0'][str(f)] for f in range(5)])
        np.testing.assert_allclose(baseline,old,atol=1e-12,rtol=0)
        summaries=[]
        for method,values in scores.items():
            s=np.array([values[str(f)] for f in range(5)])
            delta=s-baseline
            summaries.append({'method':method,'folds':5,'mean_TS_AUC':float(s.mean()),
                'median_TS_AUC':float(np.median(s)),'worst_TS_AUC':float(s.min()),
                'mean_delta_M0':float(delta.mean()),'median_delta_M0':float(np.median(delta)),
                'worst_fold_delta_M0':float(delta.min()),'positive_matched_folds':int((delta>0).sum()),
                'standalone_diagnostic':method in ('CURRENT_ONLY','EVENT_ONLY','SCALE_ONLY'),
                'conditional_DEVELOPMENT_only':True,'promotion_not_decided_by_mean_alone':True})
        for row in results:
            row['delta_M0']=row['TS_AUC']-float(baseline[row['fold']])
        csv_write('CV_RESULTS_NEXT7_G1.csv',results)
        csv_write('MODEL_COMPARISON_NEXT7_G1.csv',summaries)
        save('G1_OUTER_SCORES.json',{'created_utc':utc_now(),'scores':scores,'summaries':summaries,
            'M0_exact_frozen_RP_baseline_parity':'PASS','new_full_scientific_candidates':12})
        ranking=sorted([r for r in summaries if r['method'] not in ('M0','AMPLITUDE_0.9_M0','CURRENT_ONLY','EVENT_ONLY','SCALE_ONLY')],
            key=lambda r:(-r['mean_delta_M0'],r['method']))
        selected=[r['method'] for r in ranking[:3]]
        for method in ('M_EVENT','M_CURRENT'):
            if method not in selected:
                selected.append(method)
        for row in ranking:
            if len(selected)>=5:
                break
            if row['method'] not in selected:
                selected.append(row['method'])
        assert len(selected)==5
        strong=any(r['mean_delta_M0']>=.0025 for r in ranking if r['method'] in selected)
        save('G1_FINALISTS_FROZEN.json',{'status':'FROZEN','created_utc':utc_now(),'selected':selected,
            'draws':5000 if strong else 2000,'draw_count_decided_before_bootstrap':True,
            'ranking':ranking,'outer_score_sha256':digest(BASE/'G1_OUTER_SCORES.json'),
            'selection_policy_sha256':digest(BASE/'G1_BOOTSTRAP_SELECTION_POLICY.json'),
            'not_a_promotion_or_deployment_decision':True,
            'known_state_gate_failure':'Uncorrected union event memory exceeds50% latch on multiple16384 null backgrounds'})
        usage=read_json(BASE/'runtime/CANDIDATE_USAGE.json')
        usage.update({'full_completed':12,'updated_utc':utc_now(),'G1_fitted_finalists':selected})
        save('runtime/CANDIDATE_USAGE.json',usage,False)
        save('runtime/CURRENT_RESEARCH_STATUS.json',{'best':ranking[0]['method'],
            'best_mean_delta':ranking[0]['mean_delta_M0'],'bottleneck':'B4_LONG_NULL_FALSE_LATCH',
            'next_task':'paired uncertainty, null-growth hypothesis and structural contribution forensic',
            'screen_count':6,'full_count':12,'created_utc':utc_now()},False)
        print([(r['method'],r['mean_TS_AUC'],r['mean_delta_M0'],r['positive_matched_folds']) for r in summaries],flush=True)
    guard.check_state()
    note('G1 scores released once','Freeze candidates; uncertainty and long-null structural repair next','F4_NULL_GROWTH_CORRECTION')


if __name__=='__main__':
    main()
