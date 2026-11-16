"""Attach explicitly labelled fitted-model diagnostics to component comparisons."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    out=ROOT/'artifacts/next2/component_synthesis';out.mkdir(parents=True,exist_ok=True)
    model_path=ROOT/'MODEL_RESULTS_NEXT2.csv';models=pd.read_csv(model_path,float_precision='round_trip')
    sources={};cost_rows=[];diagnostics=[]
    diagnostic_folders=sorted((ROOT/'artifacts/next2/component_stress_cv0').iterdir())+sorted((ROOT/'artifacts/next2/reused_component_stress_cv0').iterdir())
    for directory in diagnostic_folders:
        receipt=directory/'RESULTS.json'
        if not receipt.exists():continue
        r=json.loads(receipt.read_text());assert r['status']=='PASS'
        sources[str(receipt.relative_to(ROOT))]=sha256(receipt)
        counter=pd.read_csv(directory/'COUNTEREXAMPLES.csv',float_precision='round_trip')
        long=pd.read_csv(directory/'LONG_NULL.csv',float_precision='round_trip')
        row={'experiment_id':directory.name,'synthetic_fitted_model':'original full-CV fold 0; not all-DEV model'}
        for scenario in ('M1','M2','V1','V2','V3','V4','A1','A2','T1','T2','D1','D2','X1'):
            one=counter[(counter.scenario==scenario)&(counter.alpha==.05)];assert len(one)==1
            row['cv0_synthetic_'+scenario+'_ts_auc']=float(one.iloc[0].ts_auc)
            row['cv0_synthetic_'+scenario+'_tau512_clean_detection_alpha05']=float(one.iloc[0].tau512_clean_detection_rate)
        one=counter[(counter.scenario=='O1')&(counter.alpha==.05)];assert len(one)==1
        row['cv0_outlier_alarm_excess_alpha05']=float(one.iloc[0].outlier_alarm_excess)
        for bg in ('N0','N1','N2','N3'):
            one=long[(long.regime==bg)&(long.horizon==16384)&(long.alpha==.05)];assert len(one)==1
            row['cv0_'+bg+'_null_fpr_16384_alpha05']=float(one.iloc[0].eventual_alarm_rate)
        diagnostics.append(row)
    latency_receipts=sorted((ROOT/'artifacts/next2/component_latency').glob('*/RESULTS.json'))+sorted((ROOT/'artifacts/next2/reused_component_latency').glob('*/RESULTS.json'))
    for path in latency_receipts:
        r=json.loads(path.read_text());assert r['status']=='PASS'
        assert sha256(path.parent/'SERIES_TIMING.csv')==r['table_sha256']
        sources[str(path.relative_to(ROOT))]=sha256(path)
        cost_rows.append({'experiment_id':r['identity']['candidate'],'latency_fitted_model':'original full-CV fold 0; isolated fixed probes',
            'state_size':r['state_array_bytes_max'],'private_per_series_numeric_bytes':r['numerical_memory']['private_per_series_numeric_bytes'],
            'latency':r['point_distribution']['median_us'],'latency_unit':'microseconds per scalar point',
            'historical_init_median_us':r['historical_initialization']['median_us'],'historical_init_p95_us':r['historical_initialization']['p95_us'],
            'historical_init_p99_us':r['historical_initialization']['p99_us'],
            'feature_update_median_us':r['feature_state_update']['median_us'],'model_prediction_median_us':r['model_prediction']['median_us'],
            'point_p95_us':r['point_distribution']['p95_us'],'point_p99_us':r['point_distribution']['p99_us'],
            'direct_1M_seconds':r['direct_1M_seconds'],'extrapolated_10M_seconds':r['extrapolated_10M_seconds'],
            'direct_10M_measured':False,'h_plus_online_amortized_us_per_point':r['h_plus_online_amortized_us_per_point']})
    costs=pd.DataFrame(cost_rows);synthetic=pd.DataFrame(diagnostics)
    assert len(costs)==len(synthetic)==28
    assert costs.experiment_id.is_unique and synthetic.experiment_id.is_unique
    models=models.drop(columns=[c for c in costs.columns if c!='experiment_id' and c in models],errors='ignore')
    models=models.drop(columns=[c for c in synthetic.columns if c!='experiment_id' and c in models],errors='ignore')
    models=models.merge(costs,on='experiment_id',how='left',validate='one_to_one').merge(synthetic,on='experiment_id',how='left',validate='one_to_one')
    models.to_csv(model_path,index=False)
    families={'EVIDENCE_MEMORY_RESULTS.csv':('M2_','N2_','D205','B101','B104','J102','Y001'),
        'BAYES_AGE_ABLATION.csv':('A2_','V007','B102','B104','J102','Y001'),
        'GLR_NECESSITY_RESULTS.csv':('D202','B101','B102','B104','J102','Y001'),
        'SERIAL_CORRECTION_RESULTS.csv':('J102','Y001'),
        'DISTILLATION_RESULTS.csv':('D2','B10','A2_','V007','J102','Y001')}
    for name,prefixes in families.items():
        frame=models[models.experiment_id.str.startswith(prefixes)].copy();frame['record_type']='SCIENTIFIC_COMPONENT_ABLATION'
        if name=='DISTILLATION_RESULTS.csv':
            runtime=pd.read_csv(ROOT/'RUNTIME_PRUNING_RESULTS.csv',float_precision='round_trip')
            runtime['record_type']='EXACT_RUNTIME_VARIANT';runtime['experiment_id']=runtime['runtime_variant']
            runtime['full_mean']=runtime.full_cv_mean;runtime['median']=runtime.full_cv_median;runtime['worst_fold']=runtime.full_cv_worst
            runtime['delta']=runtime.full_cv_mean-float(models.loc[models.experiment_id=='J102_SERIAL_VARIANCE_512','full_mean'].iloc[0])
            runtime['feature_count']=runtime.materialized_tree_input_features
            runtime['latency']=runtime.point_median_us;runtime['state_size']=runtime.state_array_bytes_including_shared_selectors
            frame=pd.concat([frame,runtime],ignore_index=True)
        frame.to_csv(ROOT/name,index=False)
    # Supplemental generators are added as distinct diagnostic records, not mixed with original cohorts.
    cross=pd.read_csv(ROOT/'artifacts/next2/cross_background_scale/RESULTS.csv',float_precision='round_trip')
    cross=cross.merge(models[['experiment_id','full_mean','delta','status']],left_on='candidate',right_on='experiment_id',how='left')
    cross['diagnostic_suite']='CROSS_BACKGROUND_PARTITION_3'
    scale=pd.read_csv(ROOT/'SCALE_DOWN_RESULTS.csv',float_precision='round_trip')
    if 'diagnostic_suite' in scale:scale=scale[scale.diagnostic_suite!='CROSS_BACKGROUND_PARTITION_3']
    scale['diagnostic_suite']='ORIGINAL_FROZEN_SUITE'
    pd.concat([scale,cross],ignore_index=True).to_csv(ROOT/'SCALE_DOWN_RESULTS.csv',index=False)
    qpath=ROOT/'artifacts/next2/component_null_quantiles/QUANTILES.csv'
    qr=json.loads((qpath.parent/'RESULTS.json').read_text());assert qr['table_sha256']==sha256(qpath)
    quantiles=pd.read_csv(qpath,float_precision='round_trip').drop(columns=['fitted_fold','artifact_kind'])
    supplemental_qpath=ROOT/'artifacts/next2/reused_component_null_quantiles/QUANTILES.csv'
    sqr=json.loads((supplemental_qpath.parent/'RESULTS.json').read_text());assert sqr['table_sha256']==sha256(supplemental_qpath)
    supplemental_quantiles=pd.read_csv(supplemental_qpath,float_precision='round_trip').drop(columns=['fitted_fold','artifact_kind'])
    quantiles=pd.concat([quantiles,supplemental_quantiles],ignore_index=True)
    long_path=ROOT/'LONG_NULL_RESULTS_NEXT2.csv';long=pd.read_csv(long_path,float_precision='round_trip')
    joined=long.merge(quantiles,on=['candidate','regime','horizon'],how='left',suffixes=('','_cv0'),validate='many_to_one')
    mask=joined.fitted_fold==0
    for column in quantiles.columns:
        if column in ('candidate','regime','horizon'):continue
        if column+'_cv0' in joined:
            joined.loc[mask,column]=joined.loc[mask,column+'_cv0']
            joined=joined.drop(columns=[column+'_cv0'])
        elif column not in long:
            joined.loc[~mask,column]=np.nan
    joined.to_csv(long_path,index=False)
    late=[]
    for path in sorted((ROOT/'artifacts/next2/late_alarm_diagnostic').glob('*/ANALYSIS.csv')):
        r=json.loads((path.parent/'ANALYSIS.json').read_text());assert r['table_sha256']==sha256(path)
        sources[str(path.relative_to(ROOT))]=sha256(path);late.append(pd.read_csv(path,float_precision='round_trip'))
    assert len(late)==4
    pd.concat(late,ignore_index=True).to_csv(out/'LONG_LATE_ALARM_RESULTS.csv',index=False)
    costs.to_csv(out/'COMPONENT_COSTS.csv',index=False);synthetic.to_csv(out/'COMPONENT_SYNTHETIC_WIDE.csv',index=False)
    # Restore preserved ledger history and the newly enriched research records.
    history=pd.read_csv(ROOT/'artifacts/next2/initialization/EXPERIMENT_LEDGER_NEXT_PRESERVED.csv',float_precision='round_trip')
    history['ledger_section']='PREVIOUS_SESSION_PRESERVED'
    current=models[models.evidence_origin=='NEXT2_RESEARCH'].copy();current['ledger_section']='NEXT2_RESEARCH'
    pd.concat([history,current],ignore_index=True).to_csv(ROOT/'EXPERIMENT_LEDGER.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'RESEARCH_SYNTHESIS','created_utc':utc_now(),'component_models':len(costs),
        'scientific_model_rows':len(models),'source_sha256':sources,'script_sha256':sha256(Path(__file__)),
        'scope':'Real DEV CV scores remain primary; diagnostics label their fitted model and generator. CV0 costs and CV0 synthetic observations are not all-DEV deployment observations.',
        'limitations':'Generic ablation banks can still compute omitted feature columns internally. B104_COMPACT removes actual GLR and GLR-memory work; D202 retains the GLR maximum needed by its retained memory. Exact runtime variants have their own full-DEV measured costs and proofs.'})
    checkpoint('Component quality, cost and fixed counterexample synthesis',['Inspect mechanism and cost tradeoffs','Build research figures and exact claim provenance'])


if __name__=='__main__':main()
