"""Read measured costs and ownership receipts without selecting the fastest trial."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    policy_path=ROOT/'configs/next2/RUNTIME_MEASUREMENT_LOCK.json'
    policy=json.loads(policy_path.read_text())
    memory_path=ROOT/'artifacts/next2/numeric_memory/MEMORY_BY_H.csv'
    memory_receipt=json.loads((memory_path.parent/'RESULTS.json').read_text())
    assert memory_receipt['status']=='PASS' and memory_receipt['files_sha256'][memory_path.name]==sha256(memory_path)
    memory=pd.read_csv(memory_path,float_precision='round_trip')
    frozen=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    reference_by_id={r['candidate']['experiment_id']:r for r in frozen['references'].values()}
    results={};sources={str(policy_path.relative_to(ROOT)):sha256(policy_path),str(memory_path.relative_to(ROOT)):sha256(memory_path)}
    rows=[];pair_rows=[]
    for variant in policy['variant_order']:
        folder=ROOT/'artifacts/next2/latency'/variant
        receipt=folder/'RESULTS.json';r=json.loads(receipt.read_text())
        assert r['status']=='PASS' and r['identity']['policy_sha256']==sha256(policy_path)
        assert r['identity']['recipe']==policy['variants'][variant]
        for name,digest in r['files_sha256'].items():assert sha256(folder/name)==digest
        sources[str(receipt.relative_to(ROOT))]=sha256(receipt);results[variant]=r
    for variant,r in results.items():
        recipe=policy['variants'][variant];eid=recipe['candidate_id']
        original=variant[0]+'_ORIGINAL' if variant[0] in 'JYB' and not variant.startswith('BL') else 'J_ORIGINAL'
        baseline=results[original]
        own=memory[memory.variant==variant];base_memory=memory[memory.variant==original]
        assert len(own)==len(base_memory)==3
        private=int(own.private_per_series_numeric_bytes.max());base_private=int(base_memory.private_per_series_numeric_bytes.max())
        shared=int(own.model_and_prepared_shared_numeric_bytes.max())
        feature_contract=101 if variant.startswith('BL') else len(reference_by_id[eid]['full']['feature_names'])
        materialized=feature_contract
        if recipe['runtime']=='tree_used':
            proof=ROOT/'artifacts/next2/used_runtime'/eid/'RESULTS.json'
            exact=json.loads(proof.read_text());assert exact['status']=='PASS' and exact['prediction_max_abs_diff']==0
            materialized=exact['materialized_tree_input_features'];sources[str(proof.relative_to(ROOT))]=sha256(proof)
        scientific=reference_by_id[eid]['full'] if eid in reference_by_id else json.loads((ROOT/'artifacts/next2/experiments'/eid/'RESULTS.json').read_text())['full']
        row={'runtime_variant':variant,'candidate':eid,'runtime':recipe['runtime'],'comparison_runtime':original,
            'full_cv_mean':scientific['mean'],'full_cv_median':scientific['median'],'full_cv_worst':scientific['worst_fold'],
            'scientific_model_feature_contract':feature_contract,'materialized_tree_input_features':materialized,
            'materialized_feature_reduction_fraction':1-materialized/feature_contract,
            'private_per_series_numeric_bytes':private,'state_array_bytes_including_shared_selectors':r['state_array_bytes_max'],
            'model_and_prepared_shared_numeric_bytes':shared,'private_state_reduction_fraction':1-private/base_private,
            'legacy_property_state_reduction_fraction':1-r['state_array_bytes_max']/baseline['state_array_bytes_max'],
            'point_median_us':r['point_distribution']['median_us'],'point_p95_us':r['point_distribution']['p95_us'],
            'point_p99_us':r['point_distribution']['p99_us'],'point_median_reduction_fraction':1-r['point_distribution']['median_us']/baseline['point_distribution']['median_us'],
            'historical_init_median_us':r['historical_initialization']['median_us'],'historical_init_p95_us':r['historical_initialization']['p95_us'],
            'historical_init_p99_us':r['historical_initialization']['p99_us'],
            'historical_init_median_reduction_fraction':1-r['historical_initialization']['median_us']/baseline['historical_initialization']['median_us'],
            'feature_update_median_us':r['feature_state_update']['median_us'],'model_prediction_median_us':r['model_prediction']['median_us'],
            'h_plus_online_amortized_us_per_point':r['h_plus_online_amortized_us_per_point'],
            'amortized_reduction_fraction':1-r['h_plus_online_amortized_us_per_point']/baseline['h_plus_online_amortized_us_per_point'],
            'direct_1M_seconds':r['continuous_stream']['measured_first_million_seconds'],
            'direct_10M_seconds':r['continuous_stream']['measured_prediction_seconds'],
            'continuous_10M_reduction_fraction':1-r['continuous_stream']['measured_prediction_seconds']/baseline['continuous_stream']['measured_prediction_seconds'],
            'state_growth_after_direct_10M_bytes':r['continuous_stream']['state_array_bytes_after']-r['continuous_stream']['state_array_bytes_before'],
            'model_file_bytes':r['model_file_bytes'],'model_format':recipe['model_format'],
            'first_history_plus_jit_seconds':r['first_history_plus_jit_seconds_fresh_process'],
            'role_status':'RESEARCH_RUNTIME_COMPARISON_ONLY','quality_semantics_unchanged':not variant.startswith('BL'),
            'receipt':str((ROOT/'artifacts/next2/latency'/variant/'RESULTS.json').relative_to(ROOT))}
        rows.append(row)
        if variant==original:continue
        probe=pd.read_csv(ROOT/'artifacts/next2/latency'/variant/'SERIES_TIMING.csv',float_precision='round_trip')
        ref=pd.read_csv(ROOT/'artifacts/next2/latency'/original/'SERIES_TIMING.csv',float_precision='round_trip')
        joined=probe.merge(ref,on=['repeat','dataset_id','historical_points','online_points'],suffixes=('_candidate','_reference'),validate='one_to_one')
        assert len(joined)==384
        for quantity in ('historical_init_ns','online_loop_ns','point_median_ns'):
            # Average the three repetitions within each series before describing ratios.
            means=joined.groupby('dataset_id')[[quantity+'_candidate',quantity+'_reference']].mean()
            ratios=means[quantity+'_candidate']/means[quantity+'_reference']
            pair_rows.append({'runtime_variant':variant,'reference_runtime':original,'quantity':quantity,'series':128,'repeats_per_series':3,
                'median_ratio':float(ratios.median()),'q05_ratio':float(ratios.quantile(.05)),'q95_ratio':float(ratios.quantile(.95)),
                'fraction_series_faster':float((ratios<1).mean()),'interval_type':'empirical spread across fixed series; not a confidence interval'})
    table=pd.DataFrame(rows);table.to_csv(ROOT/'RUNTIME_PRUNING_RESULTS.csv',index=False)
    out=ROOT/'artifacts/next2/runtime_summary';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(out/'PAIRED_SERIES_COSTS.csv',index=False)
    write_json(ROOT/'LATENCY_NEXT2.json',{'status':'MEASURED_RESEARCH_RESULTS','created_utc':utc_now(),'final_role_selection':False,
        'policy_sha256':sha256(policy_path),'variants':{k:{a:b for a,b in v.items() if a not in ('identity','files_sha256')} for k,v in results.items()},
        'limitations':'All fixed trials retained. Sequential isolated workspace processes; ordinary OS and power conditions uncontrolled. Prefix timer instrumentation and separate feature/tree dispatch add overhead. Direct continuous 1M/10M excludes H; H-inclusive extrapolations are separately labelled. Fixed real buffer repetition measures cost, not quality.'})
    write_json(ROOT/'STATE_MEMORY_NEXT2.json',{'status':'PASS','created_utc':utc_now(),
        'accounting_receipt_sha256':sha256(memory_path.parent/'RESULTS.json'),
        'boundaries':'Private per-series retained NumPy ultimate-owner storage excludes shared selectors and model/prepared arrays. Legacy state property includes shared selectors. Python objects, JIT code, allocator headers and transient peak H work are not in retained numeric byte counts.',
        'observations':memory.to_dict('records'),'continuous_growth_check':{k:v['continuous_stream']['state_array_bytes_after']-v['continuous_stream']['state_array_bytes_before'] for k,v in results.items()},
        'rss_scope':'Benchmark RSS includes interpreter, JIT, loaded data and model; it is not per-series state allocation.'})
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'sources_sha256':sources,
        'script_sha256':sha256(Path(__file__)),'runtime_variants':len(rows),'paired_cost_rows':len(pair_rows),
        'files_sha256':{p:sha256(ROOT/p) for p in ('RUNTIME_PRUNING_RESULTS.csv','LATENCY_NEXT2.json','STATE_MEMORY_NEXT2.json')},
        'no_quality_reselection':True})
    checkpoint('Complete isolated runtime and state comparisons',['Component timing and scientific tradeoffs','Separate final audit after six research hours'])
    print(table[['runtime_variant','point_median_us','historical_init_median_us','direct_10M_seconds','private_per_series_numeric_bytes']].to_string(index=False),flush=True)


if __name__=='__main__':main()
