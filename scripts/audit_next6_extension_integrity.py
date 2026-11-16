"""Read-only identity and provenance audit, distinct from performance evaluation."""
from pathlib import Path
import sys, os, argparse, csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np, joblib
from src.next6_extension.common import *
from src.next6.data import source
from src.next6.common import ids_hash

SCALARS={
 'BOOTSTRAP_IMPLEMENTATION_LOCK.json':{'source_sha256':'src/next6_extension/bootstrap_graph.py'},
 'BOOTSTRAP_SIGN_RECOVERY_LOCK.json':{'source_sha256':'scripts/audit_next6_extension_bootstrap_signs.py'},
 'CALIBRATION_TRANSPORT_LOCK.json':{'source_sha256':'scripts/transport_next6_extension_calibration.py','runtime_sha256':'src/next6_extension/fresh_runtime.py'},
 'COHERENT_NULL_REPAIR_LOCK.json':{'source_sha256':'src/next6_extension/factorized_coherent.py','old_source_sha256':'src/next6_extension/factorized.py'},
 'COHERENT_RECHECK_LOCK.json':{'coherent_source_sha256':'src/next6_extension/factorized_coherent.py','runner_sha256':'scripts/validate_next6_extension_coherent.py'},
 'EXTENDED_SYNTHETIC_LOCK.json':{'generator_source_sha256':'src/next6_extension/synthetic.py','original_generator_lock_sha256':'COUNTEREXAMPLE_SUITE_LOCK.json'},
 'FINITE_INFLUENCE_AUDIT_LOCK.json':{'source_sha256':'scripts/audit_next6_extension_finite_influence.py'},
 'FRESH_RP_EVALUATION_LOCK.json':{'source_sha256':'scripts/evaluate_next6_extension_rp.py','threshold_receipt_sha256':'artifacts/next6_extension/calibration/fresh/DONE.json'},
 'FUSED_OBSERVABLE_RUNTIME_LOCK.json':{'adapter_sha256':'src/next6_extension/runtime.py','audit_sha256':'scripts/audit_next6_extension_observable_runtime.py'},
 'HEAD_MEMORY_ATTRIBUTION_LOCK.json':{'source_sha256':'scripts/attribute_next6_extension_head_memory.py'},
 'INTERACTION_POWER_LOCK.json':{'source_sha256':'src/next6_extension/categorical_controls.py','graph_source_sha256':'src/next6_extension/factorized.py'},
 'LATENCY_WORKLOAD_LOCK.json':{'source_sha256':'scripts/benchmark_next6_extension_runtime.py'},
 'LEGACY_MEMORY_DETAIL_LOCK.json':{'source_sha256':'scripts/attribute_next6_extension_legacy_memory.py'},
 'MACRO_NETWORK_AUDIT_LOCK.json':{'source_sha256':'scripts/audit_next6_extension_macro_network.py'},
 'OBSERVABLE_NETWORK_EXPORT_LOCK.json':{'source_sha256':'src/next6_extension/observer.py','coherent_graphs_frozen':'artifacts/next6_extension/COHERENT_GRAPHS_FROZEN.json'},
 'REFERENCE_VALIDITY_REPAIR_LOCK.json':{'adapter_sha256':'src/next6_extension/validated_runtime.py','unchanged_fused_source_sha256':'src/next6_extension/runtime.py','audit_sha256':'scripts/audit_next6_extension_reference_validity.py'},
 'REFINEMENT_IMPLEMENTATION_LOCK.json':{'design_lock_sha256':'artifacts/next6_extension/FRESH_STATE_DESIGN_LOCK.json'},
 'BLOCK_BOUNDARY_AUDIT_LOCK.json':{'source_sha256':'scripts/audit_next6_extension_block_boundaries.py','graph_source_sha256':'src/next6_extension/factorized_coherent.py'},
 'SCALE_BOTTLENECK_AUDIT_LOCK.json':{'source_sha256':'scripts/audit_next6_extension_scale_bottleneck.py'},
}

def verify_map(mapping,base=ROOT):
    for name,sha in mapping.items():
        p=base/name
        assert p.is_file(),str(p)
        assert digest(p)==sha,'Changed bytes: '+str(p)
    return len(mapping)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preliminary','final'],required=True);args=parser.parse_args()
    name=f'audit/{args.stage.upper()}_INTEGRITY.json';assert not (BASE/name).exists()
    with activity(args.stage+'_identity_provenance_audit','INTEGRITY_AUDIT'):
        old=read_json(ROOT/'artifacts/next6/PROTECTED_BASELINE_SHA256.json');nold=verify_map(old)
        frozen=read_json(BASE/'INITIAL_NEXT6_FREEZE.json');ninitial=verify_map(frozen['files_sha256'])
        sources=0
        for lock,mapping in SCALARS.items():
            d=read_json(BASE/lock)
            for key,path in mapping.items():assert digest(ROOT/path)==d[key],f'{lock}: {key}';sources+=1
        for lock in ('REFINEMENT_IMPLEMENTATION_LOCK.json','EXTENDED_EVALUATION_IMPLEMENTATION_LOCK.json'):
            sources+=verify_map(read_json(BASE/lock)['sources'])
        receipt=read_json(BASE/'audit/CORE_ENGINE_TEST_RECEIPT.json');assert receipt['status']=='PASS' and receipt['counts']=={'tests':62,'errors':0,'failures':0,'skipped':0}
        sources+=verify_map(receipt['source_sha256']);sources+=verify_map(receipt['tests_sha256'])
        rp=ROOT/'data/next5/research_5d54e316ff5bf25d';manifest=read_json(rp/'MANIFEST.json');verify_map(manifest['files_sha256'],rp)
        raw=manifest['identity']['raw_sha256'];verify_map(raw,ROOT/'data/raw')
        # Raw parquet was hashed as bytes only; all array admission uses RP6400.
        data=source();assert len(data.ids)==6400 and len(set(data.guard.split.groups[int(i)] for i in data.ids))==6400
        model_rows=[];p=np.load(BASE/'rp_replay/macro_probabilities.npy',allow_pickle=False)
        counts=np.load(BASE/'rp_replay/series_transition_counts.npy',allow_pickle=False)
        support=np.load(BASE/'rp_replay/series_transition_support.npy',allow_pickle=False)
        assert counts.shape==support.shape==(6400,2,7,7) and np.isfinite(counts).all() and np.all(counts>=0)
        assert p.shape==(5,2,7,7)
        graphlocks={kind:read_json(BASE/lock)['graphs'] for kind,lock in [('fresh_graphs','FRESH_GRAPHS_FROZEN.json'),('coherent_graphs','COHERENT_GRAPHS_FROZEN.json')]}
        modellock=read_json(BASE/'OBSERVABLE_NETWORKS_FROZEN.json')['models']
        cut=read_json(BASE/'calibration/fresh/DONE.json')['thresholds']['FRESH']
        for fold in range(5):
            tr=data.series['fold']!=fold;train=set(map(int,data.ids[tr]));valid=set(map(int,data.ids[~tr]));data.guard.partition(train,valid);assert len(train)==5120 and len(valid)==1280
            for kind in graphlocks:
                path=BASE/f'{kind}/fold_{fold}/graph.joblib';assert digest(path)==graphlocks[kind][str(fold)];g=joblib.load(path);g.validate();assert set(g.provenance['training_ids'])==train and g.provenance['training_ids_sha256']==ids_hash(train)
                assert not g.selected
            modelpath=BASE/f'observable_models/fold_{fold}/model.joblib';assert digest(modelpath)==modellock[str(fold)];m=joblib.load(modelpath);m.validate()
            assert m.threshold==cut and m.provenance['training_ids_sha256']==ids_hash(train) and set(m.graph.provenance['training_ids'])==train
            v=counts[tr].sum(axis=0)+.5;expected=v/v.sum(axis=-1,keepdims=True)
            np.testing.assert_array_equal(expected,p[fold]);np.testing.assert_array_equal(expected,m.macro_probabilities);np.testing.assert_array_equal(support[tr].sum(axis=0),m.macro_support)
            model_rows.append({'fold':fold,'training_series':len(train),'heldout_series':len(valid),'training_groups_disjoint':True,'model_sha256':digest(modelpath),'macro_tables_reconstructed_from_original_training_only':True,'selected_cross_edges':len(m.graph.selected)})
        positive=joblib.load(BASE/'coherent_power/KNOWN_POSITIVE_GRAPH.joblib');positive.validate();assert len(positive.selected)>0
        block=read_json(BASE/'block_cache/MANIFEST.json');verify_map(block['files_sha256'],BASE/'block_cache');assert block['blocks']==197192 and block['original_split_sha256']==data.guard.split.digest
        boundary=read_json(BASE/'BLOCK_BOUNDARY_AUDIT_DONE.json');assert boundary['blocks_audited']==197192 and boundary['all_five_selection_unchanged']
        viewer=read_json(BASE/'audit/VIEWER_AUDIT.json');assert viewer['status']=='PASS' and viewer['viewer_sha256']==digest(BASE/'DYNAMIC_STATE_VIEWER.html') and viewer['cases']==11
        data.guard.check_state()
        completed={'protected_original_files':nold,'initial_NEXT6_frozen_files':ninitial,'locked_source_and_test_checks':sources,'RP_cache_files_hashed':len(manifest['files_sha256']),'raw_files_byte_hashed_only':len(raw),'RP_series':6400,'RP_unique_original_groups':6400,'block_pairs':197192,'model_checks':model_rows,'core_tests_unchanged_since_PASS':62,'current_viewer_audit_PASS':True,'known_positive_nonempty_graph_valid':True}
        if args.stage=='final':
            assert digest(BASE/'audit/sources/PRELIMINARY_INTEGRITY_SOURCE.py')==read_json(BASE/'audit/PRELIMINARY_INTEGRITY.json')['source_sha256']
            for required in ['SYNTHETIC_FINAL_STATISTICS_DONE.json','GRAPH_BOOTSTRAP_FINAL_STATISTICS_DONE.json','LATENCY_NEXT6_EXTENSION.json','FINITE_INFLUENCE_AUDIT_DONE.json','COHERENT_POWER_DONE.json','COHERENT_RAW_NULL_DONE.json','REFERENCE_VALIDITY_AUDIT_DONE.json','audit/FRESH_ENV_PORTABLE_PROBE.json']:
                assert (BASE/required).is_file(),required
            graph=read_json(BASE/'GRAPH_BOOTSTRAP_FINAL_STATISTICS_DONE.json');assert graph['graphs']==1000 and graph['sign_reconstructions']==1000
            syn=read_json(BASE/'SYNTHETIC_FINAL_STATISTICS_DONE.json');assert syn['cohorts']==138 and syn['rows_audited']==259072
            power=list((BASE/'coherent_power').glob('*/replicate_*.json'));rawnull=list((BASE/'coherent_raw_null').glob('*/replicate_*.json'));assert len(power)==1680 and len(rawnull)==160
            for p in power:
                r=read_json(p);assert r['null_coherence_validated'] and len(r['selected'])==r['selected_edges']
            for p in rawnull:
                r=read_json(p);assert r['series']==512 and r['selected_edges']==0 and r['same_fresh_feature_engine'] and r['null_coherence_validated']
            finite=read_json(BASE/'FINITE_INFLUENCE_AUDIT_DONE.json');assert finite['cases']==46080 and finite['all27_columns_bitwise_coupled'] and finite['all_mutable_state_bitwise_coupled']
            with (BASE/'FINITE_INFLUENCE_STRESS_CASES.csv').open(encoding='utf-8-sig') as f:assert sum(1 for _ in csv.DictReader(f))==46080
            for fold in range(5):
                d=BASE/f'graph_bootstrap/fold_{fold}';signed=BASE/f'bootstrap_signs/fold_{fold}';assert len(list(d.glob('replicate_*.json')))==len(list(signed.glob('replicate_*.json')))==200
                for replicate in range(200):
                    a=read_json(d/f'replicate_{replicate:03}.json');b=read_json(signed/f'replicate_{replicate:03}.json');assert a['fold']==b['fold']==fold and a['replicate']==b['replicate']==replicate and a['multiplicity_sha256']==b['exact_multiplicity_sha256']
                    assert {(e['source'],e['target'],e['lag'],e['selection_phase']) for e in a['selected']}=={(e['source'],e['target'],e['lag'],e['selection_phase']) for e in b['edges']}
                    assert all(e['both_internal_halves_same_sign'] for e in b['edges'])
            completed['all_long_running_statistics_complete']=True
            completed['coherent_power_receipts_checked']=len(power);completed['coherent_raw_null_receipts_checked']=len(rawnull)
    save(name,{'status':'PASS','stage':args.stage,'created_utc':utc_now(),'checks':completed,'source_sha256':digest(Path(__file__)),'scope':'Byte identities, original training membership and table reconstruction; no new predictive fitting. Application guards are not an OS sandbox.','excluded_data_access':'No Shadow/Seal/raw parquet row decoding; canonical raw files byte-hashed only.'})
    print('PASS',args.stage,completed,flush=True)

if __name__=='__main__':main()
