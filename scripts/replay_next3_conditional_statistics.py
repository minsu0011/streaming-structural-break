"""Reviewer command: reconstruct existing summaries from compact saved draws."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import csv
import importlib.metadata
import os
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import atomic_json, digest, read_json, utc_now
from src.next3.portable_report_contract import new_output_path, verify_extracted_manifest
from src.next3.claim_checks import ClaimChecks


def csv_candidates(path, names):
    with path.open(encoding='utf-8-sig',newline='') as stream: rows = list(csv.DictReader(stream))
    if len(rows) != len(names) or {row['candidate'] for row in rows} != set(names):
        raise ValueError('Exact registered candidate rows required: '+str(path))
    return {row['candidate']:row for row in rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='review_next3_conditional_statistics.json')
    args=parser.parse_args(); began=time.perf_counter()
    proof=verify_extracted_manifest(ROOT); output=new_output_path(ROOT,args.output,proof['files'])
    versions={}
    for line in (ROOT/'requirements-lock.txt').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.startswith('#'): continue
        name,version=line.split('=='); installed=importlib.metadata.version(name)
        if installed != version: raise RuntimeError('Complete pinned review environment required: '+name)
        versions[name]=installed
    if len(versions) != 63: raise RuntimeError('All 63 pinned packages must be present')
    base=ROOT/'artifacts/next3'; original=base/'selection_uncertainty'; independent=base/'pair_matrix_audit/run_v1'
    paths=[Path(__file__),ROOT/'src/next3/conditional_statistics.py',ROOT/'src/next3/claim_checks.py',
        ROOT/'requirements-lock.txt',*[original/name for name in ('RESULTS.json','POLICY.json','CONDITIONAL_POOL_ENVELOPE.csv','AGGREGATE_CONTRAST_DRAWS.npz')],
        independent/'RESULTS.json',independent/'CONDITIONAL_SUMMARY_RECONSTRUCTED.csv']
    if any(path.relative_to(ROOT).as_posix() not in proof['files'] for path in paths):
        raise RuntimeError('Every statistical replay input must be listed in the verified manifest')
    receipt=read_json(original/'RESULTS.json'); policy=read_json(original/'POLICY.json')
    pair=read_json(independent/'RESULTS.json')
    assert receipt['status'] == pair['status'] == 'PASS'
    assert receipt['selection_adjusted'] is receipt['scientific_promotion_authority'] is False
    assert receipt['replicates'] == policy['replicates'] == 2000 and receipt['candidate_count'] == 12
    for name,sha in receipt['files_sha256'].items(): assert digest(original/name) == sha
    assert digest(independent/'CONDITIONAL_SUMMARY_RECONSTRUCTED.csv') == pair['files_sha256']['CONDITIONAL_SUMMARY_RECONSTRUCTED.csv']
    active=[True]; phase=['RECONSTRUCTION']; writes=[]; forbidden=[]; network=[]
    def audit(event,arguments):
        if not active[0]: return
        if event in ('socket.connect','socket.connect_ex'):
            network.append(event); raise RuntimeError('No network in a saved-statistics replay')
        if event == 'open':
            raw,mode,flags=arguments
            if not isinstance(raw,(str,bytes,os.PathLike)): return
            name=os.fsdecode(raw)
            if os.path.normcase(os.path.normpath(name)) == os.path.normcase(os.path.normpath(os.devnull)): return
            path=Path(name).resolve()
            if path.is_relative_to(ROOT/'data') or (phase[0] == 'RECONSTRUCTION' and path.suffix.lower() in ('.npy','.parquet','.sbnext2','.joblib')):
                forbidden.append(str(path)); raise RuntimeError('No raw data, model or point-array input in saved-statistics replay')
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:
                writes.append(str(path)); raise RuntimeError('No persistent writes during numeric reconstruction')
        if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.truncate','os.link','os.symlink'):
            writes.append(event); raise RuntimeError('No filesystem mutation during numeric reconstruction')
    sys.addaudithook(audit)
    try:
        from src.next3.conditional_statistics import load_saved_aggregate
        result=load_saved_aggregate(original/'AGGREGATE_CONTRAST_DRAWS.npz',policy['candidates'],replicates=2000)
        first=csv_candidates(original/'CONDITIONAL_POOL_ENVELOPE.csv',policy['candidates'])
        second=csv_candidates(independent/'CONDITIONAL_SUMMARY_RECONSTRUCTED.csv',policy['candidates'])
        checks=ClaimChecks()
        checks.numeric('original_critical_value',receipt['critical_value'],result['critical_value'],
            (original/'RESULTS.json').relative_to(ROOT).as_posix(),'critical_value')
        primary_fields=('estimate_delta','conditional_percentile_ci_2_5','conditional_percentile_ci_97_5',
            'conditional_standard_error','joint_max_abs_standardized_quantile95',
            'joint_conditional_envelope_lower','joint_conditional_envelope_upper','performance_effect_threshold')
        secondary_fields={'observed_mean_delta':'estimate_delta','conditional_ci_low':'conditional_percentile_ci_2_5',
            'conditional_ci_high':'conditional_percentile_ci_97_5','probability_delta_positive':'probability_delta_positive',
            'joint_conditional_envelope_lower':'joint_conditional_envelope_lower',
            'joint_conditional_envelope_upper':'joint_conditional_envelope_upper'}
        for row in result['rows']:
            name=row['candidate']; previous=first[name]; alternate=second[name]
            assert previous['reference'] == row['reference']
            for field in primary_fields:
                checks.numeric(name+'/D002/'+field,previous[field],row[field],
                    (original/'CONDITIONAL_POOL_ENVELOPE.csv').relative_to(ROOT).as_posix(),name+'/'+field)
            for source,target in secondary_fields.items():
                checks.numeric(name+'/D007/'+source,alternate[source],row[target],
                    (independent/'CONDITIONAL_SUMMARY_RECONSTRUCTED.csv').relative_to(ROOT).as_posix(),name+'/'+source)
            for field in ('positive_under_joint_conditional_envelope','effect_size_gate_pass','selection_adjusted'):
                assert previous[field] == str(row[field])
            assert alternate['selection_adjusted'] == alternate['independent_validation_of_generalization'] == 'False'
        assert len(checks.claims) == 169
        modules={}
        for name,module in list(sys.modules.items()):
            if name != 'src' and not name.startswith('src.'): continue
            file=getattr(module,'__file__',None)
            if file is None: continue
            path=Path(file).resolve(); assert path.is_relative_to(ROOT),name
            relative=path.relative_to(ROOT).as_posix()
            assert relative in proof['files'] and digest(path) == proof['files'][relative]['sha256']
            modules[name]={'relative':relative,'sha256':digest(path)}
        phase[0]='VERIFY_LISTED_BYTES'
        after=verify_extracted_manifest(ROOT)
        assert after['files'] == proof['files'] and not writes and not forbidden and not network
    finally:
        active[0]=False
    report={'status':'PASS_EXISTING_CONDITIONAL_STATISTICS_REPLAY','created_utc':utc_now(),
        'candidate_count':result['candidate_count'],'existing_draw_rows':result['replicates'],
        'observed_rows':1,'aggregate_shape':[2001,12],'numeric_cross_checks':len(checks.claims),
        'absolute_comparison_tolerance':1e-14,'existing_critical_value_reconstructed':result['critical_value'],
        'reconstructed_existing_summaries':result['rows'],'numeric_comparisons':checks.claims,
        'observed_row_excluded_from_draw_summaries':True,'sample_standard_error_ddof':1,'quantile_method':'linear',
        'new_fits':0,'new_Monte_Carlo_draws':0,'new_point_prediction_or_quality_evaluations':0,
        'raw_competition_inputs_needed':False,'models_needed':False,'point_prediction_arrays_needed':False,
        'selection_adjusted':False,'independent_validation_of_generalization':False,
        'original_sampling_regenerated_by_this_command':False,'actual_archive_bytes_verified_by_this_command':False,
        'source_manifest_sha256':proof['manifest_sha256'],'listed_files_verified_before_and_after':proof['file_count'],
        'project_modules':modules,'pinned_versions':versions,'persistent_write_attempts':writes,
        'raw_model_or_point_array_access_attempts':forbidden,'network_attempts':network,
        'script_sha256':digest(Path(__file__)),'elapsed_seconds':time.perf_counter()-began,
        'scope':'Reconstruction of previously saved aggregate conditional summaries only. Does not regenerate original whole-series draws, read fitted OOF points, correct adaptive selection, establish unconditional coverage, train or evaluate a new model, or independently verify final ZIP bytes.'}
    atomic_json(output,report,immutable=True)
    print('NEXT3 SAVED CONDITIONAL STATISTICS REPLAY PASS',len(checks.claims),'NUMERIC CHECKS',flush=True)


if __name__ == '__main__': main()
