"""Close completed research after the requested wall-time minimum; no time padding."""
from pathlib import Path
import sys,os,csv,datetime,json,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6_extension.common import *

def instant(value):return datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
def merged_seconds(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    return sum((b-a).total_seconds() for a,b in merged),merged

def main():
    session=read_json(BASE/'SESSION.json');now=datetime.datetime.now(datetime.timezone.utc)
    assert now>=instant(session['minimum_continue_until_utc']),'The explicitly requested seven additional hours have not elapsed.'
    assert not (BASE/'FINAL_AUDIT_NEXT6_EXTENSION.json').exists()
    for name in ('SYNTHETIC_FINAL_STATISTICS_DONE.json','GRAPH_BOOTSTRAP_FINAL_STATISTICS_DONE.json','LATENCY_NEXT6_EXTENSION.json','BLOCK_BOUNDARY_AUDIT_DONE.json','SCALE_BOTTLENECK_AUDIT_DONE.json','audit/FINAL_VISUAL_REPRODUCTION.json'):
        assert (BASE/name).exists(),name
    if not (BASE/'audit/FINAL_INTEGRITY.json').exists():subprocess.run([sys.executable,'-B',str(ROOT/'scripts/audit_next6_extension_integrity.py'),'--stage','final'],cwd=ROOT,check=True)
    integrity=read_json(BASE/'audit/FINAL_INTEGRITY.json');assert integrity['status']=='PASS'
    figures=sorted((BASE/'FIGURES').glob('*.png'));assert len(figures)==20 and {int(p.name[:2]) for p in figures}==set(range(1,21))
    for p in figures:assert p.with_suffix('.svg').is_file() and p.stat().st_size>10000
    imageaudit=read_json(BASE/'audit/FINAL_FIGURE_REVIEW.json');assert imageaudit['status']=='PASS'
    for name,sha in imageaudit['files_sha256'].items():assert digest(BASE/name)==sha
    calfreeze=read_json(BASE/'BASELINE_CALIBRATION_FROZEN.json');menu=read_json(BASE/'EXTENDED_SYNTHETIC_LOCK.json')
    for fold in range(5):
        path=BASE/f'calibration/fold_{fold}/DONE.json';assert digest(path)==calfreeze['receipts'][str(fold)]
        assert instant(read_json(path)['created_utc'])<=instant(calfreeze['created_utc'])
        for case in menu['cases']:
            rec=read_json(BASE/f'extended_synthetic/fold_{fold}/{case}_DONE.json');assert rec['calibration_sha256']==digest(path) and instant(rec['created_utc'])>instant(calfreeze['created_utc'])
    stamp=utc_now();start=instant(session['started_utc']);end=instant(stamp)
    with (BASE/'EXPERIMENT_LEDGER.csv').open(encoding='utf-8-sig',newline='') as f:ledger=list(csv.DictReader(f))
    assert ledger and all(None not in r and None not in r.values() for r in ledger)
    intervals=[];bykind={}
    for row in ledger:
        a=instant(row['started_utc']);b=instant(row['ended_utc']);assert a<=b
        assert abs((b-a).total_seconds()-float(row['wall_seconds']))<1.,row['task']
        a=max(a,start);b=min(b,end)
        if a<=b:intervals.append((a,b));bykind.setdefault(row['kind'],[]).append((a,b))
    union,merged=merged_seconds(intervals);elapsed=(end-start).total_seconds();assert union<=elapsed+1e-6 and elapsed>=7*3600
    time_receipt={'status':'RESEARCH_AUDIT_COMPLETE','started_utc':start.isoformat(),'completed_utc':stamp,'elapsed_wall_seconds':elapsed,'elapsed_hours':elapsed/3600,'minimum_requested_additional_hours':7,'minimum_met':True,'recorded_task_union_wall_seconds':union,'other_elapsed_seconds_for_coding_review_and_unlogged_work':elapsed-union,'sum_of_recorded_task_wall_seconds_with_overlaps':sum(float(r['wall_seconds']) for r in ledger),'sum_of_recorded_process_CPU_seconds_not_a_machine_total':sum(float(r['process_cpu_seconds']) for r in ledger),'task_rows':len(ledger),'failed_task_rows_preserved':sum(r['status']!='COMPLETE' for r in ledger),'union_intervals':[[a.isoformat(),b.isoformat()] for a,b in merged],'per_kind_union_wall_seconds_nonadditive':{k:merged_seconds(v)[0] for k,v in bykind.items()},'interpretation':'Parallel wall intervals are merged, not added to claim elapsed research time. CPU sums may include overlapping or nested measurements. Source editing and reading are not process-time profiled. Packaging finishes later and has its own elapsed-time receipt. No ten-hour or operating-system isolation claim.'}
    save('TIME_ACCOUNTING_NEXT6_EXTENSION.json',time_receipt)
    decision={'created_utc':stamp,'decision':'NO_PROMOTION','retained_roles':{'champion':'BL80','robustness':'D2','fallback':'J102'},'new_output':'Bounded observable-state explanation and descriptive normal/post transition probabilities, not a calibrated historical-break classifier.','confirmed_repairs':['Coherent whole-ownstate support backoff preserves null interaction constraint','Safe float64 H reductions for finite float32 extremes','Insufficient reference yields neutral unavailable evidence','Fused scalar calculation matches all RP outputs','Finite contamination influence couples within conservative +128 bound'],'unresolved_limits':['Weak scale-contraction confirmation; quantified negative-channel bounds and four-block loss','Fixed H2048/horizon1024 calibration does not control every H length or longer monitoring horizon','All five RP discriminative interaction graphs remain empty','Seven displayed states omit clock and bounded internal memory; not a full Markov generative state','Adaptive development reuse is not independent confirmation'],'new_supervised_recipes':0,'outer_score_threshold_or_weight_tuning':False,'hidden_or_external_competition_test_used':False}
    decision['unresolved_limits'].append('Categorical small-sample null control selected false links in 3/40 fits (7.5%); no formal false-discovery guarantee')
    rp_diagnostic=read_json(BASE/'FRESH_RP_EVALUATION.json');runtime_summary=read_json(BASE/'LATENCY_NEXT6_EXTENSION.json');bootstrap_summary=read_json(BASE/'BOOTSTRAP_INTERPRETATION_SUMMARY.json')
    decision['diagnostic_evidence']={'RP_persistent_evidence_only_TS_AUC':rp_diagnostic['mean_diagnostic_TS_AUC'],'RP_fixed_prediction_conditional_CI':rp_diagnostic['conditional_CI'],'not_a_new_learned_detector':True,'final_scalar_runtime':next(r for r in runtime_summary['rows'] if r['method']=='OBSERVABLE_FUSED27'),'mechanical_share_of_resampled_selected_link_occurrences':bootstrap_summary['mechanical_share_of_link_occurrences'],'original_M3_delta_vs_BL80':0.0004692843700441207,'unchanged_minimum_effect_gate':0.0012}
    save('EXTENSION_RESEARCH_DECISION.json',decision)
    final={'status':'PASS','started_utc':session['started_utc'],'completed_utc':stamp,'elapsed_hours':elapsed/3600,'minimum_additional_seven_hours_met':True,'decision':'NO_PROMOTION','original_BL80_D2_retained':True,'integrity_receipt_sha256':digest(BASE/'audit/FINAL_INTEGRITY.json'),'core_engine_tests':62,'initial_NEXT6_tests_separately_preserved':1371,'RP_series':6400,'RP_online_points':3203157,'original_grouped_outer_folds':5,'original_graph_training_resamples':1000,'selected_coefficient_sign_reconstructions':1000,'baseline_fold_scenario_cohorts':115,'paired_synthetic_seeds_per_cohort':512,'synthetic_horizons':[1024,4096],'coherent_known_interaction_fits':1680,'same_engine_raw_null_graphs':160,'finite_influence_cases':46080,'scientific_figure_pairs':20,'interactive_cases':11,'original_protected_files_unchanged':701,'initial_NEXT6_frozen_files_unchanged':706,'all_public_runtime_core_sources_unchanged_since_test_receipt':True,'adaptive_development_not_confirmatory':True,'remaining_limits':decision['unresolved_limits'],'handoff_status':'Separate extracted-ZIP and Desktop20 receipt is produced after this research audit.','time_accounting_sha256':digest(BASE/'TIME_ACCOUNTING_NEXT6_EXTENSION.json'),'audited_script_sha256':digest(Path(__file__))}
    save('FINAL_AUDIT_NEXT6_EXTENSION.json',final)
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/report_next6_extension.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/audit_next6_extension_document_links.py'),'--stage','final'],cwd=ROOT,check=True)
    print('Research audit PASS; packaging remains.',elapsed/3600,flush=True)

if __name__=='__main__':main()
