"""Freeze references, gates, and mechanistic scenarios before NEXT2 experiments."""
from pathlib import Path
import csv
import json
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.guard import SealGuard
from src.utils.artifacts import sha256,write_json,utc_now,commit_id
from src.next2.session import checkpoint


def main():
    out = ROOT/'artifacts/next2/initialization'
    out.mkdir(parents=True,exist_ok=True)
    lock_path = ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'
    if lock_path.exists():
        raise RuntimeError('NEXT2 policy already frozen; initialization cannot overwrite it.')
    guard = SealGuard(ROOT)
    roles = {'FROZEN_PRIMARY':'J102_SERIAL_VARIANCE_512',
             'FROZEN_SCIENTIFIC_REFERENCE':'Y001_LONG_VARIANCE_ARCH1',
             'FROZEN_ROBUST_REFERENCE':'B104_LONG_VARIANCE_NO_GLR_MEMORY'}
    references = {}
    for role,eid in roles.items():
        exp = ROOT/'artifacts/next/experiments'/eid/'RESULTS.json'
        report = json.loads(exp.read_text(encoding='utf-8'))
        model_path = ROOT/'artifacts/next/refits'/eid/'cache_research_v1/model.joblib'
        desc = model_path.with_name('model.json')
        references[role] = {'experiment_id':eid,'candidate':report['candidate'],
            'full':report['full'], 'model_path':model_path.relative_to(ROOT).as_posix(),
            'model_sha256':sha256(model_path),'descriptor_sha256':sha256(desc),
            'experiment_sha256':sha256(exp)}
    source_files = sorted({p for pattern in ('src/**/*.py','configs/**/*.json','tests/*.py','scripts/*.py') for p in ROOT.glob(pattern)
                           if 'next2' not in p.relative_to(ROOT).as_posix()})
    report_names = ['00_GPT_READ_FIRST_NEXT.md','01_GPT_HANDOFF_NEXT.md','MODEL_RESULTS_NEXT.csv',
        'CV_RESULTS_NEXT.csv','PAIRED_BOOTSTRAP_RESULTS.csv','SEQUENTIAL_STAT_RESULTS.csv','GLR_RESULTS.csv',
        'BAYES_LITE_RESULTS.csv','WHITENING_RESULTS.csv','INNOVATION_NORMALIZATION_RESULTS.csv',
        'HISTORICAL_REGIME_RESULTS.csv','MECHANISM_FRACTURE_RESULTS.csv','BREAK_AGE_NEXT.csv',
        'TIME_FRACTURE_NEXT.csv','CURRENT_TOP_FORENSIC.md','LIMITATIONS_NEXT.md','NEXT_SESSION_PLAN.md',
        'EXPERIMENT_LEDGER.csv','SEAL_OPEN_RECOMMENDATION.md']
    inventory = []
    for name in report_names:
        p = ROOT/name
        raw = p.read_text(encoding='utf-8-sig')
        item = {'path':name,'sha256':sha256(p),'bytes':p.stat().st_size}
        if p.suffix == '.csv':
            records = list(csv.DictReader(raw.splitlines()))
            item.update(rows=len(records),columns=list(records[0]) if records else [])
        else:
            item['characters_read'] = len(raw)
        inventory.append(item)
    shutil.copyfile(ROOT/'EXPERIMENT_LEDGER.csv',out/'EXPERIMENT_LEDGER_NEXT_PRESERVED.csv')
    write_json(out/'PREVIOUS_REPORTS_REVIEWED.json',inventory)
    protected = {p.relative_to(ROOT).as_posix():sha256(p) for p in source_files}
    for name in ('SEAL_PARTITION_500x4_LOCK.json','SPLIT_LOCK_REAL.json','ROBUSTNESS_SPLIT_LOCK.json',
                 'REDUCED_TEST_USAGE_LEDGER.json','data/raw/X_train.parquet','data/raw/y_train_index.parquet',
                 'SOURCE_MANIFEST_NEXT.json','DEV_CANDIDATES_NEXT.json'):
        protected[name] = sha256(ROOT/name)
    write_json(out/'FROZEN_REFERENCES.json',{'created_utc':utc_now(),'git_commit':commit_id(),
        'references':references,'protected_sha256':protected})
    oldpolicy = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    policy = {'version':'NEXT2_V1','created_utc':utc_now(),
        'reference_lock_sha256':sha256(out/'FROZEN_REFERENCES.json'),
        'user_spec_sha256':sha256(ROOT/'docs/USER_SPECIFICATION_NEXT2.md'),
        'references':roles,'metric':'pair-weighted within-online-time TS-AUC, mean of 5 fixed DEV folds',
        'minimum_research_seconds':21600,'additional_audit_seconds':[2700,3600],
        'seals_opened':{k:False for k in guard.lock['partitions']},'reduced_new_usage':0,'public_submission_new':0,
        'screen':{'ids':oldpolicy['screen']['ids'],'folds':[0,4],
            'reference':'J102_SERIAL_VARIANCE_512 matched screen models reused',
            'full_admission':'screen mean delta >= -0.002 AND each screened fold delta >= -0.008',
            'complement_full_admission':'standalone screen mean > 0.54 AND mean within-time rank correlation with matched primary < 0.95',
            'exact_old_experiments':'reuse complete old CV/forensics; reuse failed screens as failed, no new waiver',
            'runtime_exact_equivalence':'no retraining for unchanged features/model; verify complete DEV bitwise predictions'},
        'simplification':{'mean_delta_min':-.0005,'median_delta_min':-.001,'worst_delta_min':-.001,
            'bootstrap_probability_delta_above_minus_0005_min':.90,
            'alternate_mean_delta_min_each':-.003,'alternate_worst_fold_delta_min_each':-.01,
            'h_regime_score_delta_min_each':-.01,
            'complexity':'strict decrease in actual state arrays or paired median prepared point runtime; model/column count alone insufficient'},
        'performance':{'mean_delta_min':.0015,'median_delta_min':0.,'worst_delta_min':-.001,
            'bootstrap_probability_positive_min':.85,'alternate_nonnegative_count_min':2,
            'h_regime_score_delta_min_each':-.01},
        'generation':{'requires':'performance and all engineering gates',
            'rule':'mean delta >= .0035 OR (mean delta >= .0025 AND P > .95 AND all 3 alternate mean deltas > 0 AND independent mechanism gain)',
            'independent_mechanism_gain':'a non-variance complement with supported non-SCALE mechanism (>=100 break series) paired CI lower > 0 and mean delta >= .002 versus frozen J102'},
        'bootstrap':{'replicates':2000,'seed_base':202609094,'seed':'base + fold',
            'unit':'whole DEV series, fixed fitted OOF, paired same multiplicities; conditional not selection-corrected'},
        'robustness':'reuse exact prior ALT_1/2/3 partitions and H-regime lock; train 20 additional fits for nominated finalists',
        'finalist_admission':'any full candidate with mean >= J102-.0005 and median >= J102-.001 and worst >= J102-.001; complements admitted separately before fixed blending',
        'complement':{'standalone_mean_min_exclusive':.54,'rank_correlation_max_exclusive':.95,
            'alternative':'supported mechanism CI lower > 0 and delta >= .002 versus J102',
            'blend_primary_weights':[.9,.8,.67,.5],'weight_optimization':False},
        'fixed_small_grids':{'energy_cap_abs_z':[3,4,6],
            'ages':{'all':[1,2,4,8,16,32,64,128,256,512],'short':[1,2,4,8,16,32],
                    'medium':[16,32,64,128],'long':[128,256,512],
                    'sparse32':[32,64,128,256,512],'sparse64':[64,128,256,512]},
            'memory':['cumulative_max','decayed_max','ewma','leaky','resettable','posterior_like'],
            'normalization':['arch1','arch2','ewma'], 'cdf_bins':[8,16]},
        'adaptive_policy':'record one hypothesis and immutable candidate config before new predictions; no gate relaxation; no failed-screen override; no mass hyperparameter sweep',
        'forbidden_features':['dataset_id','tau','online_total_length','future_suffix','mechanism_label','fold'],
        'synthetic_authority':'mechanistic diagnosis only, never replacement for real DEV quality gates'}
    write_json(lock_path,policy)
    scenarios = [
        ('N0','Gaussian null',0.,0.,0.,0.,None,None,1.,0.),
        ('N1','AR null',.6,.6,0.,0.,None,None,1.,0.),
        ('N2','ARCH null',.3,.3,.65,.65,None,None,1.,0.),
        ('N3','heavy-tail null',-.4,-.4,.4,.4,5,5,1.,0.),
        ('M1','mean increase',.3,.3,0.,0.,None,None,1.,.5),
        ('M2','mean decrease',.3,.3,0.,0.,None,None,1.,-.5),
        ('V1','variance increase x1.5',.3,.3,.65,.65,None,None,1.5,0.),
        ('V2','variance increase x2',.3,.3,.65,.65,None,None,2.,0.),
        ('V3','variance decrease x0.7',.3,.3,.65,.65,None,None,.7,0.),
        ('V4','variance decrease x0.5',.3,.3,.65,.65,None,None,.5,0.),
        ('A1','ARCH persistence increase',.3,.3,.2,.85,None,None,1.,0.),
        ('A2','ARCH persistence decrease',.3,.3,.85,.2,None,None,1.,0.),
        ('T1','Gaussian to Student t3',.3,.3,0.,0.,None,3,1.,0.),
        ('T2','Student t3 to Gaussian',.3,.3,0.,0.,3,None,1.,0.),
        ('D1','positive autocorrelation change',0.,.6,0.,0.,None,None,1.,0.),
        ('D2','autocorrelation sign reversal',.6,-.6,0.,0.,None,None,1.,0.),
        ('O1','single giant outlier, no persistent break',.3,.3,0.,0.,None,None,1.,0.),
        ('X1','variance decrease and dependence',.2,.7,.4,.4,5,5,.5,0.)]
    keys = ['id','description','phi_before','phi_after','arch_before','arch_after','t_df_before','t_df_after','variance_factor','mean_shift_stationary_sd']
    suite = {'version':'NEXT2_COUNTEREXAMPLES_V1','created_utc':utc_now(),'seed':2026090902,
        'rng':'numpy PCG64 SeedSequence([seed, partition, scenario ordinal, series index])',
        'historical_length':2048,'burn_in':1024,'online_length':1024,
        'tau_cycle':[128,256,512],'calibration_null_series_per_regime':1000,'evaluation_series_per_scenario':512,
        'partitions':{'calibration':0,'counterexamples':1,'long_null':2},
        'variance_factor_means':'multiply stationary innovation variance; multiply innovations by sqrt(factor), not factor',
        'ar_scaling':'innovation long-run variance = 1-phi^2 so AR stationary marginal variance is 1 before interventions',
        'arch_scaling':'omega=(1-alpha)*(1-phi^2)*variance_factor, carry state through transition',
        'student_t':'unit variance standardization sqrt((df-2)/df)',
        'outlier':'at scheduled tau add alternating sign 20 stationary SD, labels remain null',
        'scenarios':[dict(zip(keys,row)) for row in scenarios],
        'variance_direction_backgrounds':['Gaussian(phi=0,alpha=0)','AR(phi=.6,alpha=0)',
            'ARCH(phi=.3,alpha=.65)','heavy(phi=-.4,alpha=.4,t5)'],
        'long_null':{'horizons':[1024,4096,16384],'series_per_regime':512,
            'regimes':['N0','N1','N2','N3'],'quantiles':[.5,.9,.95,.99,.999],
            'threshold_alphas':[.01,.05], 'rule':'strict > split-conformal order statistic of disjoint 1024-point calibration maxima; no retuning by horizon'},
        'models':'frozen all-DEV J102/Y001/B104 plus explicitly identified new all-DEV finalists; never mix fold-0 and all-DEV claims',
        'diagnostic_only':True}
    write_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json',suite)
    write_json(ROOT/'SEAL_GUARD_NEXT2.json',{'status':'MEMBERSHIP_PASS_TESTS_PENDING','created_utc':utc_now(),
        'dev_series':len(guard.dev_ids),'seal_series':len(guard.seal_ids),'dev_seal_overlap':len(guard.dev_ids&guard.seal_ids),
        'seals_opened':{k:v['opened'] for k,v in guard.lock['partitions'].items()},
        'reduced_new_usage':0,'public_submission_new':0,'seal_lock_sha256':guard.lock_sha,
        'reduced_ledger_sha256':guard.ledger_sha})
    checkpoint('References and prospective NEXT2 policies frozen',
        ['Re-test seal/scorer guards','Reuse exact component bootstrap evidence','Implement actual runtime pruning'],
        artifacts=['configs/next2/RESEARCH_POLICY_LOCK.json','COUNTEREXAMPLE_SUITE_LOCK.json',
                   'artifacts/next2/initialization/FROZEN_REFERENCES.json'])
    print('NEXT2 FROZEN',len(protected),'protected files',len(inventory),'previous reports',flush=True)


if __name__ == '__main__':
    main()
