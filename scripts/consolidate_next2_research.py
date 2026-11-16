"""Rebuild reviewable research tables from receipts; this does not finalize roles or handoff."""
from pathlib import Path
import sys
import json
import shutil
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.decision_gates import assess
from src.next2.session import clock,checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

OUT=ROOT/'artifacts/next2/research_tables'
J='J102_SERIAL_VARIANCE_512'
SPLITS=('ALT_1','ALT_2','ALT_3','LEAVE_ONE_H_REGIME_OUT')
SOURCES={}


def read(path):
    path=ROOT/path if not Path(path).is_absolute() else Path(path)
    SOURCES[path.relative_to(ROOT).as_posix()]=sha256(path)
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.suffix=='.json' else pd.read_csv(path,float_precision='round_trip')


def save(frame,name):
    frame.to_csv(ROOT/name,index=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    policy=read('configs/next2/RESEARCH_POLICY_LOCK.json')
    frozen=read('artifacts/next2/initialization/FROZEN_REFERENCES.json')
    reference=frozen['references']['FROZEN_PRIMARY']['full']
    reused=read('artifacts/next2/component_reuse/DISTILLATION_REUSED.csv')
    reports={}
    for eid in reused.experiment_id:
        reports[eid]=(read(f'artifacts/next/experiments/{eid}/RESULTS.json'),'EXACT_PRIOR_EXPERIMENT_REUSE')
    for path in sorted((ROOT/'artifacts/next2/experiments').glob('*/RESULTS.json')):
        r=read(path)
        reports[r['candidate']['experiment_id']]=(r,'NEXT2_RESEARCH')
    # Full 24-scope forensics take precedence over the all-time attribution extension.
    paired=[read(path) for path in sorted((ROOT/'artifacts/next2/forensic').glob('*/PAIRED_FRACTURES.csv'))]
    paired.append(read('artifacts/next2/component_reuse/PAIRED_CONTRASTS.csv'))
    path=ROOT/'artifacts/next2/component_effects/PAIRED_EFFECTS.csv'
    if path.exists():
        effects=read(path);effects['scope']='all_time';paired.append(effects)
    paired=pd.concat(paired,ignore_index=True).drop_duplicates(['candidate','reference','scope'],keep='first')
    save(paired,'PAIRED_BOOTSTRAP_NEXT2.csv')
    save(paired[paired.scope.str.startswith('mechanism_')],'MECHANISM_RESULTS_NEXT2.csv')

    reference_stress={split:read(f'artifacts/next/robustness/candidates/{J}/{split}/RESULTS.json') for split in SPLITS}
    stress,alt_rows,regime_rows={},{},[]
    flat_alt=[]
    for eid in reports:
        folder=ROOT/'artifacts/next2/robustness'/eid
        if not folder.exists():
            folder=ROOT/'artifacts/next/robustness/candidates'/eid
        summaries={}
        for split in SPLITS:
            path=folder/split/'RESULTS.json'
            if not path.exists():
                continue
            r=read(path)
            if r.get('status')!='PASS':
                continue
            scores=np.asarray(r['fold_scores']);ref=np.asarray(reference_stress[split]['fold_scores'])
            assert scores.shape==ref.shape==(5,)
            summary={'candidate':eid,'reference':J,'split':split,'mean':float(scores.mean()),
                'mean_delta':float(scores.mean()-ref.mean()),'worst':float(scores.min()),
                'worst_delta':float(scores.min()-ref.min()),'minimum_matched_fold_delta':float((scores-ref).min()),
                'fold_scores':scores.tolist(),'reference_fold_scores':ref.tolist()}
            summaries[split]=summary
            for f,(value,baseline) in enumerate(zip(scores,ref)):
                row={k:v for k,v in summary.items() if k not in ('fold_scores','reference_fold_scores')}
                row.update(fold=f,ts_auc=float(value),reference_ts_auc=float(baseline),fold_delta=float(value-baseline),
                    record_type='REFITTED_LEAVE_H_REGIME_OUT' if split==SPLITS[-1] else 'REFITTED_ALTERNATE_SPLIT')
                (regime_rows if split==SPLITS[-1] else flat_alt).append(row)
        stress[eid]=summaries
    save(pd.DataFrame(flat_alt),'ALT_SPLIT_RESULTS_NEXT2.csv')
    slices=paired[paired.scope.str.startswith('regime_')].copy();slices['record_type']='ORIGINAL_OOF_H_REGIME_SLICE'
    save(pd.concat([pd.DataFrame(regime_rows),slices],ignore_index=True),'REGIME_RESULTS_NEXT2.csv')

    models,cv,gates=[],[],{}
    for eid,(report,origin) in reports.items():
        c=report['candidate'];full=report.get('full',{});screen=report.get('screen',{})
        evidence=paired[(paired.candidate==eid)&(paired.reference==J)&(paired.scope=='all_time')]
        pair=evidence.iloc[0].to_dict() if len(evidence) else None
        ss=stress[eid]
        mechanism=paired[(paired.candidate==eid)&(paired.reference==J)&paired.scope.str.startswith('mechanism_')&
            (paired.scope!='mechanism_SCALE_DOMINANT')]
        independent=bool(any(bool(r.get('supported',False)) and r.ci_2_5>0 and r.estimate_mean_delta>=.002
            for _,r in mechanism.iterrows())) if len(mechanism) else None
        gate=assess(full if full.get('status')=='COMPLETE' else None,reference,pair,
            [ss[s] for s in SPLITS[:3] if s in ss],ss.get(SPLITS[-1]),policy=policy,
            independent_mechanism_gain=independent)
        gates[eid]=gate
        feature_count=len(full.get('feature_names',screen.get('feature_names',[]))) or None
        if c.get('blend'):
            feature_count=sum(len(reports[c['blend'][key]][0]['full']['feature_names']) for key in ('primary','complement'))
        row={'experiment_id':eid,'parent':c.get('parent_id'),'generation':c.get('generation'),
            'hypothesis':c.get('hypothesis'),'component_change':str(c.get('change')),
            'counterexample_target':c.get('counterexample_target'),'feature_count':feature_count,'state_size':None,
            'model':c.get('learner'),'screen_score':screen.get('mean'),'full_mean':full.get('mean'),
            'median':full.get('median'),'worst_fold':full.get('worst_fold'),
            'delta':full.get('mean',float('nan'))-reference['mean'],
            'bootstrap_ci_low':pair.get('ci_2_5') if pair else None,'bootstrap_ci_high':pair.get('ci_97_5') if pair else None,
            'bootstrap_p':pair.get('probability_delta_positive') if pair else None,
            'bootstrap_noninferiority_p':pair.get('probability_delta_above_minus_0005') if pair else None,
            **{'alt_split_'+str(i+1):ss.get(s,{}).get('mean') for i,s in enumerate(SPLITS[:3])},
            'regime_worst':ss.get(SPLITS[-1],{}).get('worst'),'regime_min_matched_delta':ss.get(SPLITS[-1],{}).get('minimum_matched_fold_delta'),
            'long_null_status':'SEE_ARTIFACT_KIND_SPECIFIC_TABLE','synthetic_status':'SEE_ARTIFACT_KIND_SPECIFIC_TABLE','latency':None,
            'status':report['status'],'reason':report.get('reason'),'evidence_origin':origin,
            'performance_scientific_gate':gate['performance_scientific']['status'],
            'performance_gate':gate['performance']['status'],'distilled_gate':gate['distilled']['status'],
            'generation_gate':gate['generation']['status'],'new_fits':0 if origin.startswith('EXACT_') or c.get('blend') or c.get('parity_reference') else
                len(screen.get('fold_scores',[]))+len(full.get('fold_scores',[])),
            'failed_performance_checks':';'.join(gate['performance']['failed']),
            'failed_distilled_checks':';'.join(gate['distilled']['failed'])}
        models.append(row)
        for scope,r in [('screen',screen),('full',full)]:
            for f,value in zip(([0,4] if scope=='screen' else range(5)),r.get('fold_scores',[])):
                cv.append({'experiment_id':eid,'scope':scope,'fold':f,'ts_auc':value,'evidence_origin':origin,'seal_rows':0})
    models=pd.DataFrame(models)
    save(models,'MODEL_RESULTS_NEXT2.csv');save(pd.DataFrame(cv),'CV_RESULTS_NEXT2.csv')
    history=read('artifacts/next2/initialization/EXPERIMENT_LEDGER_NEXT_PRESERVED.csv')
    history['ledger_section']='PREVIOUS_SESSION_PRESERVED'
    current=models[models.evidence_origin=='NEXT2_RESEARCH'].copy();current['ledger_section']='NEXT2_RESEARCH'
    save(pd.concat([history,current],ignore_index=True),'EXPERIMENT_LEDGER.csv')
    save(models[models.experiment_id.str.startswith('C2')],'COMPLEMENT_STANDALONE_RESULTS.csv')
    save(models[models.experiment_id.str.startswith('BL2_')],'COMPLEMENT_BLEND_RESULTS.csv')

    # Keep the original reference summaries before creating combined tables.
    for name in ('COUNTEREXAMPLE_RESULTS.csv','LONG_NULL_RESULTS_NEXT2.csv'):
        preserved=OUT/('ORIGINAL_'+name)
        if not preserved.exists():
            shutil.copyfile(ROOT/name,preserved)
    counter=[read(OUT/'ORIGINAL_COUNTEREXAMPLE_RESULTS.csv')]
    counter[0]['artifact_kind']='frozen all-DEV model'
    long=[read(OUT/'ORIGINAL_LONG_NULL_RESULTS_NEXT2.csv')]
    long[0]['artifact_kind']='frozen all-DEV model'
    for name,target in [('COUNTEREXAMPLES.csv',counter),('LONG_NULL.csv',long)]:
        frame=read('artifacts/next2/finalist_counterexamples/'+name);frame['artifact_kind']='frozen all-DEV model';target.append(frame)
        for path in sorted((ROOT/'artifacts/next2/component_stress_cv0').glob('*/'+name)):
            frame=read(path);frame['fitted_fold']=0;target.append(frame)
        for path in sorted((ROOT/'artifacts/next2/reused_component_stress_cv0').glob('*/'+name)):
            frame=read(path);frame['fitted_fold']=0;frame['diagnostic_supplement']='THREE_REUSED_ABLATIONS';target.append(frame)
    counter=pd.concat(counter,ignore_index=True);long=pd.concat(long,ignore_index=True)
    save(counter,'COUNTEREXAMPLE_RESULTS.csv');save(long,'LONG_NULL_RESULTS_NEXT2.csv')
    synthetic=counter.merge(models[['experiment_id','full_mean','delta','status']],left_on='candidate',right_on='experiment_id',how='left')
    save(synthetic[synthetic.scenario.isin(['V1','V2','V3','V4','X1'])],'SCALE_DOWN_RESULTS.csv')
    save(synthetic[synthetic.scenario.isin(['N3','T1','T2','O1'])],'HEAVY_TAIL_ROBUSTNESS_RESULTS.csv')
    save(synthetic[synthetic.scenario.isin(['N2','A1','A2'])],'ARCH_ROBUSTNESS_RESULTS.csv')
    family={'EVIDENCE_MEMORY_RESULTS.csv':models.experiment_id.str.startswith(('M2_','N2_','D205','B101')),
            'BAYES_AGE_ABLATION.csv':models.experiment_id.str.startswith(('A2_','V007','B102','Y001')),
            'GLR_NECESSITY_RESULTS.csv':models.experiment_id.str.startswith(('D202','B101','B102','B104','Y001','J102')),
            'SERIAL_CORRECTION_RESULTS.csv':models.experiment_id.str.startswith(('J102','Y001')),
            'DISTILLATION_RESULTS.csv':models.experiment_id.str.startswith(('D2','B10','A2_','V007','J102','Y001'))}
    for name,mask in family.items():
        save(models[mask],name)
    save(paired[paired.scope.str.startswith(('age_','time_'))],'TIME_AGE_RESULTS_NEXT2.csv')
    write_json(OUT/'GATES.json',{'created_utc':utc_now(),'policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),
        'gates':gates,'final_selection':False,'engineering_status':'PENDING_SEPARATE_AUDIT'})
    write_json(OUT/'RESULTS.json',{'status':'RESEARCH_SNAPSHOT','created_utc':utc_now(),'clock':clock(),
        'model_records':len(models),'new_experiment_records':int((models.evidence_origin=='NEXT2_RESEARCH').sum()),
        'original_cv_new_fits':int(models.new_fits.sum()),'paired_comparisons':len(paired),
        'synthetic_counterexample_rows':len(counter),'synthetic_long_null_rows':len(long),
        'source_sha256':SOURCES,'script_sha256':sha256(Path(__file__)),'no_final_role_selection':True})
    checkpoint('Consolidated NEXT2 research tables',['Finish outstanding diagnostics','Isolated latency and state accounting'])
    print('NEXT2 TABLES',len(models),'models',len(paired),'paired contrasts',flush=True)


if __name__=='__main__':
    main()
