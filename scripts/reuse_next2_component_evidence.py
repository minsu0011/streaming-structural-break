"""Recompute NEXT2 paired contrasts from immutable existing matched OOF draws."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.bootstrap import paired_summary
from src.utils.artifacts import sha256,write_json,utc_now
from src.next2.session import checkpoint


def load(eid):
    path=ROOT/'artifacts/next/experiments'/eid/'RESULTS.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    folder=ROOT/'artifacts/next/forensic'/eid
    forensic=json.loads((folder/'RESULTS.json').read_text(encoding='utf-8')) if (folder/'RESULTS.json').exists() else None
    if forensic:
        assert forensic['status']=='PASS'
        for name,digest in forensic['files_sha256'].items():
            assert sha256(folder/name)==digest
    robust=ROOT/'artifacts/next/robustness/candidates'/eid/'RESULTS.json'
    stress=json.loads(robust.read_text(encoding='utf-8')) if robust.exists() else None
    return result,folder,stress


def main():
    primary='J102_SERIAL_VARIANCE_512'
    scientific='Y001_LONG_VARIANCE_ARCH1'
    jobs=[('D0',primary,primary,'Full current reference'),
          ('D1',scientific,primary,'Remove serial correction'),
          ('D3','B101_LONG_VARIANCE_NO_MEMORY',scientific,'Remove GLR max/decayed memory'),
          ('D4','B104_LONG_VARIANCE_NO_GLR_MEMORY',scientific,'Remove explicit GLR and GLR memory'),
          ('D7','B102_LONG_VARIANCE_NO_BAYES',scientific,'Remove Bayes-lite'),
          ('D9','B103_LONG_VARIANCE_FORECAST_ONLY',scientific,'Remove raw energy channel'),
          ('D11','V007_VARIANCE_FORECAST',scientific,'Remove ages 256 and 512'),
          ('ARCH2','Y002_LONG_VARIANCE_ARCH2',scientific,'Replace ARCH1 with ARCH2'),
          ('ARCH_EWMA','Y003_LONG_VARIANCE_EWMA',scientific,'Replace ARCH1 with EWMA')]
    # Exact IDs are discovered from artifacts; aliases are not trusted.
    existing={p.parent.name:p for p in (ROOT/'artifacts/next/experiments').glob('*/RESULTS.json')}
    for j,(tag,eid,parent,change) in enumerate(jobs):
        if eid not in existing:
            matches=[name for name in existing if name.startswith(eid.split('_')[0]+'_')]
            if len(matches)!=1:
                raise RuntimeError('Ambiguous previous candidate '+eid)
            jobs[j]=(tag,matches[0],parent,change)
    p,pfolder,pstress=load(primary)
    pfull=p['full']
    rows,contrasts,altrows,regimerows=[],[],[],[]
    all_sources={}
    for tag,eid,parent,change in jobs:
        result,folder,stress=load(eid)
        sources={f'artifacts/next/experiments/{eid}/RESULTS.json':sha256(existing[eid])}
        full=result.get('full',{})
        row={'ablation':tag,'experiment_id':eid,'parent':parent,'component_change':change,
            'generation':'NEXT_REUSED_IN_NEXT2','evidence_status':'REUSED_EXACT',
            'previous_stage_status':result['status'],'full_mean':full.get('mean'),
            'median':full.get('median'),'worst_fold':full.get('worst_fold'),
            'feature_count':len(full.get('feature_names',[])) or None,
            'screen_mean':result.get('screen',{}).get('mean')}
        if full.get('status')=='COMPLETE':
            row.update(delta=full['mean']-pfull['mean'],median_delta=full['median']-pfull['median'],
                worst_delta=full['worst_fold']-pfull['worst_fold'])
            for reference in sorted({primary,parent}):
                ref,ref_folder,_=load(reference)
                scopes=pd.read_csv(folder/'PAIRED_FRACTURES.csv')
                for item in scopes.to_dict('records'):
                    scope=item['scope']
                    c=np.stack([np.load(folder/f'fold_{f}_{scope}.npy',allow_pickle=False) for f in range(5)],axis=1)
                    r=np.stack([np.load(ref_folder/f'fold_{f}_{scope}.npy',allow_pickle=False) for f in range(5)],axis=1)
                    assert c.shape==r.shape==(2001,5)
                    comparison={'ablation':tag,'candidate':eid,'reference':reference,'scope':scope,
                        'supported':item['supported'],'break_series':item['break_series'],**paired_summary(r,c),
                        'probability_delta_above_minus_0005':float(np.mean((c.mean(axis=1)-r.mean(axis=1))[1:]>-.0005))}
                    contrasts.append(comparison)
                    if reference==primary and scope=='all_time':
                        row.update(bootstrap_ci_low=comparison['ci_2_5'],bootstrap_ci_high=comparison['ci_97_5'],
                            bootstrap_p=comparison['probability_delta_positive'],
                            bootstrap_noninferiority_p=comparison['probability_delta_above_minus_0005'])
            if stress:
                for c,r in zip(stress['summaries'],pstress['summaries']):
                    assert c['split_name']==r['split_name']
                    stressrow={'candidate':eid,'reference':primary,'split':c['split_name'],
                        'mean':c['mean'],'mean_delta':c['mean']-r['mean'],
                        'worst':c['worst'],'worst_delta':c['worst']-r['worst']}
                    (regimerows if c['split_name']=='LEAVE_ONE_H_REGIME_OUT' else altrows).append(stressrow)
                row['alternate_catastrophe_absent']=all(c['mean']-r['mean']>=-.003 and c['worst']-r['worst']>=-.01 for c,r in zip(stress['summaries'][:3],pstress['summaries'][:3]))
                row['h_regime_catastrophe_absent']=all(a-b>=-.01 for a,b in zip(stress['summaries'][3]['fold_scores'],pstress['summaries'][3]['fold_scores']))
            row['noninferiority_numeric_gates_pass']=bool(row['delta']>=-.0005 and row['median_delta']>=-.001 and row['worst_delta']>=-.001 and row['bootstrap_noninferiority_p']>=.9 and row.get('alternate_catastrophe_absent') and row.get('h_regime_catastrophe_absent'))
            row['promotion_status']='RUNTIME_EVIDENCE_REQUIRED' if row['noninferiority_numeric_gates_pass'] else 'NOT_DISTILLED_ELIGIBLE'
            sources[folder.relative_to(ROOT).as_posix()+'/RESULTS.json']=sha256(folder/'RESULTS.json')
        else:
            row['promotion_status']='PREVIOUS_SCREEN_FAILURE_REUSED_NO_REFIT'
        rows.append(row)
        all_sources.update(sources)
    out=ROOT/'artifacts/next2/component_reuse'
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'DISTILLATION_REUSED.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'PAIRED_CONTRASTS.csv',index=False)
    pd.DataFrame(altrows).to_csv(out/'ALT_SPLITS.csv',index=False)
    pd.DataFrame(regimerows).to_csv(out/'H_REGIMES.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'reused_candidates':len(rows),
        'new_fits':0,'paired_contrasts':len(contrasts),'source_sha256':all_sources,
        'policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),
        'interpretation':'Matched removals are vs their explicit parent, role eligibility is vs frozen J102. CI overlap zero is not proof of equivalence.'})
    checkpoint('Prior matched component evidence reused under new gates',
        ['Complete actual pruning verification','Screen missing matched ablations'],
        artifacts=['artifacts/next2/component_reuse/RESULTS.json'])
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
