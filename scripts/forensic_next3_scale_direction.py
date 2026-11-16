"""Evaluation-only directional SCALE proxies; never a training input or target."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.evaluation import load_full_oof
from src.next.data import prepare_dev
from src.next.bootstrap import PreparedAUC,paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities


def main():
    set_num_threads(2);hyp=BASE/'hypotheses/D001_DIRECTIONAL_SCALE_FORENSIC.json';registration=read_json(hyp)
    for name,sha in registration['sources_sha256'].items():assert digest(ROOT/name)==sha
    data=prepare_dev(ROOT);out=BASE/'directional_scale';out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULTS.json').exists():raise RuntimeError('Do not overwrite directional diagnostic')
    source=ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    signatures=pd.read_parquet(source).set_index('dataset_id')
    data.guard.admit(signatures.index.to_numpy(),purpose='NEXT3 evaluation-only signed SCALE mechanism proxy')
    fixed=read_json(ROOT/'ROBUSTNESS_SPLIT_LOCK.json')['mechanism']
    direction={};errors=[]
    for row in signatures[signatures.signature=='SCALE_DOMINANT'].itertuples():
        h,o,tau=data.get(int(row.Index));assert tau==row.tau and tau>=0
        pre=np.concatenate([np.asarray(h,dtype=np.float64),np.asarray(o[:tau],dtype=np.float64)])[-fixed['window']:]
        post=np.asarray(o[tau:tau+fixed['window']],dtype=np.float64)
        assert min(len(pre),len(post))>=fixed['minimum_each_side']
        effect=float(np.log(max(float(post.std()),1e-8)/max(float(pre.std()),1e-8)))
        error=abs(abs(effect)-float(row.scale_effect));assert error<=1e-12;errors.append(error)
        direction[int(row.Index)]='decrease' if effect<0 else 'increase' if effect>0 else 'equal'
    counts={kind:sum(v==kind for v in direction.values()) for kind in ('decrease','increase','equal')}
    previous=ROOT/'artifacts/next2/dev_scale_direction/DIRECTION_COUNTS.csv'
    known=pd.read_csv(previous)
    for kind,label in [('decrease','OBSERVED_SD_DECREASE'),('increase','OBSERVED_SD_INCREASE')]:
        row=known[(known.scope=='existing_signature')&(known.signature=='SCALE_DOMINANT')&(known.direction==label)]
        assert len(row)==1 and int(row.iloc[0]['series'])==counts[kind]
    _,_,metadata,reference,refsource=load_full_oof(data,'BL2_C205_DEPENDENCE_80')
    candidates=['J102_SERIAL_VARIANCE_512']
    for path in sorted((BASE/'experiments').glob('*/full/RESULTS.json')):
        if read_json(path).get('finalist_admitted'):candidates.append(path.parent.parent.name)
    atomic_json(out/'POLICY.json',{'created_utc':utc_now(),'hypothesis_sha256':digest(hyp),'candidate_rule':'Frozen J plus every completed full-admitted NEXT3 candidate at diagnostic start',
        'candidates':candidates,'direction_counts':counts,'maximum_direction_reconstruction_error':max(errors),
        'reference_sources':refsource,'no_per_series_direction_export':True},immutable=True)
    ids=metadata['dataset_id'];series_direction=np.asarray([direction.get(int(sid),'other') for sid in ids])
    is_null=signatures.reindex(ids).tau.to_numpy()<0
    bootstrap_cache={}
    for fold in range(5):
        rows=np.flatnonzero(metadata['fold']==fold)
        code,draws=draw_group_multiplicities(ids[rows],2000,registration['seed_base']+fold)
        bootstrap_cache[fold]=(rows,code,draws)
    reports=[];scores=[]
    for candidate in candidates:
        _,_,meta,pred,sources=load_full_oof(data,candidate)
        for key in metadata:np.testing.assert_array_equal(metadata[key],meta[key])
        saved={kind:[] for kind in ('decrease','increase')};baseline={kind:[] for kind in saved}
        for fold,(rows,code,draws) in bootstrap_cache.items():
            for kind in saved:
                mask=(series_direction[rows]==kind)|is_null[rows]
                y,t=metadata['target'][rows][mask],metadata['time_online'][rows][mask]
                a=PreparedAUC.create(y,reference[rows][mask],t,code[mask],draws.shape[1]).evaluate(draws)
                b=PreparedAUC.create(y,pred[rows][mask],t,code[mask],draws.shape[1]).evaluate(draws)
                baseline[kind].append(a);saved[kind].append(b)
                scores.append({'candidate':candidate,'fold':fold,'direction':kind,'ts_auc':b[0],'reference_ts_auc':a[0],
                    'delta':b[0]-a[0],'break_series':len(np.unique(ids[rows][series_direction[rows]==kind]))})
        for kind in saved:
            a=np.asarray(baseline[kind]).T;b=np.asarray(saved[kind]).T
            reports.append({'candidate':candidate,'reference':'BL2_C205_DEPENDENCE_80','direction':kind,'break_series':counts[kind],
                'supported_at_100_breaks':counts[kind]>=100,'candidate_mean':float(b[0].mean()),'reference_mean':float(a[0].mean()),
                'diagnostic_only':True,**paired_summary(a,b)})
        progress('Evaluation-only signed SCALE diagnostic',f'{candidate}: fixed 79/53 direction support; no promotion authority below 100',experiment='D001_DIRECTIONAL_SCALE_FORENSIC')
        print('NEXT3 SIGNED SCALE',candidate,counts,flush=True)
    pd.DataFrame(reports).to_csv(out/'PAIRED_DIRECTIONAL_SCORES.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'FOLD_DIRECTIONAL_SCORES.csv',index=False)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'hypothesis_sha256':digest(hyp),'candidate_count':len(candidates),
        'direction_counts':counts,'new_fits':0,'seal_rows':0,'no_training_use':True,'conditional_not_selection_adjusted':True,
        'scope':'Observed raw-window SD direction inside frozen SCALE_DOMINANT signature; not latent innovation-variance change.',
        'both_directions_below_supported_threshold':all(counts[k]<100 for k in ('decrease','increase')),
        'files_sha256':{name:digest(out/name) for name in ('POLICY.json','PAIRED_DIRECTIONAL_SCORES.csv','FOLD_DIRECTIONAL_SCORES.csv')}},immutable=True)


if __name__=='__main__':main()
