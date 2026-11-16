"""DEV-only health audit of a fitted representation; no fitting or score selection."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse, gc, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.streaming import feature_names, FEATURE_FAMILIES
from src.models.representations import bundle_class
from src.validation.splits import FrozenSplit, assert_development_ids
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args()
    registry=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text())
    result=next(r for r in registry['results'] if r['experiment_id']==args.candidate)
    representation=result['candidate'].get('representation','base')
    model=bundle_class(representation).load(ROOT/'artifacts/models/real'/args.candidate/'fold_0.joblib')
    from src.models.representations import representation_feature_names
    names=representation_feature_names(representation,model.feature_config);split=FrozenSplit(ROOT/'artifacts/validation')
    families=FEATURE_FAMILIES*(2 if representation=='cross_order_ar4_ar8_equal' else 1)
    oof=json.loads((ROOT/'data/processed/oof'/args.candidate/'MANIFEST.json').read_text())
    if oof['split_sha256']!=split.digest:raise RuntimeError('OOF split identity mismatch')
    cache=ROOT/oof['cache_directory'];manifest=json.loads((cache/'MANIFEST.json').read_text())
    if not all((cache/f'fold_{k}.parquet').exists() for k in range(5)):raise ValueError('This audit expects complete calibrated fold caches')
    n=0;sums=np.zeros(len(names));squares=sums.copy();finite=sums.copy();low=np.full(len(names),np.inf);high=-low.copy();samples=[]
    for fold in range(5):
        path=cache/f'fold_{fold}.parquet'
        if sha256(path)!=manifest['fold_sha256'][str(fold)]:raise RuntimeError('Feature cache byte identity changed')
        frame=pd.read_parquet(path,columns=['dataset_id',*names]);assert_development_ids(frame.dataset_id.unique(),split)
        expected=split.frame.loc[(split.frame.role=='DEVELOPMENT_POOL')&(split.frame.fold==fold),'dataset_id']
        if set(frame.dataset_id.unique())!=set(expected):raise RuntimeError('Incomplete fold coverage')
        matrix=frame.loc[:,names].to_numpy(dtype=np.float32);n+=len(matrix);finite+=np.isfinite(matrix).sum(axis=0)
        # Float64 aggregation by column avoids a second full dense fold copy.
        for j in range(len(names)):
            col=matrix[:,j].astype(np.float64);sums[j]+=col.sum();squares[j]+=np.dot(col,col)
        low=np.minimum(low,matrix.min(axis=0));high=np.maximum(high,matrix.max(axis=0))
        sample_idx=np.unique(np.linspace(0,len(matrix)-1,min(50000,len(matrix)),dtype=int));samples.append(matrix[sample_idx].copy())
        del matrix,frame;gc.collect()
    sample=np.concatenate(samples);rows=[]
    for j,name in enumerate(names):
        values=sample[:,j];unique,counts=np.unique(values,return_counts=True);mode=int(np.argmax(counts))
        rows.append({'name':name,'family':families[j],'full_rows':n,'finite_rate':float(finite[j]/n),
            'full_mean':float(sums[j]/n),'full_variance':float(max(0.,squares[j]/n-(sums[j]/n)**2)),
            'full_min':float(low[j]),'full_max':float(high[j]),'sample_distinct_values':len(unique),
            'sample_modal_value':float(unique[mode]),'sample_modal_fraction':float(counts[mode]/len(values)),
            'sample_quantiles':np.quantile(values,[0,.0001,.001,.01,.5,.99,.999,.9999,1]).tolist(),
            'sample_fraction_abs_at_least_12':float(np.mean(abs(values)>=12)),
            'near_constant':bool(high[j]-low[j]<=1e-12),'modal_saturation_9999pct':bool(counts[mode]/len(values)>=.9999)})
    selected=np.std(sample.astype(np.float64),axis=0)>1e-12
    correlation=np.corrcoef(sample[:,selected].astype(np.float64),rowvar=False);selected_names=np.asarray(names)[selected]
    pairs=sorted([{'left':selected_names[i],'right':selected_names[j],'correlation':float(correlation[i,j])}
        for i in range(len(selected_names)) for j in range(i+1,len(selected_names))],key=lambda x:abs(x['correlation']),reverse=True)[:20]
    report={'status':'PASS' if np.all(finite==n) else 'FAIL','timestamp':utc_now(),'candidate':args.candidate,'representation':representation,
        'feature_hash':model.feature_hash,'cache_manifest_sha256':sha256(cache/'MANIFEST.json'),'split_sha256':split.digest,
        'rows':n,'sample_rows':len(sample),'sample_policy':'at most 50000 evenly spaced row indices per frozen DEV fold; no labels used',
        'features':rows,'largest_absolute_correlations':pairs,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False,
        'scope':'Post-fit diagnostic only. Constant-column removal remains learned independently from each training fold. Feature values and model scores are unchanged.',
        'cost_scope':f'The online kernels compute {len(names)} features across the declared representations. Production historical MAD calibration is restricted to actual tree split columns. Incremental online feature-kernel savings per family have not been implemented; initialization and point timing are measured separately.'}
    directory=ROOT/'artifacts/real/feature_health'/args.candidate
    write_json(directory/'AUDIT.json',report)
    pd.DataFrame([{k:v for k,v in row.items() if k!='sample_quantiles'} for row in rows]).to_csv(directory/'DISTRIBUTIONS.csv',index=False)
    pd.DataFrame(correlation,index=selected_names,columns=selected_names).to_csv(directory/'CORRELATIONS.csv')
    print('FITTED FEATURE AUDIT',report['status'],args.candidate,'rows',n,'constant',[r['name'] for r in rows if r['near_constant']],flush=True)
    if report['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
