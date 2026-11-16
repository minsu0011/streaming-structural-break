"""New admitted ResearchPool descriptors; old per-series forensic tables stay shut."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.data import prepare_research
from src.next.regimes import historical_descriptors,mechanism_signature


def main():
    directory=BASE/'diagnostics';policy=read_json(directory/'POLICY.json');transform=read_json(directory/'HISTORICAL_TRANSFORM.json')
    assert policy['mechanism_source_sha256']==digest(ROOT/'src/next/regimes.py')
    if (directory/'RESEARCH_SIGNATURES.npz').exists():raise RuntimeError('Preserve fixed diagnostic artifacts')
    data=prepare_research();descriptors=[];signatures=[];directions=[]
    for index,sid in enumerate(data.ids,1):
        h,o,tau=data.get(int(sid));description=historical_descriptors(h)
        descriptors.append([description[name] for name in transform['columns']])
        signatures.append(mechanism_signature(h,o,tau,policy['mechanism']))
        direction='other'
        if signatures[-1]['signature']=='SCALE_DOMINANT':
            pre=np.r_[h.astype(float),o[:tau].astype(float)][-policy['mechanism']['window']:]
            post=o[tau:tau+policy['mechanism']['window']].astype(float)
            signed=np.log(max(float(post.std()),1e-8)/max(float(pre.std()),1e-8))
            assert abs(abs(signed)-signatures[-1]['scale_effect'])<1e-12
            direction='decrease' if signed<0 else 'increase' if signed>0 else 'equal'
        directions.append(direction)
        if index%500==0:
            progress('ResearchPool diagnostic descriptors',str(index)+'/6400; H-only regimes, separate post-hoc mechanisms')
            print('NEXT5 DIAGNOSTICS',index,flush=True)
    values=np.asarray(descriptors,dtype=np.float64);assert np.isfinite(values).all()
    standardized=np.clip((values-np.asarray(transform['mean']))/np.asarray(transform['scale']),-8.,8.)
    centers=np.asarray(transform['centers']);distance=np.sum((standardized[:,None,:]-centers[None,:,:])**2,axis=2)
    regime=np.argmin(distance,axis=1).astype(np.int8)
    np.savez_compressed(directory/'RESEARCH_H_DESCRIPTORS.npz',dataset_id=data.ids,values=values,columns=np.asarray(transform['columns']),historical_regime=regime)
    fields={name:np.asarray([row[name] for row in signatures]) for name in signatures[0]}
    np.savez_compressed(directory/'RESEARCH_SIGNATURES.npz',dataset_id=data.ids,tau=data.series['tau'],historical_regime=regime,
        online_length_diagnostic_only=np.diff(data.series['online_offsets']),observed_scale_direction=np.asarray(directions),**fields)
    labels,counts=np.unique(fields['signature'],return_counts=True)
    files=('RESEARCH_H_DESCRIPTORS.npz','RESEARCH_SIGNATURES.npz','HISTORICAL_TRANSFORM.json')
    data.guard.check_state()
    atomic_json(directory/'RESEARCH_DIAGNOSTIC_LOCK.json',{'created_utc':utc_now(),'status':'FROZEN','series':len(data.ids),
        'regime_counts':{str(i):int(np.sum(regime==i)) for i in range(5)},'mechanism_counts':dict(zip(map(str,labels),map(int,counts))),
        'files_sha256':{name:digest(directory/name) for name in files},'policy_sha256':digest(directory/'POLICY.json'),
        'source_sha256':digest(Path(__file__)),'new_fits':0,'shadow_rows':0,'seal_rows':0,'training_feature_use_forbidden':True,
        'old_full_DEV_per_series_tables_loaded':False,'posthoc_mechanisms_not_latent_DGP_ground_truth':True},immutable=True)
    progress('ResearchPool diagnostic metadata frozen','Original H regimes maintained; post-hoc mechanism and scale direction are evaluation-only')


if __name__=='__main__':main()
