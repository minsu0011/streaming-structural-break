"""Outcome-blind historical clone detection with indexed candidate search."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from src.utils.artifacts import write_json, sha256, utc_now

POLICY = {
    'version':1,'frozen_before_any_score':True,'inputs':['historical values','integer ID for stable representatives only'],
    'forbidden_inputs':['tau','target','online values','online length','prediction','CV score'],
    'exact':'SHA256 of canonical little-endian float64 historical bytes including length',
    'normalized_exact':'SHA256 of full mean/SD-normalized history rounded to 6 decimals, including length; signed zero canonicalized',
    'fingerprint':'128 evenly spaced interpolated normalized points; seeded Gaussian projection to 12 dimensions',
    'candidate_index':'scipy cKDTree, top 8 neighbors per series; no all-pairs distance matrix',
    'projection_seed':20260908,'fingerprint_points':128,'projection_dimensions':12,'neighbors':8,
    'minimum_history_for_near':128,'near_requires_equal_length':True,
    'normalized_rms_threshold':0.01,'normalized_max_abs_threshold':0.10,
    'prefix_rule':'64-value exact prefix is diagnostic only; a common short prefix alone never merges',
    'normalization_floor':1e-12,'constant_rule':'normalized exact groups of same length are merged; absolute location is not identity',
    'nonfinite_rule':'NaN/Inf mask must agree for near merge; normalized replacement uses finite mean',
    'coarse_signature':'length log2 bin, historical kurtosis bin, lag1 bin, zero-scale flag; diagnostic only',
    'component_rule':'connected components, representative minimum integer ID; conservative transitive grouping',
    'limitations':['Top-8 projected neighbors can miss clones; no guarantee all near duplicates are detected.',
        'Distinct realizations from one process are not merged merely for similar statistics.',
        'Different-length clones are not near-merged; identical short prefixes are reported separately.']}

def freeze_policy(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path=directory/'GROUP_POLICY.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != POLICY:
        raise RuntimeError('Grouping policy already frozen; cannot revise using outcomes')
    write_json(path,POLICY)
    (directory/'GROUP_POLICY.md').write_text('# Historical-only grouping policy\n\n'+json.dumps(POLICY,indent=2),encoding='utf-8')
    return sha256(path)

def canonical_history(h):
    h=np.asarray(h,dtype=np.float64).copy()
    h[h==0]=0.0
    h[np.isnan(h)]=np.nan
    return h

def normalize(h):
    h=canonical_history(h); finite=h[np.isfinite(h)]
    mean=float(finite.mean()) if len(finite) else 0.
    sd=float(finite.std()) if len(finite) else 0.
    return (np.where(np.isfinite(h),h,mean)-mean)/max(sd,POLICY['normalization_floor'])

def historical_signature(h):
    h=canonical_history(h); z=normalize(h)
    length=len(h).to_bytes(8,'little')
    rounded=np.round(z,6); rounded[rounded==0]=0.0
    nonfinite=np.where(np.isnan(h),1,np.where(np.isposinf(h),2,np.where(np.isneginf(h),3,0))).astype('uint8')
    exact=hashlib.sha256(length+h.astype('<f8').tobytes()).hexdigest()
    normalized=hashlib.sha256(length+rounded.astype('<f8').tobytes()+nonfinite.tobytes()).hexdigest()
    prefix=hashlib.sha256(h[:64].astype('<f8').tobytes()).hexdigest()
    fingerprint=np.interp(np.linspace(0,max(len(z)-1,0),POLICY['fingerprint_points']),np.arange(len(z)),z) if len(z) else np.zeros(POLICY['fingerprint_points'])
    lag1=float(np.mean(z[1:]*z[:-1])) if len(z)>1 else 0.
    kurt=float(np.mean(z**4)-3) if len(z) else 0.
    coarse=f'{int(np.log2(max(len(z),1)))}:{int(np.clip(kurt,-3,100)//2)}:{int(np.clip(lag1,-1,1)*10)}:{int(np.std(z)<1e-8)}'
    return {'historical_sha256':exact,'normalized_historical_sha256':normalized,'prefix64_sha256':prefix,
            'coarse_historical_signature':coarse},fingerprint.astype(np.float32)

def group_histories(metadata,histories,fingerprints,output_directory):
    """metadata contains historical identity only; labels must never be passed."""
    allowed={'dataset_id','historical_sha256','normalized_historical_sha256','prefix64_sha256','coarse_historical_signature'}
    if set(metadata.columns)-allowed:
        raise ValueError('Nonhistorical metadata passed to grouping')
    directory=Path(output_directory); policy_hash=freeze_policy(directory)
    ids=metadata.dataset_id.to_numpy(dtype=np.int64); parent=np.arange(len(ids)); merges=[]
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j,reason,rmse=None,max_abs=None):
        a,b=root(i),root(j)
        if a==b:return
        if ids[a]>ids[b]:a,b=b,a
        parent[b]=a
        merges.append({'id_a':int(ids[i]),'id_b':int(ids[j]),'reason':reason,'normalized_rmse':rmse,'normalized_max_abs':max_abs})
    for column in ('historical_sha256','normalized_historical_sha256'):
        for indices in metadata.groupby(column,sort=False).indices.values():
            for j in indices[1:]:union(int(indices[0]),int(j),column)
    # One representative per exact/normalized component avoids degenerate huge flat clusters.
    representatives=np.array(sorted(set(root(i) for i in range(len(ids)))))
    rng=np.random.default_rng(POLICY['projection_seed'])
    projection=rng.normal(size=(POLICY['fingerprint_points'],POLICY['projection_dimensions']))/np.sqrt(POLICY['fingerprint_points'])
    vectors=np.asarray(fingerprints,dtype=np.float64)[representatives]@projection
    pairs=set(); checked=0; closest=[]
    # Near matches require equal length, so index within that eligibility stratum.
    # Searching all lengths first would waste the top-k quota on ineligible pairs.
    lengths=np.array([len(histories[i]) for i in representatives])
    for length in np.unique(lengths):
        eligible=np.flatnonzero(lengths==length)
        if len(eligible)<2:continue
        tree=cKDTree(vectors[eligible])
        _,neighbors=tree.query(vectors[eligible],k=min(POLICY['neighbors']+1,len(eligible)),workers=1)
        for row,near in enumerate(neighbors):
            for neighbor in np.atleast_1d(near):
                i,j=int(representatives[eligible[row]]),int(representatives[eligible[neighbor]])
                if i!=j:pairs.add((min(i,j),max(i,j)))
    for i,j in sorted(pairs):
        a,b=histories[i],histories[j]
        if len(a)!=len(b) or len(a)<POLICY['minimum_history_for_near']:continue
        if not np.array_equal(np.isfinite(a),np.isfinite(b)):continue
        checked+=1
        delta=normalize(a)-normalize(b)
        rmse=float(np.sqrt(np.mean(delta*delta))); maximum=float(np.max(np.abs(delta)))
        closest.append((rmse,int(ids[i]),int(ids[j]),maximum))
        if rmse<=POLICY['normalized_rms_threshold'] and maximum<=POLICY['normalized_max_abs_threshold']:
            union(i,j,'near_normalized_full_history',rmse,maximum)
    assignments=pd.DataFrame({'dataset_id':ids,'group_id':[int(ids[root(i)]) for i in range(len(ids))]})
    sizes=assignments.groupby('group_id').size()
    report={'status':'PASS','timestamp':utc_now(),'policy_sha256':policy_hash,'series_count':len(ids),
        'group_count':len(sizes),'non_singleton_groups':int((sizes>1).sum()),'grouped_series':int(sizes[sizes>1].sum()),
        'largest_group':int(sizes.max()),'exact_duplicate_excess':int(metadata.historical_sha256.duplicated().sum()),
        'normalized_duplicate_excess':int(metadata.normalized_historical_sha256.duplicated().sum()),
        'identical_prefix64_excess':int(metadata.prefix64_sha256.duplicated().sum()),
        'coarse_signature_count':int(metadata.coarse_historical_signature.nunique()),
        'indexed_candidate_pairs':len(pairs),'full_history_pairs_checked':checked,'merges':merges,
        'closest_nonexact_candidates':[{'normalized_rmse':x[0],'id_a':x[1],'id_b':x[2],'normalized_max_abs':x[3]} for x in sorted(closest)[:50]],
        'all_pairs_matrix_created':False,'index_strata':'equal historical length, required for eligible near pairs',
        'target_or_tau_used':False,'score_seen':False}
    write_json(directory/'DUPLICATE_AUDIT.json',report)
    assignments.to_parquet(directory/'HISTORICAL_GROUPS.parquet',index=False)
    return assignments,report
