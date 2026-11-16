"""Exact score accounting and deletion sensitivity, without refitting or new CIs."""
import numpy as np
from src.scoring.ts_auc import ts_auc
from src.next.bootstrap import PreparedAUC

TIME_BINS=((0,10,'0-9'),(10,25,'10-24'),(25,50,'25-49'),(50,100,'50-99'),
           (100,200,'100-199'),(200,400,'200-399'),(400,None,'400+'))


def time_credits(target,prediction,reference,times):
    score,detail=ts_auc(target,prediction,times,return_details=True)
    old,refdetail=ts_auc(target,reference,times,return_details=True)
    assert [(d['time_online'],d['weight']) for d in detail]==[(d['time_online'],d['weight']) for d in refdetail]
    total=sum(d['weight'] for d in detail);rows=[]
    if not total:raise ValueError('Eligible pair mass required for a time decomposition')
    for low,high,label in TIME_BINS:
        pairs=0;numerator=0.
        for a,b in zip(detail,refdetail):
            if a['time_online']<low or (high is not None and a['time_online']>=high):continue
            pairs+=a['weight']
            if a['weight']:numerator+=a['weight']*(a['auc']-b['auc'])
        rows.append({'time_bin':label,'pair_mass':pairs,'fraction_of_fold_pair_mass':pairs/total,
            'bin_delta':numerator/pairs if pairs else None,'contribution_to_fold_delta':numerator/total})
    np.testing.assert_allclose(sum(r['contribution_to_fold_delta'] for r in rows),score-old,rtol=0,atol=1e-14)
    return rows


def delete_one_scores(target,prediction,times,ids):
    groups,code=np.unique(ids,return_inverse=True);n=len(groups)
    if n<2:raise ValueError('At least two whole-series groups required')
    counts=np.ones((n+1,n),dtype=np.int32);counts[np.arange(1,n+1),np.arange(n)]=0
    values=PreparedAUC.create(target,prediction,times,code,n).evaluate(counts)
    np.testing.assert_allclose(values[0],ts_auc(target,prediction,times),rtol=0,atol=1e-14)
    return groups,values
