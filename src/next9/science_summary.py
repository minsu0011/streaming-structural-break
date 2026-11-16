"""Equal-original-fold contrasts and precommitted model promotion criteria."""
import numpy as np
from src.next9.inference import interval

CONTRASTS=(('M1','M0'),('M2','M1'),('M3','M2'),('M4','M3'),('M4','M2'),
           ('M5','M4'),('M6','M5'),('M7','M0'),('M6','TABULAR_SAME_INPUT'),('M5','TABULAR_SAME_INPUT'))


def aggregate_contrast(records,samples,candidate,reference,*,major_only=False):
    kept=[]
    for record,bootstrap in zip(records,samples):
        a,b=record['points'][candidate]['TS_AUC'],record['points'][reference]['TS_AUC']
        if a is None or b is None or (major_only and not record['major_domain_supported']):continue
        kept.append((record['fold'],float(a-b),np.asarray(bootstrap[candidate])-np.asarray(bootstrap[reference])))
    if not kept:return dict(status='INSUFFICIENT_SUPPORT',candidate=candidate,reference=reference,folds=[],point=None,ci_low=None,ci_high=None)
    points=[r[1] for r in kept]
    result=interval(float(np.mean(points)),np.mean(np.stack([r[2] for r in kept]),axis=0))
    return dict(**result,candidate=candidate,reference=reference,folds=[r[0] for r in kept],fold_deltas=points,
        positive_folds=sum(v>0 for v in points),worst_fold=min(points),original_equal_fold_weight=True,
        group_bootstrap_conditional_on_frozen_fitted_procedures=True)


def score_promotion(normal,early,domains):
    core=(normal.get('status')=='SUPPORTED' and len(normal['folds'])==5 and normal['point']>=.0012
          and normal['ci_low']>0 and normal['positive_folds']>=4 and normal['worst_fold']>=-.001)
    early_ok=early.get('status')=='SUPPORTED' and len(early['folds'])==5 and early['point']>=-.001
    failures=[];supported=[]
    for item in domains:
        r=item['contrast']
        if len(r.get('folds',[]))<2 or r.get('status')!='SUPPORTED':continue
        supported.append(dict(mode=item['mode'],domain=item['domain']))
        if r['point']<-.001 or r['worst_fold']<-.005:
            failures.append(dict(mode=item['mode'],domain=item['domain'],mean=r['point'],worst=r['worst_fold']))
    return dict(core_score_gate=bool(core),early_gate=bool(early_ok),major_domain_failures=failures,
        supported_major_domains=supported,score_gates_pass=bool(core and early_ok and supported and not failures),
        runtime_leakage_reproduction_still_required=True,canonical_role_changed=False)
