"""Fixed interpretation of held-out orientation evidence, never model selection."""


def opposite_pattern(evidence):
    o=evidence.get('orientation',{})
    a,b=o.get('first',{}),o.get('second',{})
    if a.get('status')!='SUPPORTED' or b.get('status')!='SUPPORTED':return 0
    if a['ci_low']>0 and b['ci_high']<0:return 1
    if b['ci_low']>0 and a['ci_high']<0:return -1
    return 0


def orientation_pass(record,expected=None):
    if record.get('status')!='COMPLETE' or record.get('difference_Holm_p',1)>=.05:return False
    e=record['evidence'];pattern=opposite_pattern(e)
    if pattern==0 or (expected is not None and pattern!=expected):return False
    for reference in ['O1','STATE_ONLY']:
        c=e['contrasts'][record['method']+'__vs__'+reference]
        for metric in ['TS_AUC','logloss_gain']:
            if c[metric]['status']!='SUPPORTED' or c[metric]['ci_low']<=0:return False
    return True


def tabular_comparison(record):
    if record.get('status')!='COMPLETE':return dict(noninferior=False,superior=False)
    c=record['evidence']['contrasts'][record['method']+'__vs__O6_TABULAR']['TS_AUC']
    supported=c['status']=='SUPPORTED'
    return dict(noninferior=bool(supported and c['ci_low']>=-.001),
                superior=bool(supported and c['ci_low']>0),margin=.001)


def domain_replication(records,pattern):
    """Two independent outer folds support; any supported opposite pattern vetoes."""
    good={r['fold'] for r in records if orientation_pass(r,pattern)}
    opposite={r['fold'] for r in records if r.get('status')=='COMPLETE'
              and opposite_pattern(r['evidence'])==-pattern}
    return dict(supported=len(good)>=2 and not opposite,positive_folds=sorted(good),
                contrary_folds=sorted(opposite),minimum_positive_folds=2)
