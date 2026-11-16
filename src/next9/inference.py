"""Exact same-time ranks with paired whole-original-group uncertainty."""
import hashlib
import numpy as np
from src.next8.paired_bootstrap import PreparedRank
from src.next9.movement_screen import grouped_means


def weights_for(groups, partition, draws=2000):
    groups = np.asarray(groups)
    key = hashlib.sha256(('NEXT9_GROUP_BOOTSTRAP_V1|'+str(partition)+'|'+','.join(map(str, groups))).encode()).hexdigest()
    rng = np.random.default_rng(9101604 ^ int(key[:8], 16))
    weights = np.empty((draws, len(groups)), dtype=np.uint16)
    for k in range(draws):
        weights[k] = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
    return weights, dict(draws=draws, seed=9101604, partition=partition,
        group_count=len(groups), resample_key_sha256=key,
        weight_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
        unit='Whole original dependency group; fixed fitted procedure')


def interval(point, samples):
    samples = np.asarray(samples)
    finite = samples[np.isfinite(samples)]
    if not np.isfinite(point) or len(finite) < .95*len(samples):
        return dict(status='INSUFFICIENT_SUPPORT', point=None if not np.isfinite(point) else float(point),
                    ci_low=None, ci_high=None, p_two_sided=1., valid_draws=len(finite))
    centered = finite-point
    return dict(status='SUPPORTED', point=float(point),
        ci_low=float(np.quantile(finite,.025)), ci_high=float(np.quantile(finite,.975)),
        p_two_sided=float((1+np.sum(np.abs(centered)>=abs(point)))/(len(finite)+1)),
        valid_draws=len(finite), centered_bootstrap_test=True)


def rank_samples(y, p, t, ids, all_groups, weights):
    if not len(y):
        return np.nan, np.full(len(weights),np.nan)
    prepared = PreparedRank.prepare(y,p,t,ids)
    columns = np.searchsorted(all_groups, prepared.groups)
    if not np.array_equal(all_groups[columns], prepared.groups):
        raise ValueError('Rank group outside original paired resampling population')
    values, _ = prepared.evaluate(weights[:, columns])
    return prepared.point_auc, values


def compare(y, predictions, times, groups, partition, *, direction_states=None, pair=None, draws=2000):
    y, times, groups = map(np.asarray,(y,times,groups))
    unique = np.unique(groups)
    if not len(unique):
        return dict(status='INSUFFICIENT_SUPPORT', reason='No groups in frozen common range')
    weights, receipt = weights_for(unique, partition, draws)
    points, samples, losses, loss_samples = {}, {}, {}, {}
    for name,p in predictions.items():
        p = np.asarray(p)
        point, boot = rank_samples(y,p,times,groups,unique,weights)
        points[name],samples[name] = point,boot
        if name == 'CURRENT_F_DIRECTION':
            continue  # A raw feature is a rank diagnostic, not a probability.
        prob = np.clip(p,1e-8,1-1e-8)
        ids,mean = grouped_means(-y*np.log(prob)-(1-y)*np.log1p(-prob),groups)
        np.testing.assert_array_equal(ids,unique)
        losses[name] = float(mean.mean())
        loss_samples[name] = weights@mean/weights.sum(axis=1)
    contrasts = {}
    for candidate in predictions:
        if candidate == 'CURRENT_F_DIRECTION':
            continue
        for reference in ('O1','STATE_ONLY','O6_TABULAR'):
            if reference not in predictions or reference == candidate:
                continue
            key=candidate+'__vs__'+reference
            contrasts[key] = dict(
                TS_AUC=interval(points[candidate]-points[reference],samples[candidate]-samples[reference]),
                logloss_gain=interval(losses[reference]-losses[candidate],loss_samples[reference]-loss_samples[candidate]))
    result=dict(status='COMPLETE',bootstrap=receipt,
        TS_AUC={k:None if not np.isfinite(v) else float(v) for k,v in points.items()},
        logloss=losses,contrasts=contrasts)
    if direction_states is not None:
        f=np.asarray(predictions['CURRENT_F_DIRECTION'])
        state=np.asarray(direction_states)
        raw=[]
        for s in pair:
            keep=state==s
            point,boot=rank_samples(y[keep],f[keep],times[keep],groups[keep],unique,weights)
            raw.append((point-.5,boot-.5))
        result['orientation']=dict(first=interval(*raw[0]),second=interval(*raw[1]),
            difference=interval(raw[0][0]-raw[1][0],raw[0][1]-raw[1][1]))
    return result
