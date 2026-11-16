"""Fixed elapsed-time/H domains; all H terciles fitted inside original TRAIN."""
import numpy as np

TIME_BINS=((0,9),(10,24),(25,49),(50,99),(100,199),(200,399),(400,None))
H_BINS=((0,1249),(1250,1749),(1750,2499),(2500,3999),(4000,None))


def fit_domains(bank,scope):
    keep=np.isin(bank.ids,list(scope.training_ids));scope.admit_training(bank.ids[keep])
    result=dict(scope=scope.provenance(),terciles={})
    for name,k in [('H_LAG1',2),('H_TAIL3',4)]:
        values=bank.series['context'][keep,k]
        finite=values[np.isfinite(values)]
        cuts=np.quantile(finite,[1/3,2/3]).tolist() if len(finite) else [0.,0.]
        result['terciles'][name]=dict(column=k,cutpoints=cuts,available=bool(len(finite) and cuts[1]>cuts[0]),training_series=int(keep.sum()))
    return result


def masks(bank,rows,spec):
    rows=np.asarray(rows,dtype=np.int64);t=bank.times[rows];s=bank.series_index[rows]
    yield 'ALL','ALL',np.ones(len(rows),dtype=bool)
    for family,values,bins in [('TIME',t,TIME_BINS),('H_LENGTH',bank.series['H_length'][s],H_BINS)]:
        for k,(low,high) in enumerate(bins):
            yield family,f'{family}_{k}',(values>=low)&(True if high is None else values<=high)
    for family,r in spec['terciles'].items():
        values=bank.series['context'][s,r['column']]
        state=np.digitize(values,r['cutpoints'],right=True) if r['available'] else np.full(len(values),3)
        state=np.where(np.isfinite(values),state,3)
        for k in range(4):yield family,f'{family}_{k}',state==k
