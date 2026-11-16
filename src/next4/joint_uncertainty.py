"""Joint centered-bootstrap sensitivity of a fixed set of fitted predictions."""
import numpy as np


def describe_joint(draws):
    values=np.asarray(draws,dtype=np.float64)
    if values.ndim!=2 or values.shape[0]<3 or values.shape[1]<1 or not np.isfinite(values).all():
        raise ValueError('Finite observed-plus-bootstrap by candidate matrix required')
    observed=values[0];replicates=values[1:];se=replicates.std(axis=0,ddof=1)
    if np.any(se<=0):raise ValueError('Nonzero empirical uncertainty per candidate required')
    centered=(replicates-observed)/se
    maximum=centered.max(axis=1);critical=float(np.quantile(maximum,.95))
    lower=observed-critical*se
    tail=(1+(maximum[:,None]>=observed[None,:]/se).sum(axis=0))/(1+len(maximum))
    winners=np.argmax(replicates,axis=1);winner_frequency=np.bincount(winners,minlength=len(observed))/len(replicates)
    correlation=np.atleast_2d(np.corrcoef(replicates,rowvar=False))
    eigen=np.linalg.eigvalsh(correlation);participation=float(eigen.sum()**2/np.sum(eigen**2))
    return {'observed':observed,'bootstrap_standard_error':se,'centered_max_95_critical':critical,
        'one_sided_95_joint_lower':lower,'empirical_centered_max_tail_fraction':tail,
        'bootstrap_best_frequency':winner_frequency,'correlation':correlation,
        'correlation_participation_ratio':participation}
