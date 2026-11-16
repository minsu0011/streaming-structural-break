# Moment conditions of the fixed synthetic backgrounds

This is an explanatory calculation after the fixed counterexample results. It changes neither the generator, candidate selection nor any threshold. It uses no competition observations or predictions and performs no fitting.

The locked generator has innovation `u_t = sqrt(omega + alpha*u_(t-1)^2)*epsilon_t`, with independent standardized noise, `E[epsilon^2]=1`, and `omega=(1-alpha)*v`. The surrounding AR recurrence has `|phi|<1`; its target innovation variance is `v=1-phi^2` before a variance intervention. The following claims concern the limiting stationary innovation law, not a perfectly stationary initialization of the finite simulation.

For `alpha<1`, the stationary innovation second moment is `v`. If the standardized noise has finite fourth moment `kappa`, expanding the conditional square gives

`E[u^4] = kappa*(omega^2 + 2*omega*alpha*v + alpha^2*E[u^4])`.

Therefore a finite stationary fourth moment requires `kappa*alpha^2<1`; when it exists, it equals `kappa*v^2*(1-alpha^2)/(1-kappa*alpha^2)`. A negative denominator must never be interpreted as a negative fourth moment: it means this finite-moment solution is inadmissible. The fourth-moment restriction and Gaussian boundary `alpha<1/sqrt(3)` are consistent with Lemma 1 and Remark 2 of [Voutilainen et al., On the ARCH model with stationary liquidity (2021; published online 2020)](https://link.springer.com/article/10.1007/s00184-020-00779-x), taking constant exogenous liquidity to recover ordinary ARCH.

For our Gaussian noise, `kappa=3`. For our variance-standardized Student t noise with `nu>4`, `kappa=3*(nu-2)/(nu-4)`; t3 has no finite fourth moment. These are distributional moments, not fitted sample kurtoses.

| Locked background | Noise | ARCH alpha | kappa*alpha^2 | Limiting innovation variance | Limiting fourth moment |
| --- | --- | ---: | ---: | --- | --- |
| N0 | Gaussian | 0 | 0 | finite | finite |
| N1 | Gaussian | 0 | 0 | finite | finite |
| N2 | Gaussian | 0.65 | 1.2675 | finite | not finite |
| N3 | standardized t5 | 0.4 | 1.44 | finite | not finite |

This gives a precise reason to regard N2 and N3 as demanding tests for raw squared-energy calibration even though their second moments exist. It is compatible with the observed scale-down failure being concentrated in ARCH/tail backgrounds. It does not establish that this moment condition causes every observed model failure, or that competition data share this law.

The simulation starts from a fixed state and burns 1,024 observations. That is approximate initialization, not an exact draw from the stationary law. In particular a finite-horizon Gaussian recursion has finite moments, even when its limiting stationary fourth moment is not finite. All generated arrays are finite numerical samples. The detector also clips some normalized values, so the statement about latent innovations must not be transferred to every bounded detector feature or probability. Numerical lower floors in the generator do not affect these locked positive-omega background parameters.

The existing `SCALE_DOMINANT` DEV label also has a narrower interpretation than a latent variance reduction: it uses an observed 128-point raw-window absolute log-SD effect above 0.4, along with other dominance criteria. Its ideal variance-ratio bounds are approximately 0.4493 and 2.2255. The frozen DEV signature has only 79 observed-decrease and 53 observed-increase break series; neither directional subdivision reaches the registered 100-break support threshold. Across all signatures, 1,608 breaks show an observed decrease, 1,835 an increase, and 501 have too short a post-break window. This is descriptive coverage, with zero new directional model scores.

Evidence: `artifacts/next2/dev_scale_direction/RESULTS.json`, `DIRECTION_COUNTS.csv`, `artifacts/next2/synthetic_moments/RESULTS.json`, and the unchanged `COUNTEREXAMPLE_SUITE_LOCK.json`.
