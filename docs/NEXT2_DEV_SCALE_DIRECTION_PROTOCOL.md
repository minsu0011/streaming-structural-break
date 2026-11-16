# Observed DEV scale-direction coverage

This diagnostic checks how much real DEV evidence supports directional scale claims. It does not fit or score a model and cannot nominate, blend or promote a candidate.

Reuse the frozen mechanism windows and existing DEV-only signature allocation. The original statistic records the absolute log post/pre standard-deviation ratio; reconstruct its sign from exactly the same at-most-128-point windows and verify its absolute value against the frozen signature. Keep the original minimum of 32 observations per side. All source series must belong to the frozen 8,000 DEV IDs; no seal value may be read.

Report aggregate counts and signed-effect quantiles by existing signature, existing H regime and original fold. The directions are negative, positive or exactly zero log SD ratio. Insufficient-window cases and no-break controls have undefined direction. Do not write a new per-series target or observation file. State explicitly that an observed raw-window SD ratio is a diagnostic proxy, not the latent innovation-variance change or a known competition mechanism label.

The old SCALE_DOMINANT magnitude threshold is absolute log SD ratio 0.4. Its ideal variance-ratio boundaries are exp(-0.8) and exp(0.8); the synthetic latent variance factors 0.7 and 0.5 cannot automatically be identified with that slice. Counts below the existing 100-break support convention must not be used to claim a well-supported directional mechanism gain. No new prediction analysis or score-based subgroup selection is authorized by this protocol.
