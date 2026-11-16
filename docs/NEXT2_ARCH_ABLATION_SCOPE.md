# ARCH ablation scope

Source inspection distinguishes two historical normalization paths. The old
51-column ABCD branch fits its original robust-input AR(8) and ARCH adapter.
The new 26-column energy/forecast branch uses the newer AR(8) innovation and
variance adapter. Their contracts differ and cannot be merged by name.

D208_NO_ARCH sets the new branch's conditional coefficients to zero. The old
ABCD branch still uses ARCH. Its existing five-fold result (0.61729918) must
therefore be described as removal from the new variance branch only.

D218_FULL_PIPELINE_NO_ARCH is a distinct, source-audit-driven control. It keeps
both AR(8) whitening definitions, then removes ARCH normalization from both
branches. The old branch calibrates its features directly on AR residuals.
The new branch uses a fixed historical capped residual second moment, with
zero conditional coefficients and no historical ARCH slope fit. The forecast
channel is constant; the unchanged train-only constant-column remover handles
it. All original screen and full-CV admission gates remain fixed.

Similarly, D205_NO_CUSUM_MEMORY removes the two new-channel CUSUM inputs; it
does not remove every old ABCD CUSUM feature from the scientific definition.
The separate exact tree-used runtime does remove calculations for any old
CUSUM input unused by the frozen fitted forest.
