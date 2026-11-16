from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6_extension.common import *
save('FRESH_STATE_DESIGN_LOCK.json',{
 'created_utc':utc_now(),'designs':1,'family':'bounded factorized observable-state refinement; not an additional supervised detector recipe',
 'motivation':'Current NEXT6 normal/post state ratios may mostly encode marginal shifts; recursive filters can count one impulse repeatedly.',
 'rank_reference':'First60% H empirical midrank CDF using existing src.next.normalization.ecdf_uniform; fixed throughout online. No online refit, no AR recursion, all finite ranks bounded.',
 'nodes':['signed rank u','rank energy u^2','rank tail I(abs(u)>.9)','rank asymmetry u^3','rank dependence lag1','rank dependence lag4','rank energy dependence lag1','rank energy dependence lag4'],
 'node_count':8,'node_origin':'Existing ECDF/rank/dependence primitives reorganized into a finite-memory diagnostic representation; not identical coordinates of the original NEXT6 model.',
 'rank_definition':'u=2*midrank_CDF(x;H_first60%)-1; energy products centered by mean square rank in the H reference',
 'block_size':16,'block_policy':'Disjoint current-observation blocks anchored at online t=0; lag4 channel footprints overlap by up to4 raw points, so blocks are not claimed independent.',
 'H_calibration':'Replay last40% H out-of-sample against first60% CDF; block means; center=median, spread=max(IQR/1.349,.02). Carry only last4 ranks and previous completed H block state into online; new online block sums and confirmation memory start empty.',
 'minimum_H':512,'constant_H':'INSUFFICIENT_REFERENCE state, not normal evidence','node_states':'0 if normalized block mean < -2;2 if >2;1 otherwise;3 missing/insufficient reference. Ties normal.',
 'persistent_evidence':'For each node, minimum absolute normalized value of4 consecutive completed blocks if all4 signs agree; max across8 nodes. No same-block repeated confirmation. Incomplete blocks never count.',
 'calibration':'Pooled4 backgrounds x256 independent partition30 streams,1024 horizon; alpha.05 finite-rank ceil((n+1)*.95) of per-stream maximum persistent evidence, strict >. Freeze before partition31 evaluation.',
 'raw_shock':'abs(x-H_median)>8*max(H_MAD*1.4826,.01*H_std,1e-8); tag at current point with16-point cooldown; cannot alone create persistent state.',
 'states':['WARMUP','NORMAL','SHOCK','UNCONFIRMED','PERSISTENT','RECOVERING','INSUFFICIENT_REFERENCE'],
 'recovery':'When persistent evidence ceases, at most2 completed blocks of RECOVERING; then NORMAL/UNCONFIRMED based fresh block. No cumulative max and no reset of official breakOccurred target.',
 'graph':'At most8 cross links, at most1 parent per target, only lag1 completed block =16raw points. Exclude same family candidate edges as an interpretability restriction; all still share one raw source.',
 'graph_null':'Per own previous state, IPF loglinear null preserves phase-target, phase-parent and pooled target-parent marginals. Full phase-specific conditional tables allow phase-parent-target interaction. Compare held-out proper log probability, controlling changing marginals.',
 'graph_training':'Outer training only, original grouped5fold. Original hash-group2 halves for edge selection. Equal series-phase counts total64 pseudo-count per series; Dirichlet.5 each cell. Full and IPF-null fitted on each internal training half; evaluate other half.',
 'graph_eligibility':'Average normal/post per-series full-minus-null log probability gain >=.001, positive series fraction>=.55 in both internal halves; selected descending minimum-half gain. Minimum32 train/eval groups each phase. Context support <8 unique series backs off to own previous-state probabilities in both full and null.',
 'IPF':'Maximum1000 iterations, relative marginal max error<=1e-10; fail explicitly on nonconvergence.',
 'block_labels':'Train only complete16-point blocks and previous block wholly inside same official binary phase; skip straddling blocks. tau never an inference argument.',
 'output':'Observable state and evidence, plus full/null relationship log-ratio and excess calculated simultaneously. Not a calibrated break probability. Graph quantities are explanatory channels; no learned blend with BL80.',
 'evaluation':'Frozen partition31 + original outer folds exploratory only; report matched null/detection/delay/recovery and mechanism support. No changing this design after evaluation, no forced edge or positive conclusion.'})
checkpoint('fresh state design locked','One fixed finite-memory refinement; evaluation still unopened')
