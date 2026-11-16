from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6_extension.common import *

session={'started_utc':'2026-09-12T18:20:09+00:00','minimum_continue_until_utc':'2026-09-13T01:20:09+00:00','user_requested_minimum_additional_hours':7,
    'request':'완성될때까지 7시간이상 계속 진행하고 병목을 해결해 분별력 있는 동적 상태 전이망을 만든다.',
    'no_sleep_padding':True,'no_new_learner_recipes_or_outer_score_tuning':True,'status':'ACTIVE'}
save('SESSION.json',session)
paths=[]
for root in (ROOT/'src/next6',ROOT/'artifacts/next6'):
    for p in root.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ('numba_cache','__pycache__','handoff_staging','runtime')):continue
        if p.suffix in ('.py','.json','.csv','.md','.joblib','.zip'):paths.append(p)
save('INITIAL_NEXT6_FREEZE.json',{'created_utc':utc_now(),'files_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in paths},'policy':'Initial NEXT6 code, models, locks and reported statistics remain immutable during extension.'})
plan={
 'created_utc':utc_now(),'status':'PREREGISTERED_EXTENSION_DIAGNOSTICS','adaptive_prior_information':'Initial NEXT6 outer results and null failure modes already known. Extension is exploratory; no pristine confirmatory claim.',
 'data_scope':'Existing RP6400 and original grouped5fold only; no Shadow/Seal/public/cloud/reduced or new competition split.',
 'initial_limits':'No additional supervised detector recipes, blend tuning, promotion or replacement of BL80. One fixed observable-state refinement within the existing factorized-transition family; assess mechanism fidelity, not claim Q6 success.',
 'workstreams':[
  {'id':'D1','question':'Are frozen selected links stable under resampling training series, and do they distinguish post from normal beyond own-state marginals?',
   'training_graph_resamples':200,'folds':[0,1,2,3,4],'seed':2026091301,'method':'Whole-group training bootstrap, rerun identical fixed edge discovery. Original outer graph unchanged. Report support, selection frequency, topology count, coefficient sign and mechanical-family frequency; no new selected detector.'},
  {'id':'D2','question':'Does conditional transition evidence add break discrimination after subtracting own-state evidence and balancing matched windows?',
   'method':'Both fixed models every point. Decompose full log(Ppost/Pnormal) into own-state marginal and conditional-parent excess. Equal series weighting, actual lags, matched absolute-time windows, phase boundaries unchanged. Conditional evaluation only, no threshold fit on outer.'},
  {'id':'D3','question':'Does repeated confirmation use genuinely new raw observations, or merely filter memory?',
   'method':'Paired identical-noise intervention replay; bounded non-overlapping blocks; distinguish fresh innovation, mechanism residual and derived-memory response. Explicit shock/unconfirmed/persistent/recovery states. Single preregistered design and ablations, no supervised head fitting.'},
  {'id':'D4','question':'Can the observable-state refinement reject isolated shocks without suppressing persistent scale/dependence changes?',
   'fixed_synthetic_calibration_partition':30,'fixed_synthetic_evaluation_partition':31,'seeds_per_background_calibration':256,'seeds_per_scenario_evaluation':512,
   'fixed_scenarios':['N0','N1','N2','N3','M1','M2','V1','V2','V3','V4','A1','A2','D1','D2','T1','T2','O1'],
   'extended_interventions':['OB16','repeated isolated shock','temporary scale pulse','return to normal','late persistent change','scale change plus isolated shock'],
   'horizons':[1024,4096],'thresholds':'H-only or independent synthetic calibration only; finite-horizon false-alarm rates, matched detection/delay and recovery censoring. No anytime or calibrated break-probability claim.'},
  {'id':'D5','question':'Is the refined network reproducible, bounded, correctly labeled, and visually faithful?',
   'method':'Actual scalar/batch/prefix/suffix/serialization/interleaving tests; cold/hot latency, exact source hashes, image inspection, raw-free handoff. Existing test suite only rerun where source changes require it.'}],
 'success_conditions':['Separate mechanical predictive links from discriminative relationship changes.','Unconfirmed shock cannot acquire persistence solely from an EWMA echo.','Recovery never resets the true historical break target.','Report false alarms jointly with true detection and censored delays.','No economic causality, no hidden-test certification, no forced promotion.'],
 'design_details_next':'Lock exact bounded state refinement before viewing its evaluation partition. Keep calibration and evaluation source/receipts separate.',
 'minimum_work_until_utc':session['minimum_continue_until_utc']}
save('EXTENSION_PROTOCOL_LOCK.json',plan)
checkpoint('extension initialized','Frozen NEXT6 preserved; bottleneck hypotheses fixed')
