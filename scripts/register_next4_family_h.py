"""One fixed small temporal-state readout; no recurrent neural-network search."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next4.temporal_state import implementation_hash,feature_names
from src.next4.state_readout import source_hash
from src.next4.scheduler import register_job


def checked_xml(path,count):
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==count and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    return digest(path)


def main():
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    ids=read_json(ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json')['families']['H']['candidate_ids'];assert len(ids)==1
    candidate=ids[0];settings={'version':'coupled_linear_state_v1','kind':'two_pole_readout'}
    params={'learner':'fixed_state_logistic','weighting':'W3','ridge':.01,'maxiter':200,'frozen_BL80_weight':.9}
    execution=['scripts/evaluate_next4_state.py','src/next4/cache.py','src/next4/frozen.py','src/next4/state_readout.py',
        'src/next2/blend.py','src/next/data.py','src/next/guard.py','src/scoring/ts_auc.py']
    menu={'status':'FROZEN_BEFORE_FIRST_FAMILY_H_REAL_SCORE','created_utc':utc_now(),'family':'H','candidate_ids':ids,
        'transition':'For each channel/pole, u_t=(1-alpha)*u_prev+alpha*x_t; v_t=(1-alpha)*v_prev+alpha*u_t',
        'transition_parameters_fitted':False,'readout_parameters_fitted':33,
        'state_structure':'Four channels times three fixed poles times two coupled scalar states; nine shared finite-prefix white-input variance/covariance values',
        'input_channels':['ARCH-normalized innovation','squared innovation','lag1 innovation product','signed square-root magnitude'],
        'H_calibration':'Immutable H channel means/stds, floor 1e-6; input normalized and clipped at +/-8',
        'pole_halflives':[8,32,128],'feature_count':32,
        'features':'tanh of each first/second pole normalized by its finite-prefix unit-white-input SD; four tanh current channels and four tanh short-long second-pole contrasts',
        'variance_recursion_claim':'Exact filter covariance under independent unit-variance input, used as an empirical feature scale; no calibrated real-world test claim',
        'second_pole_impulse':'alpha**2*(k+1)*(1-alpha)**k; delays and spreads transient evidence relative to a one-pole EWMA',
        'learner':'Ridge logistic readout, 32 coefficients plus unpenalized intercept, W3 training-only weights, L-BFGS-B maxiter200/ftol1e-12/gtol1e-8/maxls30',
        'parameters':params,'fixed_combination':'0.9 frozen BL80 + 0.1 temporal-state expert, float64 sum rounded float32',
        'screen_gate':{k:policy['screen'][k] for k in ['mean_delta_min','each_fold_delta_min']},
        'stop_rule':'Exactly one fixed state-space/readout recipe. No pole, channel, ridge, nonlinear map, weighting or blend-weight variants after scores. Failed screen stops.',
        'novelty':'Coupled two-stage temporal states retain a distinct delayed impulse response; a small train-fold linear readout replaces the usual tree combination. This is a fixed state transition with learned readout, not a trained recurrent transition, GRU, LSTM or Transformer.',
        'prior_dedup_sources':{p:digest(ROOT/p) for p in ['src/next/channel_evidence.py','src/next/serial_channel_evidence.py',
            'src/next2/complement_bank.py','src/next4/sequential.py']},
        'state_test_xml_sha256':checked_xml(BASE/'maintenance/FAMILY_H_STATE_TESTS.xml',6),
        'readout_test_xml_sha256':checked_xml(BASE/'maintenance/FAMILY_H_READOUT_TESTS_CORRECTED.xml',3),
        'tests':'100 raw-feature prefix/streaming cases, direct convolution and covariance oracle, fixed finite state at 1024/4096/16384, weighted gradient check, scalar/batch/serialization parity and constant-score TS-AUC=0.5 control',
        'unit_fixture_failure_preserved':'artifacts/next4/maintenance/H_constant_fixture_001/FAILURE.json',
        'prototype_synthetic_readout_unit_fits':1,'real_H_fits_before_registration':0,'real_H_quality_scores_before_registration':0,
        'mandatory_fitted_score_diagnostics':'1024/4096/16384 long-null and isolated/burst-outlier memory; finite feature tests do not replace these',
        'feature_source_sha256':implementation_hash(settings),'model_source_sha256':source_hash(),
        'evaluation_runners':{candidate:'scripts/evaluate_next4_state.py'},'dispatch_priority':90,'policy_sha256':digest(policy_path)}
    menu_path=BASE/'families/H_MENU_LOCK.json';atomic_json(menu_path,menu,immutable=True)
    hyp={'experiment_id':candidate,'parent':'BL2_C205_DEPENDENCE_80','family':'H',
        'hypothesis':'A small stable coupled temporal state with a fitted linear readout can provide persistence evidence beyond the existing tree-combined statistics.',
        'causal_mechanism':'Current scalar residual updates bounded H-calibrated states; readout labels belong only to the outer training groups.',
        'exact_changes':{'state':'fixed coupled two-pole filters','learner':'33-parameter weighted logistic readout','independent_base_score_inputs':False},
        'parameters':params,'candidate_menu_id':'H_MENU_V1','expected_gain':'Positive fixed-blend official TS-AUC from delayed persistent evidence at small state cost.',
        'risk':'Temporal summaries may remain redundant; linear readout can underfit mechanisms; delayed response and outlier persistence can hurt early or long-null behavior.',
        'screen_gate':menu['screen_gate'],'full_gate':policy['performance'],'stop_rule':menu['stop_rule'],
        'created_before_score':True,'created_utc':utc_now(),'feature_module':'src.next4.temporal_state','feature_settings':settings,
        'feature_names':feature_names(settings),'feature_sha256':implementation_hash(settings),'model_source_sha256':source_hash(),
        'execution_source_sha256':{p:digest(ROOT/p) for p in execution},'family_menu_sha256':digest(menu_path),
        'policy_sha256':digest(policy_path),'previous_failure_distinction':menu['novelty']}
    atomic_json(BASE/'registrations'/(candidate+'.json'),hyp,immutable=True)
    register_job({'id':'H001_SCREEN','kind':'research','family':'H','candidate_id':candidate,'priority':90,
        'argv':['scripts/evaluate_next4_state.py','--candidate',candidate,'--scope','screen'],
        'description':'Single frozen coupled-state logistic readout expert, no recurrent-network search','retry_safe':False})
    progress('Family H registered','One small fixed temporal-state readout; source/feature/optimizer contracts frozen after independent filter and weighted-loss tests')
    print('NEXT4 FAMILY H FROZEN screen job queued',flush=True)


if __name__=='__main__':main()
