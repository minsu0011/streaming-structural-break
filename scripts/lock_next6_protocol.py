"""Freeze all scientific choices before NEXT6 relations or validation scores."""
from pathlib import Path
import sys,csv,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6.common import BASE,save,read_json,digest,utc_now,ids_hash
from src.next6.nodes import NAMES,KOREAN,LAGS,lineage
from src.next5.guard import ResearchGuard

def main():
    assert read_json(BASE/'PREFLIGHT.json')['status']=='PASS'
    g=ResearchGuard(ROOT);split=g.split.lock
    inner={}
    for outer in range(5):
        train=[sid for sid in sorted(g.research_ids) if split['mappings']['PRIMARY'][str(sid)]!=outer]
        groups=sorted({g.split.groups[sid] for sid in train},key=lambda gid:(hashlib.sha256(('NEXT6_INNER_20260913:'+str(outer)+':'+str(gid)).encode()).hexdigest(),gid))
        groupfold={gid:i%3 for i,gid in enumerate(groups)}
        inner[str(outer)]={str(sid):groupfold[g.split.groups[sid]] for sid in train}
    protocol={'status':'FROZEN_BEFORE_NEXT6_SCORES','created_utc':utc_now(),'data_scope':'EXISTING_NEXT5_RP_6400_ONLY',
        'research_split_sha256':g.split.digest,'shadow_lock_sha256':g.shadow_lock_sha,
        'outer_mapping':split['mappings']['PRIMARY'],'inner_mapping':inner,'inner_folds':3,
        'phase_intervals':{'pre':[-32,-1],'onset':[0,7],'earlyPost':[8,31],'establishedPost':[32,127],'laterPost':[128,None]},
        'target':'int(t>=official_tau_index) if tau exists else 0','phase_usage':'TRAIN analysis/fit labels or retrospective evaluation only; never inference',
        'nodes':list(NAMES),'node_selection':'16 fixed from source lineage before score; no supervised node pruning','lags':list(LAGS),
        'H_calibration':{'fit_fraction':.6,'minimum_H':96,'pseudo_burn':'min(32,len(pseudo)//4)',
            'state_quantiles':[.2,.8],'normalizer':'median / max(IQR/1.349,.05)','clip':12,'degenerate_state':3},
        'graph':{'method':'small ridge conditional predictive gain; separate normal/post maps',
            'max_nodes':16,'max_cross_edges':24,'max_parents':2,'ridge':.05,
            'own_lags':list(LAGS),'common_controls':['raw_energy_lag1','log_variance_forecast_lag1','log1p(current_online_time)'],
            'sampling':'every 8th actual time, capped first128 sampled points per series per phase; true lag rows retained',
            'series_weights':'equal total weight per series within label condition',
            'selection':'two grouped internal halves; both held-out relative MSE gains >=.005, same cross-coefficient sign, per-half positive-series fraction >=.55 and support >=32',
            'ranking':'max(normal,post) minimum two-half gain; deterministic tie ordering (target,source,lag)',
            'pair_menu_count':1200,'cross_edge_mechanism':'MECHANICAL/DERIVED where formula or shared filter explains; predictive not causal',
            'uncertainty':'whole-series paired loss distributions; multiplicity is diagnostic only'},
        'transition':{'factorization':'P(S_j(t)|S_j(t-1), up to2 selected lagged parents)',
            'smoothing':.5,'min_context_series':8,'fallback':'own-history then unconditional Dirichlet smoothing',
            'unknown_state':'skip unknown transition and mark missing; never low','normal_and_post':'both scored at inference',
            'calibrated_probability_claim':False},
        'finite_candidates':['M2_RELATION','M3_TRANSITION','M4_COMBINED_IF_INNER'],
        'comparators':['M0_MATCHED_BL80','M1_TABULAR_SAME_INPUT','OWN_HISTORY_ONLY','RANDOM_EDGES_SAME_SIZE','BASE_ONLY_TABULAR','AMPLITUDE_0.9_M0'],
        'model_inputs':'M1=unchanged J102+C205 features +16 H-relative nodes,current/lag1,2,4,8,16/delta/EWMA8,32; M2/M3 add fixed-schema relation/transition summaries',
        'learner_params':read_json(ROOT/'artifacts/next5/references/POLICY.json')['learner_params'],
        'weighting':'W3 fitted training only','crossfit':'Graph selection + graph parameters refit excluding each inner holdout; head training uses only those out-of-fold graph features',
        'M4_inner_gate':'enable only if mean inner TS-AUC exceeds BOTH M2 and M3 by >=.0002; decision before any outer score',
        'outer_score_access':'freeze all fold procedures and M4 eligibility before batch outer-score release; no score-adaptive changes',
        'promotion_gate':{'mean_delta_vs_M0_min':.0012,'mean_delta_vs_M1_min':.0002,'mean_delta_vs_random_min':.0002,
            'paired_CI_lower_vs_M0':0,'paired_CI_lower_vs_M1':0,'positive_matched_folds_min':4,
            'minimum_matched_fold_delta':-.001,'causal_streaming':'ALL_PASS','latency_vs_M0_max':3,
            'role':'PROMOTION_ELIGIBLE_DEV_ONLY; no overwrite or hidden-test certification'},
        'bootstrap':{'draws':1000,'seed':20260913,'unit':'whole series, stratified by fixed outer fold','conditional_on_fitted_predictions':True},
        'synthetic':{'existing_generator':'src.next2.counterexamples.generate','seed_lock_sha256':digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
            'calibration_seeds_per_background':64,'evaluation_seeds_per_scenario':64,'horizon':1024,'long_null_horizon':4096,
            'cases':['N0','N1','N2','N3','scale_increase','scale_decrease','dependence_change','O1','repeated_outlier','late_genuine_break'],
            'raw_null':'actual same engine applied to fixed stationary Gaussian/AR/ARCH/heavy-tail streams',
            'prebreak_trap':'paired control/abrupt-break prefix equality; no strong pre-signal claim',
            'amplitude_control':.9},
        'normal_comparisons':{'primary':['no-break','not-yet-break at same absolute online time'],
            'H_matching':'predefined H-length/scale/tail/lag descriptors, training-only bins',
            'pseudo_event':'32 + sha256(NEXT6_PSEUDO:group_id) mod224; independent of target and final length'},
        'case_selection':'within fixed fold0 ID-hash order select one success, delayed, false alarm, miss using training-calibrated alarm; unavailable category reported missing',
        'forbidden':['Shadow','SEAL_A','SEAL_B','SEAL_C','SEAL_D','reduced test','public/cloud test'],
        'adaptive_reuse':'RP used historically; nested procedure removes neither past reuse nor design adaptivity',
        'time_budget_seconds':36000,'audit_budget_seconds':[2700,3600],'maximum_configurations':8,'new_orchestrator':False}
    save('NEXT6_PROTOCOL_LOCK.json',protocol)
    with (BASE/'FEATURE_LINEAGE.csv').open('x',encoding='utf-8-sig',newline='') as f:
        rows=lineage();w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (BASE/'FEATURE_STATE_CATALOG.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['node','feature_name','korean_meaning','low','normal','high','missing','threshold_fit_scope'])
        for i,(n,k) in enumerate(zip(NAMES,KOREAN)):w.writerow([i,n,k,'raw<q20','q20<=raw<=q80','raw>q80','state3; degenerate or missing','first60% H engine fit; last40% H calibration'])
    (BASE/'EVENT_LABEL_CONTRACT.md').write_text('# NEXT6 라벨 계약\n\n공식 tau_index만 사용한다. target[t]=int(t>=tau), tau가 없으면 전부0. isOnset/relativeTime/phase는 학습 fold 분석 또는 사후 평가에만 사용한다. 실제 inference API는 H와 현재 scalar point만 받는다. 진짜 tau에서 상태를 reset하지 않는다. pre[-32,-1], onset[0,7], earlyPost[8,31], establishedPost[32,127], laterPost[128,+∞]를 점수 공개 전에 고정했다. 없는 구간은 결측/support로 남기며 패딩하지 않는다.\n',encoding='utf-8')
    print('NEXT6 protocol frozen',digest(BASE/'NEXT6_PROTOCOL_LOCK.json'),flush=True)

if __name__=='__main__':main()
