"""Formula/role inventory from frozen fitted artifacts, with no score reads."""
import csv
import re
import json
from itertools import product
import numpy as np
import joblib
from src.next8.common import BASE,read_json,digest
from src.next8.features import CONTEXT_NAMES,NAMES,NORMALIZATION,DETAILED
from src.next8.joint_models import raw_coefficients,preselected_linear_rule

H_FORMULAS=(
    'log1p(len(H)); observed before Online',
    'log(max(1.4826*median(abs(H-median(H))),1e-6))',
    'corr(z_H[1:],z_H[:-1]), denominator floor1e-6; z_H clipped +/-12',
    'corr(z_H[4:],z_H[:-4]), denominator floor1e-6; z_H clipped +/-12',
    'mean(abs(z_H)>3)',
    'abs(mean(z_H^3)); z_H=(H-median(H))/MAD_scale clipped +/-12')
H_KO=('관측된 H 길이의 로그','H의 강건한 척도 로그','H의 1시차 상관','H의 4시차 상관','H의 큰 절대 편차 비율','H의 절대 3차 모멘트')


def coordinate_definition(name):
    if name=='constant_context':return {'formula':'1; training intercept/prior only','parents':[],'current_observation_included':False,'role':'NUISANCE_CONSTANT'}
    if name.startswith('known_H_time_context_'):
        k=int(name.rsplit('_',1)[1]);assert 0<=k<=6
        return {'formula':H_FORMULAS[k] if k<6 else 'log1p(observed Online time t)',
            'parents':[CONTEXT_NAMES[k] if k<6 else 'time_online'],'current_observation_included':False,'role':'OBSERVABLE_CONTEXT'}
    match=re.fullmatch(r'F(\d{3})_(.+)',name)
    if match is None:raise ValueError('Unknown fitted coordinate '+name)
    j=int(match.group(1));tail=match.group(2);f=f'NX8_F{j:03d}'
    if tail.startswith('own_lag'):
        lag=int(tail[7:]);assert lag in (1,4,16)
        return {'formula':f'{f}[t-{lag}]/4; use final normalized H rows before Online zero',
            'parents':[f],'current_observation_included':False,'role':'OWN_OBSERVED_HISTORY'}
    nonlinear=re.fullmatch(r'(stable|variable)_(linear|positive|negative|square)',tail)
    if nonlinear:
        kind=nonlinear.group(2);z=f'{f}[t]/4'
        formula={'linear':z,'positive':f'max({z},0)','negative':f'max(-{z},0)','square':f'({z})^2'}[kind]
        return {'formula':formula,'parents':[f],'current_observation_included':True,
            'role':'FIXED_NONLINEAR_RESPONSE' if kind!='linear' else 'CURRENT_FEATURE_LINEAR',
            'UNKNOWN_gate':'Variable current F is zero if own/H state UNKNOWN in M3/M4/M5; M2 has no conditional gate; actual alias separately audited'}
    state=re.fullmatch(r'(own|graph)_condition_(\d)(_times_F)?',tail)
    if state:
        family,s,interaction=state.groups();s=int(s)
        return {'formula':f'1[{family}_condition(t)=={s}]'+(f' * {f}[t]/4 (zero if own/H UNKNOWN)' if interaction else ''),
            'parents':[f,'own_or_H_condition' if family=='own' else 'FACTORIZED_T3_H1_PREDICTION'],
            'current_observation_included':bool(interaction),
            'role':'CONDITIONAL_FEATURE_EVIDENCE' if interaction else 'STATE_ONLY_NUISANCE',
            'state_timing':'Own F[t-1] or H-only frozen tercile; graph uses own F[t-1], selected parent F[t-2], trained T3 h1 tables'}
    raise ValueError('Unknown fitted coordinate operation '+name)


def fitted_inventory():
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json');assert release['status']=='FROZEN'
    coordinates={};rules=[];scopes=[]
    for p in sorted((BASE/'models').glob('*/DONE.json')):
        done=read_json(p);name=p.parent.name
        assert digest(p)==release['files_sha256'][p.relative_to(BASE).as_posix()]
        selection=read_json(BASE/'h_context_selection'/name/'SELECTION.json')
        scopes.append({'scope':name,'selection_sha256':digest(BASE/'h_context_selection'/name/'SELECTION.json'),
            'stable_features':selection['stable_features'],'variable_features':selection['variable_features'],
            'aliases':done['model_aliases'],'fitting_records':done['fitting']})
        for method in sorted(set(done['model_aliases'].values())):
            path=p.parent/(method+'.joblib');assert digest(path)==release['files_sha256'][path.relative_to(BASE).as_posix()]
            model=joblib.load(path)
            original=model.training['original_input_names'];retained=set(model.names)
            for coordinate in original:
                if coordinate not in coordinates:coordinates[coordinate]={**coordinate_definition(coordinate),'used_in_procedures':[],
                    'retained_in_procedures':[],'model_coordinate':coordinate}
                coordinates[coordinate]['used_in_procedures'].append(name+'/'+method)
                if coordinate in retained:coordinates[coordinate]['retained_in_procedures'].append(name+'/'+method)
        for requested in ('M4','M5'):
            if requested not in done['model_aliases']:continue
            actual=done['model_aliases'][requested];path=p.parent/(actual+'.joblib');model=joblib.load(path)
            coef,intercept=raw_coefficients(model);weights=dict(zip(model.names,coef))
            for j in selection['variable_features']:
                prefix=f'F{j:03d}_';family=selection['orientation_state_specs'][str(j)]
                fixed={kind:float(weights.get(prefix+'variable_'+kind,0.)) for kind in ('positive','negative','square')}
                states=[]
                for own,cross in product(range(4),repeat=2):
                    if actual=='M2':beta=float(weights.get(prefix+'variable_linear',0.))/4.
                    else:beta=preselected_linear_rule(model,j,own,cross)
                    states.append({'own_or_H_state':own,'graph_predicted_state':cross,'linear_beta_per_F':beta,
                        'linear_component_direction':int(np.sign(beta)),'current_F_observed_when_rule_chosen':False,
                        'current_F_contribution_abstains':own==3 and actual!='M2'})
                rules.append({'scope':name,'feature':j,'name':NAMES[j],'requested_method':requested,'actual_alias_method':actual,
                    'model_sha256':digest(path),'training_scope':done['scope'],'normalization':NORMALIZATION,
                    'own_or_H_state_estimator':family,'own_H_condition_constant_within_series':family['kind']!='OWN_F',
                    'graph_state_estimator':'Frozen FACTORIZED_T3_H1 with confident cross-parent requirement' if actual=='M5' else 'NOT_USED_BY_ACTUAL_METHOD',
                    'state_rule_table':states,'fixed_nonlinear_weights_on_F_div4':fixed,
                    'complete_local_derivative_known_before_F':not any(fixed.values()),
                    'complete_local_derivative_formula':'beta + (b_positive if F>0 else -b_negative if F<0 else one-sided derivatives)/4 + b_square*F/8; zero when gated UNKNOWN',
                    'conditional_direction_certification':'NOT_INFERRED_FROM_FITTED_COEFFICIENTS',
                    'future_observed_confirmation_used':False})
    return list(coordinates.values()),rules,scopes


def complete_lineage(coordinates):
    original=BASE/'FEATURE_LINEAGE_NEXT8.csv'
    with original.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==153
    for row in rows:row['inventory_role']='ORIGINAL_NORMALIZED_BANK_COORDINATE'
    common={'family':'OBSERVABLE_DERIVED_CONTEXT','raw_inputs':'Same per-series H and already observed Online prefix',
        'H_fit':'Per-series H for descriptors; population cutpoints only inside corresponding TRAIN',
        'window_or_memory':'H-only constant descriptor or explicitly observed lag/prefix',
        'cumulative':False,'missing_policy':'Finite observations required; explicit UNKNOWN; no target-derived imputation',
        'shared_raw_group':'single_raw_series_all153','normalization_id':NORMALIZATION,
        'classification_scope':'Bookkeeping lineage, not newly searched bank features or additional scientific candidates'}
    for k,name in enumerate(CONTEXT_NAMES):
        rows.append({**common,'feature_id':'NX8_CTX_'+str(k),'original_name':name,'meaning_KO':H_KO[k],
            'formula':H_FORMULAS[k],'current_observation_included':False,'warmup':'Entire H available before first Online point',
            'inventory_role':'H_BACKGROUND_DESCRIPTOR','mathematical_range_NEXT8':'Source formula; H descriptors are not normalized feature values'})
    rows.append({**common,'feature_id':'NX8_TIME','original_name':'time_online','meaning_KO':'지금까지 관측한 Online 시점',
        'formula':'t starts at0; known before x[t], log1p(t) enters declared context','current_observation_included':False,
        'inventory_role':'OBSERVED_TIME_ONLY','mathematical_range_NEXT8':'nonnegative integer; no eventual length'})
    for item in coordinates:
        rows.append({**common,**item,'feature_id':'NX8_MODEL_'+item['model_coordinate'],'original_name':item['model_coordinate'],
            'meaning_KO':'동결된 모델이 사용하는 관측 입력의 변환 또는 상태 상호작용',
            'direct_transform_of':'|'.join(item['parents']),'inventory_role':'FITTED_MODEL_COORDINATE',
            'used_in_procedures':json.dumps(item['used_in_procedures']),
            'retained_in_procedures':json.dumps(item['retained_in_procedures']),
            'warmup':'Original H-seeded causal feature and state histories; no future confirmation',
            'mathematical_range_NEXT8':'Formula of clipped F in [-12,12], F/4 in [-3,3], square <=9, indicators0/1; context separate'})
    return rows


STATE_CONTRACT_KO='''# NEXT8 상태전이 계약

VALUE_SIGN은 H 기준 정규화 F 값의 양·음이다. MOVEMENT_DIRECTION은 H에서 고정한 epsilon을 기준으로 한 증가·유지·감소다. TARGET_ORIENTATION은 TRAIN에서 맞춘 break 점수의 방향 규칙이다. 세 값은 서로 다른 변수이며, 정답을 곱한 부호를 상태 입력으로 만들지 않는다.

물리적 F 상태는 H40 정규화 값의 20%·80% 분위 경계로 LOW/NORMAL/HIGH를 정한다. H<160 또는 경계가 퇴화하면 UNKNOWN이다. 정상 상태 이름은 구조변화가 없다는 정답을 뜻하지 않는다.

시점 t에서 발행하는 h∈{1,4,16} 미래 상태 예측은 F[t]와 허용된 과거 입력을 쓴다. F[t+h]는 해당 보조 문제의 학습·평가 정답으로만 사용한다. 마지막 h개 시점에서 발행한 예측도 보존하되 관측되지 않은 미래를 채워 점수화하지 않는다.

현재 x[t]를 받기 전 M5 조건은 t−1에서 발행한 T3 h1 상태 예측이다. F[t−1]의 자기 상태, 부모 F[t−2]의 상태, TRAIN에서 맞춘 전이표만 사용한다. t=0, 부모 없음, H 상태 불명, 자기 상태 지원 64개 그룹 미만, 유효 부모 요인 없음은 UNKNOWN으로 물러난다. 최대 확률≥0.60 및 1·2위 확률 차이≥0.15를 모두 요구한다. UNKNOWN은 사건 확인으로 기록하지 않는다.

T3는 자기 전이표에 최대 두 부모 조건부 확률 비율을 곱하는 작은 모형이다. 의사계수0.5, 비율 범위[0.25,4], 지원64그룹을 고정했다. 모든 상태의 거대한 결합 전이표를 학습하지 않았다. H 상태 라벨만으로 AR/ARCH/EWMA의 내부 기억이 충분히 표현된다는 Markov 주장은 하지 않는다.

T0 유지·다수, T1 자기 전이표, T2 동일 관측 입력의 제한 깊이 트리, T4 고정 seed의 차수·간선 수 일치 무작위망과 비교한다. T4가 T3와 같거나 빈 망이면 그 비교는 추가 망 가치의 증거가 아니다. proper log loss/Brier, 클래스 비중·recall 및 UNKNOWN 발행률을 함께 보고한다.

OWN_F 조건은 F[t−1] 상태다. H_LAG1/H_TAIL3 조건은 TRAIN H의 3분위 구간이며, TRAIN 범위를 벗어나거나 경계가 퇴화하면 UNKNOWN이다. H 조건은 같은 시계열 안에서 고정되어 있으므로 동적인 상태 전환으로 표현하지 않는다.

M4/M5의 선형 F 계수는 현재 F를 보기 전에 상태에 따라 선택할 수 있다. 고정된 양·음 분리 및 제곱 항은 따로 기록한다. 전체 국소 미분은 현재 F에 의존할 수 있으므로 선형 성분의 사전 선택을 완전한 의미 방향의 사전 판정으로 과장하지 않는다. 실제 사용한 별칭 모델과 UNKNOWN 동작은 별도 계약 감사로 확인한다.

모든 자기·부모·상태 선택과 추정은 해당 TRAIN 안에서 수행한다. meta 입력의 상태도 전체 선택 절차를 제외 그룹으로 cross-fit한다. A 움직임, B 조건부 정답 방향, C 동일 입력 대조군 및 BL80 대비 TS-AUC는 별도로 판정한다. O3 방향 재적합 모형의 인증을 M5 실행 경로의 인증으로 자동 이전하지 않는다.
'''
