"""Assemble explicit BLOCKED_AUTH handoff with no fabricated performance."""
import csv
import json
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,utc_now,commit_id
from src.models.baselines import BASELINES
from src.models.learners import MODEL_NAMES

def write(path,text):
    (ROOT/path).write_text(text,encoding='utf-8')

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def main():
    raw=ROOT/'artifacts/audits/RAW_DATA_MANIFEST.json'
    if raw.exists() and json.loads(raw.read_text(encoding='utf-8')).get('status')=='ACQUIRED':
        from scripts.build_handoff_real import main as real_main
        return real_main()
    access=read('artifacts/audits/DATA_ACCESS.json')
    if access['status']!='BLOCKED_AUTH':raise RuntimeError('This handoff template is for the observed authentication-blocked session only')
    latency=read('artifacts/validation/LATENCY_BENCHMARK.json');audit=read('artifacts/validation/FINAL_AUDIT.json')
    tests=audit['tests']['tests'];state_bytes=latency['feature_state_array_bytes']
    bench=latency['candidates']
    status={'status':'BLOCKED_AUTH','timestamp':utc_now(),'git_commit':commit_id(),'competition':'structural-break-real-time',
        'competition_data_acquired':False,'series_count':None,'break_ratio':None,'scorer_parity':'PASS',
        'leakage_guard':'PASS','tests_passed':tests,'feature_count':57,'feature_families':['A','B','C','D','E','F','Q_missing_flag'],
        'supervised_models_engineering_tested':list(MODEL_NAMES),'competition_cv_performed':False,'champion':None,
        'seal_opened':False,'seal_ids_allocated':False,'reduced_scored_uses':0,'determinism':'PASS_SYNTHETIC_PROTOCOL',
        'official_protocol_smoke':'PASS_SYNTHETIC','full_official_runner_smoke':'PASS_SYNTHETIC_SUBPROCESS','authenticated_crunch_tools_test':'NOT_RUN_AUTH_REQUIRED','submitted':False,
        'requested_run_hours':10,'ten_hour_real_data_research_completed':False,
        'limitation':'Autonomous foundation work completed within available unauthenticated scope; real-data experiments cannot proceed without a valid Crunch setup.'}
    write_json(ROOT/'01_PROJECT_STATUS.json',status)
    write_json(ROOT/'artifacts/validation/DEV_CHAMPION_CANDIDATE.json',{'status':'NOT_SELECTED_AUTH_BLOCKED','PRIMARY_DEV_CANDIDATE':None,'BACKUP_FAST_CANDIDATE':None,'BACKUP_ROBUST_CANDIDATE':None,'dev_cv_ts_auc':None,'reason':'No official development data; engineering B7 default is not a champion','seal_opened':False})
    tables={
        'artifacts/baselines/BASELINE_RESULTS.csv':(['model','fold','ts_auc','mean','median','std','worst_fold','best_fold','status'],[{'model':name,'status':'NOT_RUN_AUTH_REQUIRED'} for name in BASELINES]),
        'artifacts/validation/MODEL_RESULTS.csv':(['model','families','fold','ts_auc','status'],[{'model':name,'families':'ABCD','status':'NOT_RUN_AUTH_REQUIRED'} for name in MODEL_NAMES]),
        'artifacts/validation/CV_RESULTS.csv':(['model','fold','ts_auc','status'],[{'status':'NOT_RUN_AUTH_REQUIRED'}]),
        'artifacts/validation/FEATURE_ABLATIONS.csv':(['families','median_gain','worst_fold_effect','latency_cost','redundancy','status'],[{'families':f,'status':'NOT_RUN_AUTH_REQUIRED'} for f in ['A','B','C','D','AB','AD','BD','ABCD','ABCDE','ABCDF']]),
        'artifacts/validation/TIME_FRACTURE_REPORT.csv':(['model','slice','rows','ts_auc','status'],[{'status':'NOT_RUN_AUTH_REQUIRED'}])}
    for path,(fields,records) in tables.items():
        if (ROOT/path).exists():
            # Never overwrite real research results with blocked placeholders.
            with (ROOT/path).open(encoding='utf-8') as f:
                existing=list(csv.DictReader(f))
            if any(row.get('ts_auc') for row in existing):raise RuntimeError('Refusing to overwrite observed results')
        with (ROOT/path).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    write('00_START_HERE.md',f'''# Structural Break Real-Time — 첫 세션 결과

상태는 **BLOCKED_AUTH**입니다. 대회 전용 코드와 검증 기반은 구축했지만, 실제 대회 데이터는 인증이 없어 확보하지 못했습니다. 공식 API는 파일 목록을 반환했으나 서명된 다운로드 URL을 제공하지 않았고 파일 요청은 HTTP 403이었습니다. 요청한 10시간의 실제 데이터 연구런은 완료하지 못했습니다. 합성 검증을 대회 성능으로 표시하지 않았습니다.

작업 경로: `C:\\Users\\minsu\\Documents\\Structural Break`

## 바로 확인할 결과

* 전체 테스트 **{tests}개 PASS**. 공식 평가기 합성 128사례·256회 비교 최대 오차 **0**.
* 미래 구간 변경·재순회·길이 조회·미리 읽기 차단 검사 PASS. 1,000만 관측값 스트레스 검사 PASS.
* 네 가지 기본 family와 시간척도/누적 증거를 포함한 **57개 streaming feature** 구현. 상태 배열 **{state_bytes}바이트/series**, 관측값 수에 따라 증가하지 않음.
* 공식 EWMA 재현, B0~B7 baseline, 선형/HistGB/LightGBM/XGBoost 후보의 합성 학습·저장·복원 검사 완료.
* 공식 runner 보호 로직을 실제 TCP 소켓으로 실행: 2,880개 예측, 두 번 실행 결과 동일, float32 직접 재생과 동일.
* 전체 공식 runner도 별도 프로세스로 실행: 학습·병렬성 확인·추론·10% 결정성 재검사 성공, 2,400개 예측 인덱스와 공식 형식 검사 PASS.
* 실제 대회 CV, champion 선정, 공식 데이터 기반 ablation은 **미실행**. LOCAL_FINAL_SEAL은 **열지 않음**.

## 요청하신 질문 22개에 대한 답

| 질문 | 현재 답 |
|---|---|
| 1. 대회 데이터 확보? | 아니요. 인증 및 다운로드 서명 부재, HTTP 403. |
| 2. 몇 series? | 실제 확인 불가. 공식 안내의 10,000개는 로컬 감사 결과가 아닙니다. |
| 3. break/no-break 비율? | 실제 측정하지 못했습니다. |
| 4. 데이터의 중요한 특징? | 원본 미확보로 분포·중복·결측·stationarity 감사 미실행. 공식 파일 목록과 용량만 확인. |
| 5. scorer parity? | PASS. 128개 합성 사례, 정렬 후 256회 비교, 최대 오차 0. |
| 6. leakage guard? | PASS. prefix 불변성, API 제한, one-pass, train/validation 분리 검사. |
| 7. split 봉인? | seed 20260908, ID SHA256 80/20 역할 배정, 별도 SHA256 5-fold. 정책 봉인 완료, 실제 ID 배정은 데이터 확보 후. |
| 8. official baseline 점수? | 미측정. 공식 코드 출력 재현만 PASS. |
| 9. 자체 baseline 중 최고? | 판단 불가. B0~B7 구현 완료, 실제 CV 없음. |
| 10. supervised 최고? | 판단 불가. 다섯 모델 엔지니어링 검사만 수행. |
| 11. DEV CV TS-AUC? | 없음. |
| 12. fold별 점수? | 없음. |
| 13. worst fold? | 없음. |
| 14. 가장 유효한 family? | 실제 개발 데이터 없으므로 판단 불가. |
| 15. 쓸모없는 family? | 판단 불가. 결과를 지어내어 제거하지 않음. |
| 16. 추론 지연? | 합성 입력 CPU 중앙값: 수동 {bench['B7_manual']['median_us_per_point']:.1f}µs, logistic {bench['M1_logistic']['median_us_per_point']:.1f}µs, HistGB {bench['M3_histgb']['median_us_per_point']:.1f}µs, LightGBM {bench['M4_lightgbm']['median_us_per_point']:.1f}µs/point. JIT·초기화·통신 제외. |
| 17. determinism? | PASS. 합성 공식 프로토콜 두 번 실행 최대 차이 0. |
| 18. LOCAL_FINAL_SEAL 개봉? | **NO**. 데이터 자체가 없으며 seal feature/예측을 만들지 않았습니다. |
| 19. reduced local test 사용 횟수? | **0회**. 라벨 파일도 내려받지 못했습니다. |
| 20. 가장 큰 병목? | Crunch 인증. 모델 계산보다 데이터 접근이 우선입니다. |
| 21. 다음 첫 3가지? | 안전한 인증·다운로드 → 원본 감사/라벨 검증/ID split 봉인 → baseline 5-fold CV와 사전 정의 모델·ablation. |
| 22. 제출 가능한 형태? | 함수 인터페이스와 합성 공식 프로토콜은 PASS. 인증된 crunch_tools.test·실제 클라우드·제출은 미실행. |

## 이어서 작업하기

`23_REPRODUCTION_COMMANDS.md`에 환경·테스트·재개 명령이 있습니다. 인증은 사용자가 나중에 `scripts/setup_private.py`를 실행해 숨김 입력으로 처리할 수 있습니다. 토큰을 대화, 소스, README 또는 명령행 인자로 기록하지 마세요.

실제 성능이 없는 CSV에는 `NOT_RUN_AUTH_REQUIRED`와 빈 점수만 들어 있습니다. 기본 `main.py`는 B7 엔지니어링 baseline이며 DEV champion이 아닙니다. `21_LIMITATIONS.md`, `22_NEXT_SESSION_PLAN.md`, `artifacts/validation/FINAL_AUDIT.json`도 확인하십시오.
''')
    write('21_LIMITATIONS.md','''# Limitations

1. 실제 원본 데이터 미확보: series 수·break 비율·분포·중복·결측·시간 구조는 미측정. 공식 API 목록만 확보.
2. 라벨은 공식 문서의 t >= tau를 구현했으나 실제 train/reduced 파일을 통한 alignment는 PENDING. 따라서 실제 supervised CV admission은 차단됩니다.
3. split은 알고리즘만 봉인됐으며 실제 ID assignment parquet는 아직 없습니다. Seal은 개봉하지 않았습니다.
4. 모든 현재 학습·지연·프로토콜 수치는 합성 엔지니어링 검사입니다. 실제 CV, feature-family ranking, champion, 경쟁력이 검증된 모델은 없습니다.
5. 공식 double-protection과 TCP transport는 검사했지만 인증된 crunch_tools.test 및 Linux cloud 실행은 하지 못했습니다. Windows에서는 공식 병렬성이 1로 제한됩니다.
6. latency는 정상 상태 CPU 측정이며 historical 초기화, 첫 JIT, 모델 로딩 및 소켓 통신을 제외합니다. 1M/10M 환산은 실클라우드 SLA가 아닙니다.
7. numeric binary HistGB/LightGBM만 compact export합니다. 범주형·다중분류·다른 objective는 거부합니다. sklearn 내부 형식은 버전 의존적이며 원래 예측과 parity를 재검사해야 합니다.
8. V1 feature ablation은 모델 입력 열만 선택합니다. feature kernel은 전체 57개를 계산하므로 family 제거에 따른 실제 계산 절약은 아직 구현하지 않았습니다.
9. 결측/극단값 처리와 20σ clipping은 합성 안전성 검증을 통과한 설계 선택입니다. 실제 데이터 최적 정책으로 검증되지 않았습니다.
10. 온라인 lag product는 dependency proxy이며 정확한 rolling Pearson correlation과 같지 않습니다. AR 기준은 historical-only AR(1) 근사입니다.
11. 문서·결과와 달리 실제 관찰이 없는 점수를 채우거나 reduced/seal을 튜닝에 사용하지 않았습니다. 인증 부재로 요청된 10시간 실제 데이터 연구를 완료하지 못했습니다.
''')
    write('22_NEXT_SESSION_PLAN.md','''# Next session

1. `scripts/setup_private.py`로 공식 계정의 setup token을 숨김 입력하고 대회 데이터를 `data/raw`에 확보한다. 기존 원본은 hash가 다르면 덮어쓰지 않는다.
2. 원본 manifest → 전수 series metadata → 실제 train/reduced tau alignment → 실제 ID split 배정 순서로 검사한다. `scripts/audit_data.py`, `scripts/verify_labels.py`를 실행한다. 중복 series/history가 있으면 fold 간 누수 가능성을 먼저 해결하며 기존 split seed를 쇼핑하지 않는다.
3. `scripts/run_research.py --hours 8`로 사전 정의 baseline와 모델/ablation을 개발 pool에서 실행한다. scorer·label·leakage·split admission 실패 시 원인을 해결한다. LOCAL_FINAL_SEAL은 계속 미개봉.

그 다음: OOF 시간 구간·evidence-age·worst-fold를 분석하고 한 번에 1~2개 가설만 검증한다. 후보의 실제 입력 single-point latency/메모리/저장복원/결정성을 검사한 뒤 median gain, worst-fold loss, latency 기준으로 최대 3개 개발 후보를 정한다. 실제 학습 후의 compact/native parity도 다시 검사한다.

reduced diagnostic은 baseline 1회 + 마지막 후보 1회 이내로 사용 내역을 남긴다. feature/파라미터/모델 선택은 개발 CV만 사용한다. 마지막 30~45분은 source snapshot, 원본 hash 재확인, 공식 tester smoke와 handoff에 남긴다. 사용자와 별도 전략을 정하기 전 LOCAL_FINAL_SEAL을 열지 않는다.
''')
    write('23_REPRODUCTION_COMMANDS.md',r'''# Reproduction commands (PowerShell)

기존 프로젝트 환경은 건드리지 않습니다. 아래 명령은 이 프로젝트 경로에서 실행합니다.

```powershell
Set-Location -LiteralPath 'C:\Users\minsu\Documents\Structural Break'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/validation/pytest.xml
.\.venv\Scripts\python.exe scripts/verify_scorer.py
.\.venv\Scripts\python.exe scripts/verify_engineering.py
.\.venv\Scripts\python.exe scripts/stress_streaming.py
.\.venv\Scripts\python.exe scripts/verify_offline_runner.py
.\.venv\Scripts\python.exe scripts/final_audit.py
```

위 검사는 합성 fixture만 사용하며 실제 CV 점수를 생성하지 않습니다. 모델 소스 또는 feature 코드가 바뀌면 기존 artifact/cache가 거부될 수 있으므로 `verify_engineering.py`로 합성 artifact를 다시 만드십시오. 기존 데이터나 모델 파일을 무조건 삭제하지 마십시오.

실제 데이터 재개:

```powershell
.\.venv\Scripts\python.exe scripts/setup_private.py
.\.venv\Scripts\python.exe scripts/audit_data.py --data data/raw
.\.venv\Scripts\python.exe scripts/verify_labels.py --data data/raw
.\.venv\Scripts\python.exe scripts/run_research.py --data data/raw --hours 8
```

`setup_private.py`는 사용자 실행용이며 토큰을 표시하지 않는 입력창을 사용합니다. 토큰을 코드/명령행/로그에 넣지 않습니다. 업로드나 제출은 수행하지 않습니다. 현재 기본 `main.py`는 B7 엔지니어링 baseline입니다.

인증된 official tester는 실제 데이터를 확보한 뒤 별도 smoke로 실행합니다. 무심코 반복 실행해 reduced 점수 사용 횟수를 늘리지 마십시오. 현재 recorded reduced scored uses = 0.
''')
    mapping={
        '02_OFFICIAL_COMPETITION_CONTRACT.md':'docs/OFFICIAL_COMPETITION_CONTRACT.md',
        '03_RAW_DATA_MANIFEST.json':'artifacts/audits/RAW_DATA_MANIFEST.json','04_DATA_AUDIT.md':'artifacts/audits/DATA_AUDIT.md','05_DATA_AUDIT.json':'artifacts/audits/DATA_AUDIT.json',
        '06_LABEL_CONTRACT.md':'docs/LABEL_CONTRACT.md','07_LEAKAGE_CONTRACT.md':'docs/LEAKAGE_CONTRACT.md','08_SPLIT_LOCK.json':'artifacts/validation/SPLIT_LOCK.json','09_SCORER_PARITY.json':'artifacts/validation/SCORER_PARITY.json',
        '10_BASELINE_RESULTS.csv':'artifacts/baselines/BASELINE_RESULTS.csv','11_FEATURE_CATALOG.csv':'artifacts/features/FEATURE_CATALOG.csv','12_MODEL_RESULTS.csv':'artifacts/validation/MODEL_RESULTS.csv','13_CV_RESULTS.csv':'artifacts/validation/CV_RESULTS.csv',
        '14_FEATURE_ABLATIONS.csv':'artifacts/validation/FEATURE_ABLATIONS.csv','15_TIME_FRACTURE_REPORT.csv':'artifacts/validation/TIME_FRACTURE_REPORT.csv','16_LATENCY_BENCHMARK.json':'artifacts/validation/LATENCY_BENCHMARK.json',
        '17_DETERMINISM_REPORT.json':'artifacts/validation/DETERMINISM_REPORT.json','18_DEPLOYMENT_SMOKE.json':'artifacts/validation/DEPLOYMENT_SMOKE.json','19_DEV_CHAMPION_CANDIDATE.json':'artifacts/validation/DEV_CHAMPION_CANDIDATE.json','20_EXPERIMENT_LEDGER.csv':'experiments/results/EXPERIMENT_LEDGER.csv'}
    for target,source in mapping.items():shutil.copy2(ROOT/source,ROOT/target)
    print('Numbered blocked-auth handoff assembled; no scores invented.')

if __name__=='__main__':main()
