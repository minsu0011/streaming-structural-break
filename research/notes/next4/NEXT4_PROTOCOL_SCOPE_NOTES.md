# NEXT4 실행 순서와 해석 범위

`configs/next4/RESEARCH_POLICY_LOCK.json`, 각 family menu와 candidate registration이 수치 기준과 후보 구성을 결정한다. 해당 파일은 점수를 본 뒤 완화하지 않았다. 모든 실제 screen/full 학습은 원래 grouped partition을 사용했고, failed screen을 full로 올리지 않았다. E/G2의 meta 학습은 원래 outer train 내부의 honest inner OOF를 사용한다.

사용자 명세의 이상적인 stage 순서는 bootstrap → ALT → H holdout → mechanism이다. 실제 실행에서는 full/paired 결과가 나온 후보의 고정 mechanism·time·diversity 진단 일부를 ALT/H보다 먼저 수행했다. 따라서 모든 후보가 그 이상적인 stage 순서를 그대로 거쳤다고 보고하지 않는다. 이 선행 진단은 이미 등록한 scope로 만든 평가 결과이며 모델 입력이나 새 feature 선택에 사용하지 않았다. B1의 supported SCALE 개선을 확인한 뒤 기존 ALT/H 20개 refit을 추가로 등록한 것이 대표 사례다. Mechanism 분석에서 통과했다는 사실만으로 승격하지 않고, 별도 나머지 검증 결과를 요구한다.

성능 기준을 통과하지 못한 모든 모델에 ALT/H를 무조건 반복 적합하지는 않는다. B2는 원래 다섯 fold가 모두 개선되고 mean 기준을 간신히 넘은 동일 후보의 분할 민감도를 확인하기 위해, B1은 supported SCALE 강건성 기준을 통과한 후보를 확인하기 위해 각각 고정된 20개 ALT/H refit을 등록했다. 기존 frozen J102/C205는 추가 적합하지 않는다. 이 ALT/H도 같은 DEV를 재분할한 것이므로 새 독립 데이터의 검증으로 부르지 않는다.

A2/C3/D2/D3도 original SCALE CI와 overall 손실 기준을 통과했으므로, 이 네 후보의 합성 결과를 보기 전에 추가 검증 조건을 고정했다. 이미 등록한 fold-0 합성 suite에서 ARCH 감소 두 경우와 heavy-tail 감소 두 경우가 각각 개선될 때만 동일 후보의 기존 ALT/H 20개 refit을 수행한다. 조건 미달이면 해당 경로를 닫는다. 최대 추가 80 fit이며 새로운 full candidate identity, split, seed 또는 weight를 만드는 절차가 아니다. 모든 기준을 통과한 robustness 후보가 여러 개면 원래 SCALE CI lower bound, primary gain, candidate ID 순서로 하나를 고르는 규칙도 합성 결과 전에 고정했다.

F1의 72/18/10 조합은 A/C/D/H에서 이미 사용한 0.9 BL80 + 0.1 expert와 같다. 같은 조합을 새 후보로 다시 세거나 70/20/10으로 조금 바꾸지 않았다. F2는 먼저 고정한 독립 expert admission을 통과한 구성요소가 있을 때만 허용한다. 그 조건을 만족하지 않으면 실제 F2 점수와 추가 fit 없이 conditional skip을 기록한다.

합성 평가에는 training scope를 일치시킨 별도의 NEXT4 diagnostic policy를 사용한다. B2는 필요한 한 번의 all-DEV 진단 적합과 canonical all-DEV BL80을 비교한다. 다른 A/B/C/D/H 후보는 원래 full 또는 탈락한 screen의 fold 0 모델과 같은 범위의 frozen J/C/BL을 비교한다. Fold-0의 calibration 512·evaluation/long-null 256과 all-DEV의 1000·512를 구별한다. 원래 NEXT2 all-DEV 합성 결과를 fold-0 결과의 참조값으로 섞지 않는다. 낮은 오경보율에 대한 표본 정밀도와 장기 threshold 누적 효과를 결과에 함께 남긴다.

B2의 all-DEV 진단 적합은 기존의 검증된 feature cache를 사용했다. 이것을 fresh raw reproduction으로 세지 않는다. 실제 새 후보를 승격한다면 별도 fresh output에서 raw DEV → feature → fit → inference 재현을 수행해야 한다. 승격 후보가 없으면 frozen 모델을 다시 학습하지 않는다.

B2의 direct API는 DEV 8,000개·4,014,405점에서 기존 feature 계산과 정확히 일치했고 무작위 prefix 300건을 통과했다. 이 사실은 인과적 구현 검증이며 원래 primary CI 실패를 바꾸지 않는다. 속도 비교는 같은 머신에서 BL80/B2 순서를 교대한 다섯 쌍의 실제 1,000만 scalar update로 수행한다. 다른 project Python 작업은 그 동안 실행하지 않고, 필요한 queue/board metadata만 PowerShell로 기록한다. 과거 NEXT2의 다른 시점 속도 수치와 현재 측정을 섞어 ratio를 계산하지 않는다.

개발 중 unit fixture나 비교 정밀도 문제는 이전 소스·실패 XML·수정 이유를 `artifacts/next4/maintenance/`에 보존한다. 실패한 unit test의 수, loop 안에서 수행한 무작위 prefix case의 수, 실제 learner fit의 수를 서로 구별한다. 합성 prototype fit과 label-permutation control fit도 실제 후보 적합 횟수와 별도로 기록한다.

NEXT3는 사용자 중단 상태를 유지한다. 중단 당시 가장 최신의 whole-suite 시도 1047개는 완료되지 않았고, 이전 완료 whole-suite는 1019 PASS다. NEXT4 검증 결과로 이 과거의 미완료 시도를 소급해서 PASS로 바꾸지 않는다.

원장 통합에서는 NEXT4 시작 당시 root 213행과 NEXT3의 미게시 `prepared_v2` 242행을 문자열 단위로 비교했다. 앞 213행은 완전히 일치했다. 따라서 기존 242행의 값을 그대로 보존하고, 추가 열로 NEXT3가 사용자 중단 상태였다는 사실을 표시한다. E003의 기존 provisional 문구를 최종 승격으로 바꾸지 않는다. NEXT4 후보의 screen/full 적합, ALT/H 적합, all-DEV 진단 적합, permuted-label 적합은 primitive receipt에서 서로 겹치지 않게 집계한다. 통합 초안은 root 원장을 즉시 덮어쓰지 않는다.

각 candidate의 14개 필수 등록 필드, family menu, 현재 source hash, 실제 첫 DEV fit 시각과 screen→full→bootstrap 순서를 별도 chronology 표로 대조했다. 16개 screen 모두 실제 DEV fit 전에 등록됐고, 두 screen 탈락에는 full 시작 기록이 없다. Family별 순수 수학 oracle·합성 unit prototype 중 일부는 candidate 등록보다 먼저 실행됐다. 이 때문에 모든 unit sanity가 사용자 명세의 이상적인 stage 순서를 그대로 따랐다고 표현하지 않는다. 등록 이후 실측 DEV 결과에 맞춰 후보 구성을 변경한 사례는 없었다.

B2의 준비된 추론 검사는 통과했지만, CI001의 새 프로세스 검사는 import 중 Numba cache locator의 `os.mkdir` 시도 때문에 실패했다. 모델 점수를 읽기 전에 발생한 cold initialization 문제다. 실패 로그·traceback·기존 진입 파일과 검사 소스를 보존한 뒤, 기존 BL80의 read-only bootstrap을 먼저 설치하는 별도 B2 진입 파일을 추가했다. 이 새 파일의 통과 여부는 CI002 결과로 판단한다. 기존 B2 추론 함수·feature·모델·직접 속도 측정 소스는 변경하지 않았으며, 수리 예정만으로 cold 검사 PASS를 주장하지 않는다.

병목 board 005→006 사이의 실제 간격은 약 47.56분으로 30~45분 권장 간격을 넘었다. 해당 초과를 board 006의 reason에 기록했다. CPU 작업과 회계는 그 동안 계속 관측됐지만, 이것을 제때 수동 병목 검토했다는 근거로 바꾸지 않는다.

CI002는 새 B2 read-only 진입 파일을 두 개의 새 프로세스에서 검증해 통과했다. 각 프로세스에서 synthetic 20개·2,170점의 출력이 일치했고, entry import 전부터 파일 쓰기를 차단해 mutation 시도 0을 확인했다. 기존 prepared core와 같은 함수 객체 및 별도 load의 scalar 출력도 일치했다. 원래 CI001 job은 QUARANTINED 상태로 남겨 두고, 별도 failure resolution receipt가 성공한 CI002 증거를 가리킨다.

B2의 동일 합성 continuation은 실제로 완료됐고, 사전에 존재한 파일 116개의 SHA를 유지하며 이미 완료된 분석 18개를 재계산 없이 재사용했다. 원래 native-crash job은 QUARANTINED로 남겨 별도 resolution이 성공한 continuation을 가리키도록 했다. 이 수리를 모델 개선·새 fit·추가 generating seed로 집계하지 않는다.

Board 007→008도 실제 간격이 45분을 조금 넘었으며 008의 reason에 기록했다. 과거 timestamp를 변경하지 않았다. 원래 실제 관측 회계와 별도로 이 일정 준수 한계를 보고한다.

A2와 C3는 고정 heavy-tail 분산 감소 조건을 통과하지 못해 conditional ALT/H 경로를 닫았다. D2는 원래 SCALE 조건과 네 고정 감소 반례를 통과한 뒤 동일한 20개 ALT/H refit을 시작했다. 개별 합성 분석이 완료됐다는 것과 전체 suite·후속 검증이 완료됐다는 것을 구별한다. 이후 상태는 해당 immutable RESULT와 최종 역할 JSON을 따른다.

첫 B2 합성 작업은 1,500초 경과 비동기 stack dump 도중 Windows native access violation으로 종료됐다. Windows Application 이벤트는 `python313.dll`을 faulting module로 기록했다. 실패 시점의 CB_N3_070 prediction/reference cohort는 이미 완료됐지만 분석 JSON은 없었다. 같은 배열을 일반 모드와 짧은 timed-dump 모드의 새 프로세스에서 각각 세 번 평가했을 때 점수가 완전히 일치했고 짧은 검사에서는 충돌이 재현되지 않았다. 따라서 정확한 native 원인을 확정하지 않는다. [CPython의 timed traceback 문서](https://docs.python.org/3.13/library/faulthandler.html)는 별도 watchdog thread를 사용한다고 설명한다. 이후 worker에서는 그 비동기 timed dump만 끄고 fatal stack handler와 CPU/log 기반 정체 감시는 유지한다. 기존 모델·feature·scorer·합성 정책은 변경하지 않는다.

B2 합성 복구는 새 job과 별도 recovery policy를 등록한 제한적 continuation이다. 이미 저장한 candidate 파일 116개의 바이트 해시를 고정해 재사용하고, 완료 analysis를 다시 채점하지 않는다. 원래 실패 attempt와 QUARANTINED 상태는 보존한다. 미실행 후속 진단 4개의 queue 원본도 보존한 뒤 dependency만 복구 job으로 연결했다. 모델 재적합·새 seed·임계값 변경·sample 수 변경은 없다. 복구 완료 여부는 새 job과 최종 합성 RESULTS에서 확인하며, 복구 등록 자체를 완료로 세지 않는다.

D3도 가장 강한 고정 heavy-tail 감소 CB_N3_050의 점 차이가 음수여서 추가 ALT/H 경로를 닫았다. D2의 ALT/H 20개 fit은 완료됐고, SCALE의 원래 및 ALT/H paired 구간은 모두 양수였다. 전체 H 제외 평균은 소폭 음수이므로 SCALE 이식성을 전체 H 개선으로 바꾸어 설명하지 않는다.

최종 강건성 후보를 실제 파일로 검증할 때는 원래 우선순위의 가장 높은 후보부터 전체 DEV 학습 1회를 수행한다. 학습 recipe와 수치 gate는 변경하지 않는다. 원래 fold-0 합성 suite와 구별하여, 실제 all-DEV 파일을 같은 all-DEV 참조·1000개 calibration·512개 evaluation/long-null에서 확인하는 별도 nomination을 먼저 고정했다. 전체 21개 반례와 4개 long-null, 기존 memory arithmetic을 재사용한다. 실제 파일에서도 네 감소 반례의 점 차이가 각각 양수여야 하며 조건이 깨지면 원래 순위의 다음 후보만 고려한다. 이 추가 확인은 새 성능 점수를 보며 blend나 threshold를 고르는 과정이 아니다.

최종 강건성의 direct runtime은 연구 단계에서 고정 두 paired 순서로 실제 10M update를 측정하고, 별도 final audit에서 한 paired 실행을 반복한다. B2의 다섯 paired 측정과 횟수가 다름을 명시한다. 새 runtime 함수의 미세 최적화는 하지 않았다. 등록된 동일 feature/model 함수와 기존 BL80 runtime을 그대로 호출한다.

Sign reflection은 원래 모델을 고정한 뒤 등록한 입력 stress다. H와 현재 online point를 함께 반전하며, 원래 평가 label/time과 기존 bootstrap multiplicity를 유지했다. 반전 OOF는 새 선택 후보가 아니며, 반전 ensemble이나 augmentation을 만들지 않았다. 원래 primary CI의 실패를 반전 데이터의 양수 CI로 바꾸지 않는다. 결과 범위는 `NEXT4_SIGN_REFLECTION_INTERPRETATION.md`에 기록했다.
