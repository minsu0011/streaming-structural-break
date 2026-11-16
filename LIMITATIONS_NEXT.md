# NEXT 한계와 해석 범위

P(Δ>0)는 2,000개 재표본에서 차이가 양수였던 비율이다. 1.0이어도 개선이 확실하다는 posterior 확률이나 독립 시험의 성공 확률은 아니다.

경보 분석의 개별 SERIES.csv 11개는 원래 tau 라벨을 보존하므로 ZIP에서 제외한다. 집계 성능, paired bootstrap, 고정 정책과 해당 로컬 파일의 SHA256 증거는 남긴다. 원본 표는 로컬에서 삭제하거나 수정하지 않았다.

DEV를 반복적으로 관찰하며 후보를 제안했다. 2,000회 paired bootstrap은 이미 학습된 OOF와 선택된 후보에 조건부이다. 수백 번의 탐색을 보정한 신뢰구간이나 독립 성능 보장이 아니다. 실제 sealed 관측/라벨을 이용한 평가를 수행하지 않았다.

H regime은 전체 DEV의 historical descriptor로 군집을 만들었으므로 transductive 진단이다. 각 regime 제외 fit은 새로 수행했지만, regime 정책 자체의 독립 검증으로 해석하지 않는다. MEAN 21개, TAIL 86개, H regime 2의 166개 등 표본이 작은 절단은 불확실하다.

J102와 Y001의 평균 차이는 약 2.8e-7로 사실상 동률이다. 자기상관 보정 효과를 입증하지 못했다. GLR 제거의 CI도 0을 포함한다. B104의 runtime은 학습에서 제거한 일부 GLR 계산을 여전히 수행한다. 성능 열 제거와 계산 제거를 혼동하지 않는다.

TS-AUC는 첫 경보의 false-positive와 탐지율을 보장하지 않는다. 별도 nested null calibration은 특정 threshold와 짧은 horizon의 진단이며 배포용 threshold가 아니다. 고정 1,024-point threshold를 16,384-point null stream에 적용한 추가 연구 결과는 artifacts/next/long_null_stress를 확인한다. 실제 경쟁 데이터 생성 과정과 같다고 주장하지 않는다.

장기 null 연구의 모델은 사전에 고정한 PRIMARY/V007/Q005/Y001의 CV fold 0 모델이다. J102, B104, 50-tree 모델이나 최종 all-DEV artifact에서 동일한 false-alarm 비율을 직접 측정한 것으로 해석하지 않는다.

합성 stress에서 변동성 모델은 null FPR 편차를 줄였으나 일부 ARCH scale/persistence 및 tail 시나리오에서 기존 모델보다 나빴다. 반례를 삭제하거나 방향별 결과를 본 후 모델/threshold를 바꾸지 않았다.

초기 N013_ARCH_REFERENCE는 의도한 ABCD 51개 열 대신 E/F까지 포함한 57개 열을 써서 잘못된 matched baseline이었다. 해당 결과는 BUG로 남기고 승격에서 제외했다. 별도 E001_REFERENCE_ADAPTER_PARITY는 수정된 51개 열로 기존 HIGHEST_MEAN의 전체 OOF를 정확히 재현했다. 또 초기 일부 full 실험에 남은 500회 bootstrap은 중간 기록이며 최종 판정은 후속 2,000회 forensic 결과를 사용한다. 이 수정으로 fold 예측이나 승격 기준을 바꾸지는 않았다.

초기 next_main.py의 cold import는 Numba cache-directory 생성 시도가 차단되어 read-only 검증을 통과하지 못했다. 실제 전달 entry는 import 전에 캐시 생성을 억제하는 next_readonly_main.py 또는 대응하는 registered/serial entry다. 실패한 이전 entry와 음성 대조 증거를 지우지 않았다.

고정 후보 메뉴를 사용한 nested CV도 그 메뉴가 전체 DEV를 보고 지정되었기 때문에 전체 adaptive search의 unbiased 성능 추정은 아니다. bootstrap 역시 선택된 메뉴와 fit에 조건부이다.

SealGuard는 애플리케이션 경계다. predicate-filtered DEV 행만 사용했지만 Parquet 내부 mixed row-group decoding과 raw 전체 바이트 해시가 seal 바이트를 전혀 읽지 않았다는 OS 수준 주장은 하지 않는다. Python audit hook은 native system call sandbox가 아니다.

독립 raw 재현은 별도 소스 workspace와 새 feature/CV/all-DEV 학습을 사용했다. 그 학습에는 이전에 별도로 설치한 .repro-venv를 재사용했다. 이후 최종 감사는 새로 설치한 pinned 환경과 전달용 파일 복사본에서 선택 모델의 cold infer를 별도 검증하도록 강제한다(artifacts/next/portable_delivery/RESULTS.json). 새 환경에서는 추가 raw fit을 하지 않는다. 두 검증 모두 같은 OS/hardware 범위이며 다른 기종의 exact-byte parity를 보장하지 않는다.

공식 로컬 wire runner는 합성 protocol fixture의 train 단계에서 이미 학습된 모델을 복사했다. 실제 생산 train 구현은 별도 final_training의 DEV 원시 iterator audit로 검증했다. 이 둘을 합쳐 cloud 학습 또는 실제 SDK test를 수행했다고 주장하지 않는다.

10M latency는 고정 DEV buffer 반복으로 직접 측정한 속도/메모리 검사다. 새로운 10M 독립 관측이나 품질 평가가 아니다. 세부 component call 시간의 합은 fused scalar end-to-end 시간과 같지 않다.

점당 latency의 경계는 준비된 detector.predict_one 호출이며 timer와 float32 저장 비용을 포함한다. infer generator의 검증 분기, 파일/소켓/클라우드 전송 비용은 이 점당 수치에 포함하지 않았다. 모든 비교 모델에 동일한 경계를 적용했다.

메모리의 고정 state-array 크기와 process RSS는 다르다. rss_peak_bytes는 1M-point 간격에서 표본 추출한 최대 RSS로, 순간적인 전체 프로세스 최대 메모리를 보장하지 않는다. rss_after_bytes는 타이밍 밖의 원본 scalar 검증 뒤에 측정되어 표본 peak보다 높을 수 있다.

ZIP은 검토용 코드, 보고서, 잠긴 metadata, 최종 1~3개 모델을 담는다. raw observations/targets, feature/OOF caches, 환경 폴더, 인증정보와 이전 binary 모델은 제외된다. 전체 원시 재실행에는 사용자가 보유한 canonical raw training files 및 기존 REAL 자료가 별도로 필요하다.
