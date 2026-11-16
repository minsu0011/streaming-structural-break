# 한계

1. 고정 DEV 5 folds에서 249개 후보를 반복 비교했다. 선택 편향을 보정한 독립 성능은 없다. group bootstrap도 고정 OOF에 조건부인 진단이다.
2. 주력 break-age 0–4 TS-AUC 0.522997; 빠른 검출은 여전히 약하다. 첫 경보에 관한 DEV 진단은 `artifacts/real/first_alarm_diagnostics`에 있지만, 관측 기간과 학습 구간 임계값에 의존하며 독립 운영 성능 보증은 아니다.
3. Exact / normalized exact / prefix64 중복은 없었고 사전 indexed 12,438쌍도 merge 기준을 통과하지 않았다. 같은 길이만 검사한 근사 탐색이므로 숨은 생성 family나 모든 near duplicate의 부재를 증명하지 않는다. split/threshold는 바꾸지 않았다.
4. 사전 전체 기술 통계 감사에서는 모든 train series의 길이와 break/tau 메타데이터를 요약했다. split 동결 후 모델 학습·CV·진단은 DEV에 한정했으며 seal 성능을 계산하지 않았다. 원시 파일 SHA256 검사는 파일 전체 바이트를 읽는 작업이다.
5. 실제 Windows 공식 runner와 DEV 전체 loopback 소켓 경로는 검증했다. Linux cloud 설치·프로세스 병렬화·cloud IPC 포함 SLA는 측정하지 않았다. package lock은 로컬 재현 기준이다.
6. Point-only 속도는 historical 초기화 비용을 제외한다. 10M amortized 추정은 관측한 historical/online 길이 분포에 의존한다. 속도 백업 선정에는 CV fold-0 모델을 사용했고, 최종 latency 표에서는 전체 DEV refit 모델을 별도로 측정했다.
7. 선정된 모델의 각 표현은 57개 feature를 계산하므로 model-column ablation이 feature 계산량을 줄이지 않는다. 고정 AR4/AR8 평균은 두 표현, 총 114개를 계산한다. 일부 tail feature는 historical null clip에 걸리며 상세 비율은 feature_health audit에 있다.
8. Reduced 2회 quota는 모두 사용했다. 단 100개 series 진단을 일반화 성능이나 추가 튜닝 근거로 삼지 않는다. LOCAL_FINAL_SEAL은 미개봉이다.
9. 연구 전 원래 setup 위치 파일은 Windows byte 접근이 거부되었다. 원본을 수정하지 않고 공식 인증으로 동일 이름 데이터를 canonical data/raw에 새로 받아 SHA256을 동결했다.
10. 순위 점수는 교정된 posterior 확률로 검증하지 않았다. 실제 대회 제출이나 업로드는 수행하지 않았다.
