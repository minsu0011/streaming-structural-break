# NEXT5 고정 속도 측정 결과

2026-09-11 13:51:44 UTC에 9개 모델 × 3개 순서 × 1,000만 회, 총 270,000,000회 연속 처리가 PASS로 완료됐다. 모든 모델에서 세 반복의 float64 출력 해시가 같고 고정 H의 mutable state 배열 크기도 같았다. 전체 NEXT5 연구나 최종 감사의 완료를 뜻하지 않는다.

아래 시간은 반복별 측정치 세 개의 중앙값이다. p95/p99 열도 각 반복 분위수의 중앙값이며 전체 호출을 합친 분위수가 아니다. 상태 크기는 1,000만 회에 사용한 고정 H 기준이다. 실제 H 길이에 따른 범위와 프로세스 RSS는 원래 block 영수증에 있다.

| 모델 | point median µs | p95 µs | p99 µs | H 초기화 ms | 10M 초 | mutable state bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FROZEN_PRIMARY | 9.200 | 10.600 | 12.400 | 4.676 | 94.919 | 16925 |
| ResearchPool_BL80 | 11.400 | 12.900 | 15.300 | 5.157 | 116.271 | 17260 |
| S4B_UNIFIED_SCALE_SPECIALIST | 15.600 | 18.700 | 26.975 | 6.640 | 155.088 | 34612 |
| P1_TWO_TIMESCALE_CONFIRMATION | 13.900 | 16.100 | 19.600 | 6.474 | 141.058 | 19020 |
| W2_TRIMMED_RECENT_ENERGY | 14.100 | 16.000 | 18.800 | 6.067 | 144.240 | 23892 |
| E2_SPARSE_START_MIXTURE | 18.000 | 20.400 | 23.600 | 5.814 | 182.152 | 34156 |
| R2_CONDITIONAL_SCALE_PIT | 14.200 | 16.300 | 19.400 | 8.317 | 143.129 | 75860 |
| M1_ROBUST_LOCATION_LIKELIHOOD | 14.000 | 16.100 | 18.900 | 6.151 | 139.628 | 17870 |
| X2_SCALE_TAIL | 17.300 | 19.600 | 22.500 | 9.331 | 172.907 | 93404 |

일곱 후보 모두 동결된 세 비율(point median, amortized real sequence, 10M direct)의 3배 기준을 통과했다. 비율은 각 반복 안에서 기존 FROZEN_PRIMARY와 비교한 뒤 중앙값을 구했다. X2는 각각 1.8804347826086958배, 1.9125219417252668배, 1.813625791443375배다. 속도 기준 통과는 +0.0012 평균 품질 기준 실패를 상쇄하지 않는다.

102개 ResearchPool 표본과 고정 반복 버퍼의 공학적 측정이며 새 품질 점수는 없다. 다른 workspace Python을 실행하지 않았다. native 메타데이터 확인·병목 검토와 그 기록 복구는 발생했고 원래 모든 block을 보존했다. 자세한 범위는 [측정 읽기](NEXT5_RUNTIME_READING_GUIDE.md), [native 기록 복구](NEXT5_NATIVE_REVIEW_REPAIR.md)를 따른다.

근거: artifacts/next5/latency/fixed_v1/RESULTS.json, SUMMARY.csv, BLOCK_RATIOS.csv와 27개 repeat JSON. 최종 RESULT SHA256: 9f5a036cddb081a504454396aa3e49650e0a7f15db2235f7b44f1ca761c45782.
