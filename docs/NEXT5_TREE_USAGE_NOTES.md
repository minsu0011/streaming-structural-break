# NEXT5 트리 사용 좌표 점검

ResearchPool에서 실제로 학습한 295개 learner의 트리를 읽고, 학습 시 남긴 열 인덱스를 사전등록된 특징 이름으로 역매핑했다. 원래 특징 좌표 12,467개에 대해 사용 여부, 분기 횟수, 학습 gain을 기록했다. 새 학습이나 실제 관측·라벨 읽기는 없었다.

이 통계는 **각 새 learner 내부의 학습 gain 비율**이다. 고정 결합 예측 전체의 중요도나 held-out 성능 기여도가 아니다. 예를 들어 X2는 새 learner와 BL80을 고정 비율로 결합하므로, 아래 비율을 전체 예측의 기여율로 읽으면 안 된다. 상관된 특징끼리는 같은 신호의 분기 gain을 나눠 가지거나 서로 대체할 수 있다. 이를 인과적 중요도나 독립 신호의 증거로 사용하지 않는다.

## 5개 PRIMARY full 모델의 학습 gain 평균

| 새 learner | Gaussian scale | Student-t scale | log-energy semitail | H energy PIT | persistence | 명시적 곱 특징 | 기타 기존 좌표 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S4B | 32.13% | 8.53% | 59.34% | — | — | — | — |
| X1 | 25.43% | 7.02% | 41.84% | — | 22.06% | 3.66% | — |
| X2 | 15.71% | 3.79% | 16.81% | 57.34% | — | 6.35% | — |
| X3 | 28.48% | 5.14% | 48.41% | — | — | 1.31% | 16.66% |

X3의 기타 좌표는 함께 등록한 C205 dependence 부모 은행이다. 이 표는 동일한 학습 자료가 겹치는 5개 고정 모델의 기술적 평균이며, 5회의 독립 실험으로 계산한 오차막대는 없다. 각 모델의 최소·최대 비율은 원본 CSV에 함께 남겼다.

X2에서는 `next5_energy_pit_raw_upper_ewma128` 한 좌표가 평균 학습 gain의 35.80%를 차지했다. 5개 모델 모두에서 사용됐고 비율 범위는 32.56%~39.97%였다. conditional upper EWMA128은 평균 9.11%였다. 현재 산출물은 긴 시간척도의 보정된 energy evidence를 많이 사용하는 것으로 관측된다.

명시적 곱 특징을 모델이 사용한 사실은 확인됐지만, 곱 특징만 제거한 사전등록 ablation은 이번 메뉴에 없다. 따라서 X2와 부모 후보의 held-out 차이를 곱 특징의 순수 기여로 분해하지 않는다. X2는 두 부모의 특징 은행을 함께 학습한 효과와 곱 특징의 효과를 동시에 포함한다. 이 통계로 특징을 제거하거나 새 후보를 만들지 않는다.

근거:

- `artifacts/next5/feature_usage/fixed_v1/RESULTS.json`
- `artifacts/next5/feature_usage/fixed_v1/FEATURE_USAGE.csv`
- `artifacts/next5/feature_usage/fixed_v1/BANK_FIXED_FOLD_SUMMARY.csv`
- `artifacts/next5/feature_usage/fixed_v1/FEATURE_FIXED_FOLD_SUMMARY.csv`
- `artifacts/next5/integrity_audits/research_v1/fit_inventory/PRIMITIVE_FITS.csv`

전체 파일은 모델 해시·등록 해시·학습/검증 ID 해시를 보존한다. 모델 사용 좌표에 대한 설명은 정보 분해와 합성 충격 반응을 해석하는 보조 자료이며, 승격 기준을 대체하지 않는다.
