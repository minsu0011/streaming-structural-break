# 모델과 누적 증거

## 입력 표현

과거 H에서 정규화와 AR/ARCH 상태를 정하고, 새로운 O가 들어올 때 상태를 갱신합니다. 개발 label은 시계열별 변화 시점이며 학습기 입력에는 넣지 않습니다.

통계 기준선은 수준·분포 변화를, AR 잔차는 배경 자기상관을 제거한 변화를, 분산 은행은 에너지 변화의 여러 시간 규모를 표현합니다. Dependence complement는 분산 primary와 다른 의존성 변화를 담당합니다.

## Serial variance의 핵심

길이 n인 채널 합 S에 대해 독립성을 가정하면 분산은 n입니다. H에서 추정한 lag-1 상관 ρ를 사용하면 다음 근사를 씁니다.

```text
v(n) = n + 2 × Σ[k=1..n−1] (n−k)ρ^k
GLR 요약 = S² / (2v)
```

ρ는 안정성을 위해 제한하고, 온라인 관측으로 다시 추정하지 않습니다. Bayes 요약도 합의 분산 v를 사용하는 평균 변화의 근사 적분에 기반합니다. 실제 clipped energy 채널이 정확한 Gaussian AR(1)이라는 보장은 없으므로 이를 anytime-valid 확률이나 오경보 보증으로 설명하지 않습니다.

## NEXT2의 고정 조합

[performance 설정](../../configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json)은 serial variance 기반 LightGBM primary와 작은 dependence feature bank의 LightGBM complement를 결합합니다. 계수는 코드·설정에 고정된 값이며 경보가 나올 때마다 다시 고르지 않습니다.

후보 정의의 keep_names는 어떤 증거가 실제 학습기에 들어가는지 보여줍니다. 전체 feature bank와 최종 모델이 사용하는 부분은 다를 수 있습니다.

## 읽을 코드

- [features/streaming.py](../../src/features/streaming.py): 상태 기반 특징
- [serial_channel_evidence.py](../../src/next/serial_channel_evidence.py): 누적 합의 serial 분산
- [complement_bank.py](../../src/next2/complement_bank.py): 보완 특징
- [decision_gates.py](../../src/next2/decision_gates.py): 역할별 조건
- [ts_auc.py](../../src/scoring/ts_auc.py): 평가 지표

설정과 학습 경로가 존재하는 것과 최종 독립 평가 통과는 다른 주장입니다.
