# 모델 구조

입력은 과거 구간 H와 이후 순차 관측입니다. H에서 정규화와 AR/ARCH 상태를 정하고, 관측이 도착할 때 특징과 탐지 점수를 갱신합니다. 라벨과 미래 관측은 추론에 사용하지 않습니다.

`src/features`는 특징 상태를, `src/models`는 기준선과 학습기를, `src/streaming`은 학습·추론 경로를 담당합니다. `main.py`가 기본 진입점입니다.

NEXT 계열은 별도 연구 모듈입니다. NEXT와 NEXT2는 분산 누적 증거와 의존성 보완을, NEXT3–NEXT5는 runtime과 강건성을, NEXT6 이후는 특징 관계와 상태 전이를 분석합니다. NEXT8·NEXT9는 조건부 방향성과 변수 동역학을 다룹니다.

학습된 모델은 [릴리스](https://github.com/minsu0011/streaming-structural-break/releases/tag/local-freeze-20261116)에 있습니다. 각 연구 단계의 경로와 설정을 유지하므로 서로 다른 단계의 모델을 같은 설정으로 불러오지 않아야 합니다.
