# Streaming Structural Break 탐지

시계열을 한 번에 모두 보는 대신, 관측이 하나씩 들어올 때 분포나 의존 구조의 변화를 찾는 연구입니다. ADIA Lab / Crunch의 구조변화 탐지 문제를 바탕으로 통계 기준선에서 자기회귀 잔차, NEXT와 NEXT2의 다중 신호 구조로 확장했습니다.

## 왜 streaming인가

전체 시계열의 평균이나 미래 구간을 사용하면 사후 변화점은 잘 찾을 수 있어도 실제 순차 탐지와는 다른 문제가 됩니다. 과거 구간에서 정한 상태와 새 관측만으로 점수를 갱신하고, 아직 도착하지 않은 값이나 전체 길이를 읽지 않는 구조를 중심에 뒀습니다.

## 사용 기술

Python, NumPy, pandas, SciPy, scikit-learn, LightGBM, XGBoost, Numba를 사용합니다. 특징 상태 갱신과 경량 추론은 별도 모듈로 나누고, 학습·평가는 고정된 시계열 단위 분할로 관리합니다.

## 데이터와 모델 구조

각 시계열의 과거 구간 H와 순차 관측 O를 입력으로 사용합니다. 변화 시점 정보는 개발용 학습·평가에서만 사용하며 추론 입력으로 넣지 않습니다.

```text
과거 H → 정규화·AR/ARCH 상태
새 관측 → 통계·잔차·변동성·의존성 특징 → 학습기 → 순차 점수
                         ↑
                    이전 관측의 상태
```

평균·분산 변화와 자기상관 변화는 같은 현상이 아닙니다. 기준선, AR 잔차, 분산 증거, 의존성 보완 모델이 각각 다른 변화를 담당합니다. 탐지 점수와 오경보 확률도 구분합니다.

## 개발 과정

### 통계 기준선에서 AR4/AR8로

먼저 rolling/streaming 통계로 단순한 변화를 표현했습니다. 자기상관이 있는 시계열에서는 관측값 변화가 곧 새로운 구조변화라고 보기 어려워, AR4/AR8로 과거 패턴을 설명하고 잔차를 입력에 추가했습니다. 차수는 높다고 무조건 좋은 것이 아니라 배경 의존성을 얼마나 설명하는지에 따라 검토합니다.

### NEXT: 누적 증거가 커지는 이유를 분해

분산과 에너지 채널에 EWMA, CUSUM, GLR, Bayes 요약을 연결했습니다. 여기서 정상 상태에서도 채널에 연속 상관이 남으면 독립 관측을 가정한 누적 증거가 과장될 수 있었습니다.

Serial variance 후보는 과거 H에서 추정한 상관으로 채널 합의 분산을 보정합니다. 순간값이나 모든 요약을 일괄 변경하지 않고 문제가 있는 누적 GLR·Bayes 경로를 구분했습니다. 이 근사는 정확한 오경보 확률 보증을 제공하지는 않습니다.

### NEXT2: 강한 신호를 유지하고 부족한 신호를 보완

분산 primary에 의존성 변화의 보완 모델을 붙이고 scale·sign 등 반례도 살폈습니다. 저장된 performance 설정은 serial-variance primary와 dependence complement를 고정 blend로 결합합니다.

후보 선택에서는 평균 점수뿐 아니라 최악 fold, 대체 분할, 배경 regime, 시계열 단위 bootstrap과 실행 비용을 함께 확인합니다. 후보 수 자체를 늘리는 대신 추가 신호가 없는 특징과 반복 계산을 줄이는 방향입니다.

## 주요 판단과 한계

장기 누적 점수가 큰 것을 곧바로 확실한 경보로 해석하지 않습니다. 정상 장기 시계열에서 오경보를 따로 확인하고, 지연 경보와 점수 순위도 구분합니다. 전체 평균을 올리는 변화가 모든 종류의 break를 개선하는 것은 아닙니다.

여러 세대의 평가 조건이 달라 하나의 개선율로 묶지 않습니다. 최종 독립 평가를 통과한 운영 탐지기라고 주장하지 않으며, 연구용 설정과 실제 경보 정책을 구분합니다.

## 코드와 실행

- [src/features](src/features), [src/streaming](src/streaming): 인과적 특징과 상태 갱신
- [src/models](src/models): 통계·AR 기반 모델
- [src/next](src/next), [src/next2](src/next2): 분산·의존성 증거와 후속 모델
- [configs](configs): 후보와 선택 기준
- [tests](tests): 합성 시계열 및 연산·입력 경계

```bash
pip install -r requirements.txt
python -m pytest tests/test_streaming.py tests/test_models.py tests/test_feature_responses.py tests/test_ts_auc.py -q
```

대표 추론 경로는 [src/streaming/inference.py](src/streaming/inference.py), 학습 경로는 [src/streaming/training.py](src/streaming/training.py)입니다. 이 저장소는 라이브러리와 실험 모듈을 제공하며, 실제 대회 데이터와 공식 runner는 별도로 준비합니다. 위 명령은 연산·순차 처리 예제를 확인하는 시작점입니다.

[개발 과정](docs/wiki/Development-Journey.md) · [모델 구조](docs/wiki/Model-Evolution.md) · [병목](docs/wiki/Bottlenecks-and-Solutions.md) · [결과와 검증](docs/wiki/Validation-and-Results.md)
