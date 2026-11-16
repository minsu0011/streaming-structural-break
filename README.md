# Streaming Structural Break

순차 관측에서 평균·분산·의존 구조의 변화를 탐지하는 Python 연구입니다. ADIA Lab / Crunch의 구조변화 탐지 문제를 바탕으로, 통계 기준선에서 AR 잔차와 누적 증거, 다중 모델 결합으로 확장했습니다.

## 모델 구조

```text
과거 구간 H → 정규화 · AR/ARCH 상태 추정
새 관측 xₜ → 특징 갱신 → 학습 모델 → 순차 탐지 점수
                    ↑
               이전 시점의 상태
```

추론은 과거 구간과 현재까지 도착한 관측만 사용합니다. 변화 시점 라벨은 학습과 평가에 사용하며, 추론 입력에는 포함하지 않습니다.

| 구성 | 주요 내용 | 코드 |
| --- | --- | --- |
| 통계·AR 기준선 | 순차 통계, AR4/AR8 잔차, ARCH 정규화 | [features](src/features), [models](src/models), [streaming](src/streaming) |
| NEXT · NEXT2 | 분산 누적 증거, serial variance 보정, 의존성 보완 모델 | [next](src/next), [next2](src/next2) |
| NEXT3–NEXT5 | 경량 추론, 특징 기여도, scale·sign·이상치 강건성 분석 | [next3](src/next3), [next4](src/next4), [next5](src/next5) |
| NEXT6 · NEXT7 | 특징 관계와 상태 전이 분석 | [next6](src/next6), [확장 실험](src/next6_extension), [next7](src/next7) |
| NEXT8 · NEXT9 | 변수 동역학, 조건부 방향성, 자기 전이와 결합 모델 | [next8](src/next8), [next9](src/next9) |

NEXT 번호는 연구 단계입니다. 번호가 높다고 최종 성능이 더 좋거나 기존 모델을 대체했다는 뜻은 아닙니다.

## 시작하기

Python 3.11 이상을 사용합니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_streaming.py tests/test_models.py tests/test_feature_responses.py tests/test_ts_auc.py -q
```

위 테스트는 합성 입력의 상태 갱신·특징 반응·점수 계산을 확인합니다. 실제 데이터의 성능 평가는 원본 데이터와 해당 실험 설정이 필요합니다.

기본 학습·추론 진입점은 [main.py](main.py)이며, 구현은 [training.py](src/streaming/training.py)와 [inference.py](src/streaming/inference.py)에 있습니다. 후속 연구의 실행 스크립트는 `scripts/next7`, `scripts/next8`, `scripts/next9`에서 확인할 수 있습니다.

## 코드 탐색

| 경로 | 내용 |
| --- | --- |
| `src/` | 특징 계산, 모델, 학습·추론, 평가 구현 |
| `configs/` | 모델 설정과 실험 선택 기준 |
| `entrypoints/` | 후속 모델별 대회 학습·추론 진입점 |
| `scripts/` | 단계별 학습·분석·재현 도구 |
| `research/` | 단계별 실험 결과와 기술 메모 |
| `tests/` | 수식, 인과적 입력 경계, 상태와 추론 테스트 |
| `docs/` | 설계 설명과 연구 메모 |
| `vendor/official/` | 출처가 기록된 공식 scorer·runner 참고 코드 |
| `data/` | 데이터 준비 안내 |

설계는 [모델 구조](docs/architecture.md), 단계별 변경은 [모델 발전 과정](docs/model-evolution.md), 실행 범위는 [테스트 안내](docs/testing.md)에 정리했습니다. 단계별 평가 표와 설정 기록은 [research/reports](research/reports)에, 상세 기술 메모는 [research/notes](research/notes)에 있습니다.

## 학습된 모델

[모델 릴리스](https://github.com/minsu0011/streaming-structural-break/releases/tag/local-freeze-20261116)에 학습이 완료된 모델과 복원용 메타데이터를 보관합니다.

- `trained-models-20261116.zip`: 기존 최종 주력·백업 모델, NEXT2 배포 모델, NEXT8 고정 runtime 모델, NEXT9 outer-fold 학습 모델

`trained-models-20261116.zip`을 저장소 루트에 풀면 모델의 상대 경로가 유지됩니다. 원본 대회 데이터와 대용량 캐시는 포함하지 않습니다. 모델별 Python 패키지 버전은 잠금 파일을 참고하세요.

## 평가 범위

시계열 단위 분할과 TS-AUC를 사용하며, 평균뿐 아니라 최악 fold, 대체 분할, 배경 regime, 실행 비용을 함께 살폈습니다. 연구 단계마다 데이터 범위와 평가 조건이 달라 결과를 하나의 개선율로 합치지 않습니다.

탐지 점수의 순위 성능은 실제 경보의 오경보율이나 지연을 보장하지 않습니다. 반복적인 개발 데이터 사용에 따른 선택 편향과 장기 정상 구간의 오경보를 별도로 고려해야 합니다. 이 저장소는 연구 코드와 학습 모델을 제공하며 운영 환경의 독립 검증을 대신하지 않습니다.

## 출처

ADIA Lab / Crunch의 구조변화 탐지 문제와 공식 평가 프로토콜을 참고했습니다. 제3자 자료의 이용 조건은 [ATTRIBUTION.md](ATTRIBUTION.md)와 [NOTICE](NOTICE)를 확인하세요.

[전체 문서 안내](docs/README.md)
