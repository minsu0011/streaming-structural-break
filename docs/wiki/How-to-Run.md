# 실행과 소스 탐색

저장소 루트에서 환경을 준비합니다.

```bash
pip install -r requirements.txt
python -m pytest tests/test_streaming.py tests/test_models.py tests/test_feature_responses.py tests/test_ts_auc.py -q
```

작은 합성 입력으로 연산과 상태 갱신을 확인하는 시작점입니다. 원본 대회 데이터, 모델 파일과 공식 runner를 필요로 하는 경로는 별도로 준비해야 합니다. 공식 protocol 테스트는 vendor runner 없이 실행할 수 없습니다.

소스는 [features](../../src/features) → [streaming](../../src/streaming) → [models](../../src/models) 순서로 읽은 뒤 [next](../../src/next)와 [next2](../../src/next2)를 보면 입력 표현의 확장을 따라가기 쉽습니다.

학습용 iterator와 순차 추론 iterator의 역할을 섞지 않습니다. 고정된 seal을 단순 실행 예제를 위해 열거나 후보를 추가 튜닝하는 용도로 사용하지 않습니다. 세부 설정은 [configs](../../configs), 연구 선택 조건은 [decision_gates.py](../../src/next2/decision_gates.py)에 있습니다.
