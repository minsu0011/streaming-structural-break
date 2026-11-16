# 실행과 테스트

Python 3.11 이상의 가상환경에서 `pip install -r requirements.txt`로 의존성을 설치합니다.

```bash
python -m pytest tests/test_streaming.py tests/test_models.py tests/test_feature_responses.py tests/test_ts_auc.py -q
```

이 테스트는 합성 시계열의 상태 갱신, 특징 반응, 모델 출력과 TS-AUC 계산을 확인합니다. 원본 대회 데이터나 학습 모델 없이 시작할 수 있는 범위입니다.

후속 연구 전체의 재현에는 해당 단계의 설정, 학습 모델과 데이터가 필요합니다. 모델은 릴리스에서 받아 저장소 루트에 추출합니다. 단계별 스크립트는 당시 `artifacts/` 구조를 기준으로 작성되어 있습니다.

합성 테스트 통과와 실제 데이터의 독립 성능 검증은 별개입니다. 이 문서 정리에서는 모델 재학습이나 전체 연구 재현을 수행하지 않았습니다.
