# 테스트와 실행 확인

의존성을 설치한 뒤 저장소 루트에서 실행합니다.

```bash
python -m pytest tests/test_streaming.py tests/test_models.py tests/test_feature_responses.py tests/test_ts_auc.py -q
```

합성 입력의 상태 갱신, 특징 반응, 모델 출력과 TS-AUC를 검사합니다.

후속 단계의 학습·재현에는 원본 데이터와 해당 단계의 모델·설정이 필요합니다. [모델 릴리스](https://github.com/minsu0011/streaming-structural-break/releases/tag/local-freeze-20261116)를 저장소 루트에 추출하고 [실행 안내](wiki/How-to-Run.md)를 따릅니다.
