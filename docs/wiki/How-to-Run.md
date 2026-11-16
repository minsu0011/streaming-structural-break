# 설치와 실행

Python 3.11 이상의 가상환경에서 저장소 루트를 기준으로 실행합니다.

```bash
pip install -r requirements.txt
python -m pytest tests/test_streaming.py tests/test_models.py tests/test_feature_responses.py tests/test_ts_auc.py -q
```

이 테스트는 합성 입력의 상태 갱신과 점수 계산을 확인합니다. 기본 학습·추론 인터페이스는 [main.py](../../main.py), 후속 모델의 인터페이스는 [entrypoints](../../entrypoints)에 있습니다.

## 학습 모델

[모델 릴리스](https://github.com/minsu0011/streaming-structural-break/releases/tag/local-freeze-20261116)의 `trained-models-20261116.zip`을 저장소 루트에 추출합니다. 최종 주력·백업, NEXT2 배포, NEXT8 고정 runtime과 NEXT9 outer-fold 모델의 원래 상대 경로를 유지합니다.

실제 추론에는 선택한 모델과 일치하는 설정·입력 schema가 필요합니다. 원본 대회 데이터는 별도로 준비합니다. 추출만으로 전체 연구를 재실행하는 것은 아닙니다.

## 코드와 기록

| 경로 | 내용 |
| --- | --- |
| [src/streaming](../../src/streaming) | 기본 학습·추론 |
| [src/next2](../../src/next2) | 분산·의존성 결합 |
| [src/next9](../../src/next9) | 조건부 방향성과 전이 후속 연구 |
| [scripts](../../scripts) | 단계별 실행 도구 |
| [research](../../research) | 정리된 결과 표·기술 메모 |
| [vendor/official](../../vendor/official) | 공식 runner·scorer 참고 코드 |

과거 도구가 전제한 결과 파일 배치는 [연구 기록 안내](../../research/README.md)에 설명했습니다. 공식 runner 검사는 합성 연산 테스트와 별도입니다.
