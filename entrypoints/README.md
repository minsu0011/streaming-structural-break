# 모델별 진입점

각 파일은 대회 인터페이스의 `train`, `infer`와 `INFER_PARALLELISM`을 노출합니다. 기본 모델은 저장소 루트의 `main.py`입니다.

- `next_*`: NEXT의 기본·serial·등록·읽기 전용 변형
- `next2_*`: NEXT2 성능형·경량형·공유 runtime
- `next3_*`: exact·lean·packed runtime
- `next4_*`: 강건성 후보와 진단용 모델

저장소 루트에서 모듈로 불러올 수 있습니다. 예: `from entrypoints.next2_performance_main import train, infer`. 모델 파일과 해당 설정은 별도로 준비해야 합니다.
