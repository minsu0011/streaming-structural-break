# NEXT3 입력 복사 및 장기 스트림 호환성 점검

최종 새 환경 재학습 전에 수행한 유지보수 검증이다. 아래 결과는 최종 raw 재학습·30M public API 감사·실제 바탕화면 ZIP 재생을 대신하지 않는다.

## 원본 데이터 없는 입력 복사 점검

최종 재현 준비 스크립트의 입력 목록을 순수 함수 `planned_input_files`로 분리했다. 같은 목록에서 원본 관측·정답 테이블 두 개를 제외하고 소스·설정·guard 메타데이터와 기존 검증용 모델을 새 일반 sibling 폴더에 복사했다. 기존의 63개 고정 패키지 환경을 사용했으며 새 환경 설치라고 기록하지 않았다.

수정 후 점검은 복사 파일 244개와 실제 불러온 프로젝트 모듈 74개의 출처·SHA를 확인했다. 두 public `train`은 빈 공식 iterator를 학습 전에 거부했고, no-fit tripwire는 한 번도 호출되지 않았다. 두 `infer`는 dataset iterator에 접근하기 전에 초기 `None`을 반환했다. 새 학습·경쟁 데이터 복사·관측값 접근·품질 점수는 없었다.

처음 M649는 실패했다. 점검 harness가 tripwire 설치를 위해 `deployment_training`을 public entry보다 먼저 import하면서 `src.next2` Numba dispatcher의 disk cache가 이미 만들어진 상태가 됐다. 이후 DISTILLED의 초기 준비에서 cache directory 변경이 시도되어 차단됐다. 원래 bootstrap은 프로젝트 dispatcher가 생성되기 전에 public entry에서 설치하는 계약이며, 이미 불러온 dispatcher를 정리하는 별도 함수는 모든 `src.next2` 모듈을 대상으로 하지 않는다.

실제 최종 재현 child는 public entry를 먼저 import한다. M650에서는 점검 harness도 이 순서를 따르게 고쳤고 통과했다. 원래 모델·runtime·training 소스는 변경하지 않았다. 실패한 복사본·로그·harness 원본을 보존했으며, 임의로 내부 training 모듈을 먼저 import하는 다른 integration 순서의 read-only 보장으로 확대하지 않는다. 최종 raw child는 Python `-I`와 child에만 적용하는 `CRUNCH_AUTO_MONKEY_PATCH=false`를 명시한다.

근거: `artifacts/next3/maintenance/raw_input_copy_smoke_v1/`, `raw_input_copy_smoke_import_order_correction/`, `raw_input_copy_smoke_v2_entry_first/`.

## 원래 긴 합성 스트림의 출력 재현

고정 E003 public entry를 원래 NEXT2 N0·N1·N2·N3 long-null 각각 512개 × 16,384점과 원래 O1 control/outlier 1,024개 × 1,024점에서 실행했다. 원래 generator·seed·partition·모델·horizon을 그대로 사용했고 기존 저장 float32 출력의 SHA를 먼저 확인했다.

총 3,072개 스트림과 34,603,008개 public 출력이 모두 원래 float32 결과와 정확히 일치했다. 현재 점 하나당 출력 하나, 첫 `None` 이전 dataset 접근 없음, H/O 입력 불변을 확인했고 모든 스트림 끝의 retained numeric state는 16,925바이트였다. 각 스트림의 입력·출력 SHA와 상태 크기를 CSV에 보존했다.

이 검증에서 새 FPR·TS-AUC·threshold를 계산하거나 보정하지 않았다. 기존 Gaussian·AR·ARCH·t5 및 단일 이상치의 동작이 보존된다는 호환성 결과다. 긴 horizon의 경보 증가와 이상치 잔류가 해결됐다는 주장은 아니다. 실행 wall time에는 generator·Python 검증·배열 비교 등이 포함되므로 새 latency benchmark로 사용하지 않는다.

근거: `artifacts/next3/maintenance/LEAN_LONG_HORIZON_PARITY_PROTOCOL.json`, `lean_long_horizon_parity_v1/RESULTS.json`, `lean_long_horizon_parity_v1/PER_SYNTHETIC_SERIES.csv`. 고정 horizon 보정의 해석은 `docs/NEXT3_HORIZON_CALIBRATION_SCOPE.md`에 별도로 정리했다.
