# NEXT3 선택 장기 재생 명령

기본 전달 검증은 두 역할의 18개 합성 사례를 정방향·역방향으로 실행하는 `scripts/replay_next3_portable.py`의 98,372점 재생이다. 이 문서의 선택 명령은 원래 NEXT2의 긴 합성 스트림에서 PERFORMANCE의 동작이 보존되는지도 확인할 때 사용한다.

## 실행

최종 ZIP을 일반 폴더로 압축 해제하고 `01_GPT_HANDOFF_NEXT3.md`처럼 Python 3.13.2 및 `requirements-lock.txt`의 고정 패키지 63개를 준비한 뒤, 압축 해제한 최상위 폴더에서 실행한다.

```powershell
./.review-venv/Scripts/python.exe -I scripts/replay_next3_long_horizon.py --output review_next3_long_horizon.json
```

기본 모델 경로는 `DEPLOYMENT_ENTRY_POINTS_NEXT3.json`의 최종 PERFORMANCE 항목을 따른다. 출력 파일이 이미 있으면 다른 새 이름을 지정한다. 원본 데이터, 정답, point 예측 `.npy` 파일, 새 학습, 네트워크 접근은 재생 입력에 필요하지 않다. manifest의 모든 파일이 유지된 압축 해제본을 사용한다.

## 확인하는 내용

- 원래 N0 Gaussian, N1 AR(0.6), N2 AR/ARCH, N3 음의 AR/ARCH/t5를 각각 512개 × 16,384점 재생한다.
- 원래 O1 control 512개와 단일 이상치 512개를 각각 1,024점 재생한다. generator·seed·partition·H 길이·burn-in은 원래 고정 계약을 따른다.
- 총 3,072개 스트림·34,603,008개의 float32 출력 SHA와 생성된 입력 SHA를 고정 `PER_SYNTHETIC_SERIES.csv`와 비교한다. CSV는 case 좌표로 연결하므로 물리적 행 순서에 의존하지 않는다.
- 처음 `None` 이전 dataset 접근 없음, 현재 관측 하나당 출력 하나, 유한한 [0,1] 출력, 각 스트림 종료의 retained numeric state 16,925바이트를 확인한다.
- manifest 전체를 실행 전후 검증하며 실제 불러온 프로젝트 모듈이 복사본 내부의 등록된 파일인지 확인한다. 추론 중 지속 쓰기·경쟁 데이터·저장 `.npy` 읽기·네트워크 시도를 차단한다. 이는 Python audit hook 범위이며 OS sandbox가 아니다.

## 실제 실행 근거와 범위

M651에서 원래 저장 합성 예측과 E003 public entry 출력을 직접 비교했고, M660에서는 경쟁 데이터와 `.npy`가 전혀 없는 새 일반 sibling 복사본에서 위 CLI를 실제 실행했다. M660은 고정 canonical validation 모델을 `--model-dir`로 명시했다. 최종 기본 descriptor 경로의 선택은 최종 전달 문서 계약이며 이 유지보수 실행에서 최종 descriptor를 썼다고 기록하지 않는다.

M660은 249개 복사 파일과 불러온 프로젝트 모듈 32개를 검증했다. 3,072개 스트림·34,603,008개 출력이 모두 일치했고 입력·소스가 유지됐으며 쓰기·데이터/예측 접근·네트워크 시도는 0이었다. 기존 고정 환경을 사용했으므로 fresh environment 설치나 최종 ZIP 재생으로 기록하지 않았다. 실행 시간 약 398초는 생성·검증을 포함하며 latency benchmark로 사용하지 않는다.

근거는 `artifacts/next3/maintenance/long_horizon_copy_replay_v1/RESULTS.json`, `CHILD_RESULT.json`, `INPUT_MANIFEST.json`, `START.json`, `RUN.log`와 `OPTIONAL_LONG_HORIZON_COPY_REPLAY_PROTOCOL.json`이다. compact golden의 출처는 `LEAN_LONG_HORIZON_PARITY_PROTOCOL.json` 및 `lean_long_horizon_parity_v1/`이다.

기존 긴 horizon 경보 증가와 이상치 잔류도 그대로 재현되는 결과다. 새로운 threshold·FPR·TS-AUC·품질 개선을 계산하거나 주장하지 않는다. 최종 새 환경 raw 재학습, 별도 30M public API 감사, 실제 ZIP CRC/SHA 및 원본 접근을 차단한 기본 재생은 각각 별도의 필수 검증이다.
