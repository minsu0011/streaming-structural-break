# NEXT3 Windows 전달 경로와 소스 누락 수정

최종 감사 전 실제 일반 파일 복사에서 두 문제를 찾아 고쳤다. M681은 2026-09-10 10:30 UTC에 **13,260개 파일, 93,392,365 bytes**를 복사하고 해시를 검증했다. 당시 전체 실행 소스·설정 **904개**가 포함됐으며, 선택된 검토 명령의 입력 11개와 frozen lean 설정 8개도 확인했다. 이 수치는 유지보수 복사본의 수치다. 이후 추가되는 보고서와 최종 새 모델·소스 잠금·실제 ZIP은 별도로 검증한다.

## 실제 실패와 수정

M677은 실패 원인을 보존한 JSON의 상대 경로 165자에 긴 중간 폴더 경로가 더해져 절대 경로가 274자가 됐다. Windows 파일 생성에서 FileNotFoundError가 났다. 보존 기록을 삭제하거나 member 이름을 줄이지 않았다. 원래 복사본과 source·protocol·worker 로그를 남겼다.

내부 유지보수 staging을 `artifacts/next3/handoff/m3/payload`로, 최종 core 점검 staging을 `artifacts/next3/handoff/cp/payload`로 짧게 지정했다. 실제 M681의 가장 긴 파일 경로는 UTF-16 code unit 242개, 디렉터리는 201개다. 복사 전에 전체 예정 경로를 검사하며 파일은 최대 259개, 디렉터리는 보수적으로 247개까지만 허용한다. 보조 평면 Unicode 문자는 code unit 두 개로 계산한다.

[Microsoft의 Win32 경로 제한 문서](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation)는 전통적인 MAX_PATH와 별도의 긴 경로 지원 조건을 설명한다. 이번 수정은 짧은 일반 경로를 사용하는 방식이다. 레지스트리·시스템 설정·권한을 바꾸지 않았고, 임의로 깊은 경로의 지원을 주장하지 않는다. 실제 최종 압축 해제 경로도 SHA로 연결된 manifest를 기준으로 파일 생성 전에 검사한다.

M679는 짧은 경로에서 파일 복사를 마쳤지만, 전체 실행 소스와의 집합 대조에서 다음 7개가 빠진 것을 발견해 실패했다.

- `08_SPLIT_LOCK.json`, `SPLIT_LOCK_REAL.json`
- `main.py`, `next_main.py`, `next_readonly_main.py`, `next_registered_main.py`, `next_serial_main.py`

두 JSON은 기존 group split의 집계·해시 잠금 사본이고 관측·label 표가 아니다. 기존 `source_files(ROOT)`가 정의한 전체 실행 입력을 전달 후보에 포함하도록 고쳤다. 일반 파일, 경로, raw 관측·target 모양, 비밀 정보 및 크기 필터는 그대로 적용한다. 최종 core 감사에서도 최종 source lock의 모든 파일이 새 payload에 같은 bytes로 들어가는지 확인한다.

추가 정적 검토에서 아직 실행되지 않은 `모든 .joblib 제외`라는 점검기 가정을 바로잡았다. 기존 FROZEN_REFERENCES에 이미 보호 대상으로 등록된 학습 모델 4개는 정확한 기존 SHA를 유지해 포함한다. 새 임의 joblib 파일을 허용하지 않고, 이 점검은 모델 파일을 역직렬화하지 않는다. 공개 배포 역할의 추론은 canonical 모델을 사용한다. 이 사항은 M679의 실제 실패 지점인 7개 누락과 구분한다.

## 검증 범위

M678의 경로·기존 payload 필터·archive·manifest 검사 **73개**, M680의 경로·payload·전체 source 경계 검사 **42개**가 실패·오류·skip 없이 통과했다. 259/260 파일 경계, 디렉터리 경계, Unicode 길이, 실제 실패 member, 기존 경로 안전성 규칙을 점검했다. M681은 수정한 필터를 실제 전체 복사에 적용하고 13,260개 파일을 다시 해시했다.

실행된 실패 child 수는 이 두 실패를 포함해 15개가 됐다. 과학 fit·품질 평가·bootstrap·모델 승격은 추가하지 않았다. 두 실패를 성공으로 덮어쓰지 않았으며, `path_budget_correction_001/002`, 각각의 원래 source와 `M677/M679/M681` 작업 기록을 보존했다.

주요 근거는 `artifacts/next3/maintenance/PAYLOAD_DEPENDENCY_COVERAGE_RESULT.json`과 `CHECKPOINT_13_REPAIR_COMPLETION.json`이다. 원래 짧은 복사본은 handoff 제외 영역에 보존한다. root source manifest, 바탕화면 ZIP, 실제 압축 해제 재생, 문서화된 검토 명령과 정확히 20개 파일 검증은 최종 단계에서 별도 완료해야 한다.
