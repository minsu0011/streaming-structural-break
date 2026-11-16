# 재현 명령 (프로젝트 경로의 PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/validation/pytest_recheck.xml
.\.venv\Scripts\python.exe scripts/verify_scorer.py
```

처음 두 줄은 새 Windows 작업 폴더에서 환경을 만들 때만 실행한다. 기존 가상환경을 다시 만들 필요는 없다. 위 단계는 raw 데이터 없이 가능한 소스 검사다. package lock은 검증한 Windows Python 3.13 환경 기준이며 Linux 호환성을 보증하지 않는다.

모델과 동결 설정은 정확한 파일 바이트의 SHA256에 연결되어 있다. ZIP의 원본 줄바꿈과 내용을 유지한다. Git 체크아웃이나 편집기가 줄바꿈을 바꾸면 동작이 같아 보여도 hash 검사가 실패할 수 있으므로, 기록된 ZIP 원본으로 재현하고 hash를 임의로 갱신하지 않는다.

선정 CV를 별도 출력 위치에 재현하려면 원본과 hash가 같은 data/raw가 필요하다. 기존 작업 폴더에서는 원본 cache와 OOF를 그대로 비교한다. ZIP을 새 위치에 풀었다면 --rebuild-missing을 추가한다. 이 경우 빠진 cache를 재생성하고 ZIP에 포함된 원래 fold 모델의 예측과 비교하며, 비교 기준을 보고서에 명시한다.

다음 세 명령은 원본 cache·OOF를 보존한 기존 작업 폴더용이다. ZIP을 새 위치에 풀었을 때는 바로 아래의 별도 명령을 사용한다. 전체 최종 감사는 실제 raw 파일·선정 모델·보존된 검증 보고서가 필요하므로 raw 없이 가능한 첫 소스 검사와 구분한다. Git 메타데이터가 없는 압축 해제 폴더에서는 SOURCE_MANIFEST의 포함 목록으로 파일 감사를 수행한다.

```powershell
.\.venv\Scripts\python.exe scripts/final_audit_real.py
.\.venv\Scripts\python.exe scripts/reproduce_selected_cv.py --candidate M7_cross_order_AR4_LGB_AR8_HGB_equal --tag next_session_recheck
.\.venv\Scripts\python.exe scripts/verify_fused_real.py --candidate M7_cross_order_AR4_LGB_AR8_HGB_equal --fast-initialization
```

ZIP을 새 폴더에 푼 경우에는 manifest와 일치하는 data/raw를 먼저 준비하고 다음과 같이 실행한다. 역할별 CV 명령이 필요한 cache를 원본 raw부터 만들고, 이어서 전체 DEV refit을 별도 출력 위치에 저장한다. 기존 결과를 덮어쓰지 않도록 재실행할 때는 새로운 tag를 쓴다.

```powershell
$selectedRoles = (Get-Content DEV_CHAMPION_CANDIDATES.json -Raw | ConvertFrom-Json).candidates
foreach ($selectedRole in $selectedRoles.PSObject.Properties) {
    .\.venv\Scripts\python.exe scripts/reproduce_selected_cv.py --candidate $selectedRole.Value.experiment_id --tag next_archive_cv --rebuild-missing
    if ($LASTEXITCODE -ne 0) { throw 'Selected CV reproduction failed' }
    .\.venv\Scripts\python.exe scripts/reproduce_final_model_from_raw.py --role $selectedRole.Name --tag next_archive_refit
    if ($LASTEXITCODE -ne 0) { throw 'Selected final-model reproduction failed' }
}
```

전체 train 데이터로 `main.train()`을 호출하면 configs/deployment.json의 DEV 8,000개만 학습하고 seal ID는 값·target 처리 전에 건너뛴다. 기존 최종 모델을 덮어쓰지 말고 새 model directory를 사용한다. 입력 ID universe는 동결된 train 10,000개와 일치해야 한다.

`scripts/reproduce_snapshot.py <archive.zip>`은 ZIP을 새 폴더에 추출하고 별도 .repro-venv에서 의존성·테스트·scorer를 검사한다. 새로운 PC에서는 requirements-lock.txt에 맞는 별도 가상환경을 먼저 준비해야 한다. 이 source-only 검사는 raw 데이터를 포함한 CV 재현과 구분한다.

이 명령에 `--artifact-audit final`을 추가하면 원래 작업 폴더의 canonical data/raw를 검증된 참조로 연결하고, 압축 해제 폴더에서 전체 보존 artifact 감사도 수행한다. 이 추가 감사는 raw 바이트 해시를 읽지만 target 성능을 다시 계산하거나 인증 tester를 호출하지 않는다. 최종 전달본의 별도 환경 검증에는 이 추가 단계도 포함했다.

```powershell
py -3.13 -m venv .repro-venv
.\.repro-venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`reduced_diagnostic.py` / 일반 `crunch_tools.test()`를 다시 실행하지 않는다. 이미 허용된 baseline 1회와 primary 1회가 완료됐고 추가 target 조회를 금지했다. 인증정보는 ZIP이나 명령행에 넣지 않는다.
