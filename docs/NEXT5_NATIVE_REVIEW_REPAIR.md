# 속도 측정 중 병목 검토 게시 복구

2026-09-11 13:26 UTC의 native PowerShell 검토는 `012_REVIEW.json` history 파일을 작성했으나, 기존 `BOARD.json`을 교체하는 단계에서 exit 1로 실패했다. `File.Replace`의 백업 경로에 전달한 PowerShell `$null`이 빈 문자열로 변환되어 유효한 경로가 아니라는 오류가 발생했다. 실제 속도 측정 worker는 계속 실행됐다.

원래 스크립트, 정책, 012 history, 게시되지 않은 임시 파일과 이전 BOARD를 보존했다. 별도 작은 파일에서 `[NullString]::Value`를 사용한 native 교체를 실제 검사했다. 새 `review_next5_bottlenecks_during_timing_v2.ps1`은 명시적 null 인자와 버전·정책 경로 등 네 문자열만 바꾼 파생본이다. 별도 정책 SHA를 고정한 뒤 13:28:47 UTC에 013 검토를 정상 게시했다. 최종 history 표에서 012는 게시 실패한 부분 기록으로 표시한다.

복구 영수증은 `artifacts/next5/maintenance/NATIVE_REVIEW_PUBLICATION_REPAIR/RESOLVED.json`이다. 원래·새 source, 정책, fixture, 부분 기록 및 성공한 013의 SHA를 연결한다. 당시 BOARD/CURRENT/013의 바이트 일치와 root history 반영을 확인했다. 모든 기존 과학적 행의 값과 최고 점수도 원본과 같았다. 이후 검토로 BOARD가 바뀔 수 있으므로 013의 immutable 파일이 해당 시점의 증거다.

검토 뒤의 임시 확인 명령에서 JSON Decimal을 PowerShell Double 리터럴과 비교해 잘못 실패한 경우와 Decimal에 부적합한 `R` 문자열 포맷 오류도 있었다. 원래·현재 JSON 값과 저장된 십진 문자열을 직접 비교해 실제 점수 변경이 없음을 확인했다. 이 확인 명령의 오류를 추가 학습 실패로 세지 않는다.

최종 실패 분류에는 native 게시 오류 1건을 별도로 추가한다. 따라서 실제 실행 오류는 supervised Python 3건과 native 게시 1건의 총 4건이며, 원래 Python queue의 quarantine 11개라는 수치는 바뀌지 않는다. 새 fit·예측·draw·Shadow 사용은 없고, 이 복구를 위해 Python 프로세스를 실행하지 않았다.
