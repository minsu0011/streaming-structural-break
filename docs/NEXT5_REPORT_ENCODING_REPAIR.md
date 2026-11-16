# NEXT5 보고서 준비 명령의 인코딩 복구

2026-09-11 18:39–18:41 UTC에 완성된 실제 입력으로 보고서 코드를 메모리에서 실행했다. 최초 명령은 PowerShell의 US-ASCII 파이프가 한국어 검사 문자열을 손상시켜 assertion이 실패했다. 보고서 자체는 UTF-8 원본에서 렌더링됐고 47개 질문·45개 late 조건 행 검사는 이미 통과한 상태였다. 최종 보고서 파일은 쓰지 않았다.

한국어 검사 문자열을 ASCII 소스의 Unicode escape로 바꾼 두 번째 명령에서는 모든 렌더링 검사가 통과해 V4 준비 영수증을 썼다. 이후 내용을 콘솔에 출력하다 기본 cp949가 en dash를 표현하지 못해 종료 코드 1이 발생했다. 따라서 V4는 메모리 내 검사 통과 증거이며 해당 명령 전체가 정상 종료됐다는 증거가 아니다.

세 번째 명령은 입력에 Unicode escape를 쓰고 표준출력을 UTF-8로 지정했다. 모든 검사와 출력이 정상 종료됐으며 종료 코드는 0이다. `artifacts/next5/maintenance/KOREAN_COMPLETE_INPUT_RENDER_PREPARATION_V5.json`이 성공한 준비 영수증이고, 두 실패와 복구의 연결은 `KOREAN_RENDER_HARNESS_ENCODING_REPAIR.json`에 있다. 이전 V4는 보존했다.

보고서 소스·모델·원래 점수·예측·bootstrap draw는 이 인코딩 복구로 바뀌지 않았다. 이는 개발 점검 명령 두 건의 실패이며, 32행 분류표의 범위인 scientific/validation queue 및 native 병목 기록 게시 실패와 구별해 보존한다. 준비 문맥의 test 개수는 `NOT_YET_AUDITED_PREPARATION_SENTINEL`이므로 최종 suite 통과나 감사 완료를 주장하는 영수증이 아니다. 최종 발행은 실제 감사 시간과 모든 검사 완료를 요구한다.

19:10 UTC 갱신: 이후 최종 suite 접근 장치 오류와 성공한 전체 재검사가 추가돼 최종 통합 실패 표는 33행이 됐다. 이 글의 32행은 최초 인코딩 복구 시점의 역사적 범위다. 새 파생 보고서는 실제 성공한 1,363개 test 문맥으로 메모리 내 47개 답변·45개 late 행 검사를 통과했으며, `final_audit/interpretation_reviews/REPAIRED_REPORT_INPUT_READING.json`에 별도 기록했다. V4/V5 준비 영수증과 그 당시 sentinel을 덮어쓰지 않았다.
