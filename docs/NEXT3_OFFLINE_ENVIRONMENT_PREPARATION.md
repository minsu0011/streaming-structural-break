# NEXT3 최종 환경 설치 준비

기존 `requirements-lock.txt`의 정확한 63개 버전을 공식 PyPI wheel로 미리 내려받았다. SHA256·ZIP CRC·현재 Windows/Python 3.13 태그·배포 이름/버전을 확인했다. 버전을 바꾸거나 SDK를 실행하지 않았으며 환경 설치와 raw 학습은 아직 수행하지 않았다.

원본 검사 M624는 `py 1.11.0` wheel 내부의 두 vendored 패키지 METADATA도 최상위 배포 메타데이터로 잘못 세어 실패했다. 다운로드한 63개 파일, 원본 검사 source, 실패 log와 attempt는 그대로 남겼다. 수정 검사는 최상위 `.dist-info/METADATA`만 배포 identity로 사용하고, nested metadata는 따로 센다. 일곱 개 단위 검증과 원본 63개 wheel의 재검사를 별도로 수행했다. 바이너리 재다운로드는 하지 않았다.

사용할 cache는 `artifacts/next3/maintenance/PINNED_WHEEL_CACHE_SELECTION.json`의 receipt·SHA256으로 고정한다. 최종 감사에서 새 sibling workspace를 만들면 각 wheel을 보통 파일로 복사하고 다시 hash를 확인한 뒤 새 venv에 `--no-index --find-links`로 설치한다. `pip check`와 raw train/infer는 별도 필수 절차다. 데이터나 이전 feature cache를 대신하는 준비 작업이 아니다.

검증된 cache가 있는데 파일·버전·receipt가 달라지면 설치를 중단한다. 초기 cache 미준비 환경에서만 원래의 공식 PyPI 설치 경로를 허용한다. 이번 세션은 cache가 있으므로 정상 실행 경로가 offline 설치다. 원래 준비 script는 `artifacts/next3/maintenance/offline_reproduction_setup/original_prepare_next3_independent_reproduction.py`에 보존했다.

wheel 바이너리는 로컬 재현용 cache와 새 환경 준비 폴더에만 둔다. GPT 전달 ZIP에는 version/hash 검증 receipt를 넣고 바이너리·venv·raw competition 데이터는 제외한다. 최종 raw 재현의 11시간 10분 시간 gate와 별도 감사 시작 조건은 바꾸지 않았다.
