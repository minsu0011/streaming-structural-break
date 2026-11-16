# Structural Break Real-Time NEXT4 인계

NEXT4_ROBUSTNESS_ONLY_NO_PERFORMANCE_PROMOTION

[42개 답변](00_GPT_READ_FIRST_NEXT4.md), [실험 표](MODEL_RESULTS_NEXT4.csv), [고정 역할](DEV_CANDIDATES_NEXT4.json), [최종 감사](FINAL_AUDIT_NEXT4.json)를 먼저 읽는다.

원래 primary metric은 grouped 5-fold official TS-AUC의 평균이다. 관측 값/시계열 ID/시간/label의 역할, screen/full, standalone/composite, 조건부 H slice/실제 H 제외 학습을 섞지 않는다. 실제 선택 근거는 고정 정책과 원래 OOF이다.

실험 메뉴는 A–H 8개 family/18개 slot으로 동결했다. 16개 screen, 14개 full, E2·G1 screen 실패, F1 중복·F2 조건부 종료를 모두 보존했다. 56개 기존 nested base를 재사용했고 16개 새 meta fit은 honest inner OOF를 사용했다.

B2 평균 0.6245720862584431 (Δ 0.0013036734411336682)는 숫자 개선 문턱을 넘었으나 CI [-0.000018936917, 0.002632276899]로 탈락했다. ALT 3/3 양수, 직접 runtime 통과가 이 결정을 바꾸지 않는다. 별도 NEXT4_ROBUSTNESS는 D2_DISCRETE_SCALE_MIXTURE, 평균 0.6235961690, Δ 0.0003277562. Primary champion 교체는 아니다.

강건성 역할 D2_DISCRETE_SCALE_MIXTURE: 원래 SCALE Δ 0.0089358282, CI [0.005362693966, 0.012762924907], 변화 시계열 132개. 원래 전체 CI는 [-0.000273254935, 0.000909263558]이다. 상세 ALT/H·기전·반례 한계는 [채택 검토](docs/NEXT4_FINAL_TRANSFER_REVIEW.md)에 남겼다.

중요한 추가 근거: [B2 실패 해석](docs/NEXT4_B2_FAILURE_INTERPRETATION.md), [B1 강건성 범위](docs/NEXT4_B1_ROBUSTNESS_INTERPRETATION.md), [입력 부호 진단](artifacts/next4/sign_reflection/RESULTS.json), [방법 경계](docs/NEXT4_METHOD_BOUNDARIES.md), [프로토콜 예외](docs/NEXT4_PROTOCOL_SCOPE_NOTES.md).

실제 all-DEV 강건성 파일의 512쌍 이후 평균 signed excess 차이: O1 Δ -0.019297926753779393 CI [-0.01986783582471503, -0.01868381102408989]; OB16 Δ 0.004762802824188839 CI [0.003946145333384265, 0.005599417345865732]. 음수는 동일 반례에서 기준보다 오염이 작다는 뜻이다. D2는 단발 이상치에서는 개선했지만 반복 이상치는 악화했다. fold-0에서의 반복 이상치 개선이 실제 전체 학습 파일에는 유지되지 않았다. 전반적인 outlier repair로 주장하지 않는다.

[최종 채택 범위와 손실](docs/NEXT4_FINAL_TRANSFER_REVIEW.md)을 함께 읽는다.

[입력 부호 민감도 해석](docs/NEXT4_SIGN_REFLECTION_INTERPRETATION.md)과 [D2의 SCALE/전체 점수 분할별 도표](artifacts/next4/figures/research_005/D2_DISCRETE_SCALE_MIXTURE_TRANSFER.png)도 포함했다.

개발 중 unit fixture 오류와 두 실제 worker 실패를 원본과 함께 남겼다. B2 cold import의 캐시 mkdir은 기존 read-only bootstrap을 설치하는 추가 entry로 해결했다. 합성 worker의 Python 3.13 native access violation 원인은 확정하지 못했다. timed stack dump만 끄고 동일 과학 계산을 이어서 기존 116개 파일·18개 분석을 보존했다. 원래 실패를 PASS로 바꾸지 않았다.

실행 환경은 requirements-lock.txt의 정확한 63개 pin이다. Figure 환경은 기존 requirements-next2-figures-lock.txt를 썼다. Python 3.13.2, Windows CPU에서 검증했다. 사용자 staged 변경 9개와 기존 NEXT3 중단 상태를 보존했다.

전체 ZIP은 소스·설정·집계 결과·검증 receipt·필요 모델을 포함한다. raw 경쟁 데이터, 실제 point OOF 전체, feature cache, 환경, credentials, 과거 ZIP은 포함하지 않는다. 따라서 학습 재실행에는 RAW_DATA_REQUIRED_NEXT4.json의 SHA와 일치하는 raw X/y가 필요하다. 포함된 모델의 추론은 raw 없이 가능하며, 복사한 payload에서 새 Python 프로세스로 확인했다. 동일 pin의 기존 interpreter를 재사용했으므로 새 환경 설치 검증과는 다르다.

배포 호출은 선택 JSON의 entry에서 `infer(datasets, model_directory_path)`를 사용한다. datasets는 (historical array, online iterator)이며 첫 출력은 None, 이후 현재 관측마다 scalar 하나다. 새 프로세스 read-only 검사에서는 Python bytecode 쓰기를 `-B`로 끄며 entry가 프로젝트의 선택적 Numba cache 쓰기를 억제한다.

기존 NEXT3는 사용자가 중단한 상태를 유지한다. NEXT4의 최종 감사가 과거 NEXT3의 미완료 최종 감사를 완료로 소급하지 않는다. Desktop 20개 파일 폴더·ZIP과 전체 프로젝트 ZIP의 CRC 및 각 member SHA는 외부 delivery receipt에서 검증한다.
