# DEV 결과 요약 그림

`artifacts/real/figures/DEV_RESULTS_OVERVIEW.pdf`는 공유·확대용 벡터 PDF이며, 같은 내용의 PNG와 SVG도 포함한다. 그래프의 정확한 입력값과 출처 SHA256은 같은 폴더의 `FIGURE_INPUTS.json`에 있다. 최종 동결한 DEV 결과만 사용하며 reduced 점수는 그림에 넣지 않는다.

- A: B0–B7 및 가장 강한 보조 통계 검출기. 평균 DEV 점수다.
- B: 최종 세 역할의 고정 5개 폴드. 최고 평균 후보와 주력은 사전 승격 기준 때문에 다를 수 있다.
- C: Online step별 성능. B 패널과 같은 색으로 주력·안정성·속도 백업을 표시했다.
- D: Break-age별 구별력. 초기 구간은 약하며, 이 곡선을 경보율이나 검출 지연으로 해석하면 안 된다.
- E: 동일 설정의 AR4 구성 모델에서 A/B/C/D를 비교한다. 최종 두 모델 평균 자체의 ablation은 아니다.
- F: 실제 전체 DEV refit 모델의 계산 비용. 진한 막대는 직접 측정한 point-only 시간이고 빗금 막대는 historical 초기화까지 합한 추정이다. 긴 스트림은 실제 DEV 값 버퍼를 반복하며, 초기화는 128개 historical을 사용한다. Cloud 속도 보증은 아니다.

원시 데이터나 ML 라이브러리를 다시 실행하지 않고 저장된 숫자만으로 그림을 만들었다. 시각화 패키지는 별도 `.figure-venv`에 설치했고, 학습용 `requirements-lock.txt`, `.venv`, `.repro-venv`는 변경하지 않았다. 그림 전용 패키지 버전은 `docs/requirements-figures.txt`에 고정했다.

```powershell
py -3.13 -m venv .figure-venv
.\.figure-venv\Scripts\python.exe -m pip install -r docs/requirements-figures.txt
.\.figure-venv\Scripts\python.exe scripts/render_result_figure.py
```

위 렌더링은 포함된 `FIGURE_INPUTS.json`만 읽으므로 raw나 인증이 필요 없다. 그림 입력을 원래 보고서에서 다시 만들려면 학습 환경에서 `scripts/prepare_result_figure.py`를 실행한다. 재렌더링하면 생성 파일의 해시가 달라질 수 있으므로 원본 ZIP의 파일을 보존하고 별도 폴더에서 진행한다. 저장된 그림의 바이트는 `FIGURE_RENDER_REPORT.json`과 최종 ZIP manifest로 검증한다.
