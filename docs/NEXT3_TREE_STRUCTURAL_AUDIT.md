# NEXT3 실제 트리 구조와 경계값 감사

고정 PERFORMANCE의 J102/C205 숫자 트리를 original compact 표현과 E003 packed 표현으로 대조했다. 새 모델·runtime family·threshold·학습·경쟁 데이터 평가를 추가하지 않았다.

## 실제 결과

두 component는 각각 100개 트리, 2,900개 노드(분기 1,400개·leaf 1,500개)다. 총 5,800개 노드의 실제 사용 필드와 feature column의 과학적 이름이 같았다. 분기 threshold와 leaf 값의 float64 비트, missing 방향, 자식 연결, root 순서와 bias 비트도 일치했다. J102가 실제 사용하는 48개 feature와 C205의 24개 feature를 확인했으며, J102의 legacy/새 feature 출력 위치도 대조했다.

| Component | dtype | 시도한 인공 probe | 실행한 full-forest margin 비교 | ancestor와 충돌하여 제외 | 도달한 leaf |
|---|---|---:|---:|---:|---:|
| J102 | float32 | 9,900 | 9,679 | 221 | 1,500 / 1,500 |
| J102 | float64 | 9,900 | 9,679 | 221 | 1,500 / 1,500 |
| C205 | float32 | 9,900 | 9,687 | 213 | 1,500 / 1,500 |
| C205 | float64 | 9,900 | 9,687 | 213 | 1,500 / 1,500 |

각 leaf의 ancestor 제약을 만족하는 witness를 구성했다. 각 분기에서는 dtype으로 변환한 threshold, 양쪽 인접 representable 값, NaN, ±infinity를 시도했다. 원래 float64 threshold가 float32로 정확히 표현되지 않을 수 있으므로 float32의 `CAST_THRESHOLD`를 항상 원래 threshold와 같은 값이라고 부르지 않는다.

총 38,732개 full-forest margin이 바이트 단위로 같았다. ancestor 제약과 충돌하는 868개 시도는 실행 coverage에서 제외하고 `ARTIFICIAL_PROBE_INDEX.csv`에 이유를 남겼다. 이 숫자는 39,600번의 계획된 시도에서 실제 실행한 비교를 뺀 값이며, 868개 leaf가 미검증이라는 뜻은 아니다. 모든 leaf는 두 dtype 각각에서 도달했다.

이 입력은 인공 feature vector다. 임의의 vector가 실제 causal H/O 이력에서 생성될 수 있다고 주장하지 않는다. recurrent feature 계산과 current-point 제약은 기존 실제 스트림 parity 및 별도 최종 API 검사로 확인한다. 이 감사는 compiler의 형식적 증명, hidden-test 품질, 새 robustness 개선이나 latency benchmark가 아니다.

## 보존한 첫 실패와 수정

처음 17개 primitive 테스트는 통과했으나 M667의 실제 모델 감사는 witness 검증 단계에서 실패했다. Python oracle에서 NumPy float32 scalar와 Python float threshold를 직접 비교했기 때문이다. 설치된 NumPy 2.5.3에서 값 0.25와 바로 아래 float64 값 `0x1.fffffffffffffp-3`의 `<=` 비교는 직접 비교 시 True, 두 값을 Python float로 명시 변환하면 False였다. Python scalar를 약한 dtype으로 취급하는 NumPy promotion 규칙과 부합한다. [NumPy NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)

실제 두 compiled tree engine은 float64 threshold를 유지한다. 검증용 oracle에 명시적인 float64 비교를 적용하고 인접 float64 threshold 회귀 사례를 추가했다. 수정 후 19개 primitive 테스트와 M669의 전체 실제 모델 감사가 통과했다. 원래 helper·테스트·driver·lock은 `tree_certificate_scalar_oracle_correction_001/`에, 실패 로그는 원래 M667 job 폴더에 보존했다. runtime·모델·과학 gate는 바꾸지 않았다. 실제 worker가 이미 시작된 뒤의 사전 교체 시도는 guard가 거부했으므로 M667을 실행 없는 superseded job으로 분류하지 않았다.

근거는 `artifacts/next3/maintenance/TREE_STRUCTURAL_AUDIT_PROTOCOL.json`, `TREE_STRUCTURAL_AUDIT_IMPLEMENTATION_LOCK_CORRECTION_001.json`, `tree_structural_maintenance_v2/RESULTS.json`, `TREE_CERTIFICATE_PRIMITIVE_TESTS_CORRECTED.xml`이다. 원래 M667의 `tree_structural_maintenance_v1/`과 job failure도 남아 있다.

최종 별도 감사에서는 실제 새 raw 재학습으로 만들어진 PERFORMANCE 모델에 같은 고정 검증기를 적용하도록 계획했다. 그 실행 결과는 `tree_structural_final_audit/RESULTS.json`이 실제 생성된 뒤에만 최종 근거로 인정한다.
