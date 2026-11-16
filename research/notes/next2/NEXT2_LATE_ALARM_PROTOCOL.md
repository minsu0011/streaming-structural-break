# 장기 무변화 보정의 늦은 변화 검출 대가

β=0/0.5/1 age odds 보정은 같은 t의 순위를 유지하면서 장기 무변화 경보를 줄였다. 그러나 nested DEV에서는 실제 변화의 clean first detection도 감소했다. 이를 확인하는 보조 진단으로 Gaussian(N0)과 ARCH(N2) 배경, variance×2, 변경 시점 4096/49152, O=65536의 전체 2×2 조합을 고정한다. 각 조합은 128개의 control/intervention seed 쌍을 사용한다.

기존 H=2048, burn-in=1024, 배경 계수, variance 배수의 의미와 생성 recurrence를 유지한다. 새 RNG partition=4를 사용하고 원래 suite와 cross-background partition=3 결과를 변경하지 않는다. 모델은 동결 all-DEV J102/Y001/B104/C205와 이미 고정한 J102/C205 80/20이다. 새 fitting, β 선택, threshold 재선택은 없다. β별 threshold는 원래 독립 1024-point calibration 표본에서 계산한 값을 재사용한다.

각 조합에서 원래 TS-AUC와 β별 prebreak alarm / clean postbreak first detection / 조건부 delay / null eventual alarm을 보고한다. 조건부 delay만 줄어들고 검출이 사라지는 경우를 개선으로 해석하지 않는다. 같은 seed 쌍을 통째로 유지한다. 네 조합과 겹치는 잠재 난수는 독립 반복으로 취급하지 않는다. 이 보조 진단은 경쟁 데이터 성능이나 승격 조건의 대체물이 아니다.

추가로 각 트리의 최대 leaf 값을 합한 보수적인 margin 상한으로 모든 가능한 입력의 score 상한을 계산한다. 이 상한과 고정 threshold를 이용해 age 보정 후 경보가 더 이상 가능하지 않은 충분조건을 구한다. 상한은 여러 트리의 최댓값이 같은 입력에서 달성된다고 보장하지 않으므로 실제 최대 score나 정확한 검출 수명을 뜻하지 않는다. float32 score 경계도 보수적으로 반영한다.
