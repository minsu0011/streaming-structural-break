# NEXT4 유한 실험 메뉴의 종료 판단

이 판단은 이번에 등록한 feature·learner·혼합과 고정 DEV 분할에 한정한다. CI가 0을 포함하는 것을 효과가 정확히 0이라는 증명으로 해석하지 않는다. 원래 성능 gate를 통과하지 못한 후보에 새 seed, 가까운 blend weight, tree depth를 추가해 결과를 고르지 않았다.

| Family | 실제 screen/full | 원래 full 평균 Δ | 종료·유지 근거 |
|---|---:|---|---|
| A: dependence/whiteness | 3/3 | A1 +0.00015016, A2 +0.00022905, A3 +0.00011917 | 세 원래 전체 CI가 0을 포함한다. A2 SCALE의 양수 CI는 확인됐지만 고정 heavy-tail 감소 반례 두 개를 통과하지 못했다. 등록된 세 버전의 추가 적합을 종료했다. |
| B: scale asymmetry | 2/2 | B1 +0.00054783, B2 +0.00130367 | B2가 탐색 최고지만 원래 CI 하한 -0.00001894로 performance 실패다. B1은 SCALE·감소 반례가 좋아도 H 제외 SCALE이 불확실하고 burst 오염이 나빠졌다. 기존 후보의 진단만 완료하며 새 조합은 만들지 않았다. |
| C: distribution | 3/3 | C1 +0.00003458, C2 +0.00004793, C3 +0.00011757 | H 고정 bin의 Hellinger, JS, shape-ratio 세 버전 모두 전체 CI가 0을 포함한다. C3의 SCALE 이득은 heavy-tail 감소 두 반례에서 유지되지 않았다. |
| D: sequential likelihood | 3/3 | D1 -0.00019766, D2 +0.00032776, D3 -0.00014521 | D2의 제한된 SCALE 강건성 역할만 실제 파일 검증과 최종 감사 대상으로 유지한다. D3는 가장 강한 heavy-tail 감소에서 고정 조건 실패이며, absorbing hazard의 null 누적 압력도 진단했다. 전체 성능 승격은 없다. |
| E: pairwise objective | 2/1 | E1 -0.00004675 | E2 screen Δ -0.00313554로 full을 하지 않았다. E1의 학습 pair loss 개선은 held-out TS-AUC 개선으로 이어지지 않았다. 새 loss scale/ridge sweep을 하지 않았다. |
| F: multi-expert | 0/0 | 새 점수 없음 | F1의 72/18/10은 기존 0.9 BL80+0.1 expert와 중복이다. F2는 별도 독립 expert admission을 만족하는 세 번째 성분이 없어 조건부 종료했다. 미실행을 실패한 score처럼 기록하지 않는다. |
| G: historical context | 2/1 | G2 +0.00000177 | G1 screen Δ -0.03482594로 종료했다. G2는 H-only context가 J/C 차이 계수에 작용하는 honest nested 결합이나 전체 개선 근거가 없다. H feature의 높은 split gain을 인과적 실패 원인으로 단정하지 않는다. |
| H: compact temporal state | 1/1 | H1 -0.00068465 | 고정 두 단계 선형 recurrence와 작은 readout을 평가했다. 원래 전체 point gain은 음수이고 CI는 0을 포함한다. 분산 감소 및 mixed 합성에서도 손실을 확인했다. 더 큰 recurrent network로 즉시 확장하지 않았다. |

총 8개 선언 family 중 실제 새 DEV score가 있는 family는 7개다. Screen 16개, full 14개, 선택용 fit 102회(32 screen+70 full)다. ALT/H는 기존 B1/B2/D2 각각 20회로 총 60회이며 새 full candidate identity로 중복 세지 않는다. 그 외 all-DEV materialization, negative control, 필요할 경우 실제 fresh raw 재현은 개별 fit receipt로 별도 집계한다.

B2와 B1의 원래 차이 +0.00075585도 CI [-0.00051849,+0.00195732]로 불확실하다. 14개 원래 전체 CI가 모두 0을 포함하고 반복 DEV 탐색을 했으므로 saturation은 HIGH다. 평균 순위가 높은 모델이라는 사실과 안정적인 새 신호를 발견했다는 주장을 구별한다.

D2에서도 한계가 남는다. 전체 H 제외 평균과 DEPENDENCE의 point gain은 소폭 음수다. 실제 all-DEV 파일은 SCALE과 네 분산 감소 반례를 확인했지만, 반복 이상치 뒤 평균 excess는 +0.0047628로 악화했다. 따라서 모든 분포·이상치에서 좋아지는 모델로 결론 내리지 않는다.

수치 근거는 완료된 연구 표, 원래 paired 통계, [고정 정책](../../../configs/next4/RESEARCH_POLICY_LOCK.json), F 종료 기록에 있다. 최종 역할은 `DEV_CANDIDATES_NEXT4.json`과 이후 실제 `FINAL_AUDIT_NEXT4.json`을 함께 확인한다.
