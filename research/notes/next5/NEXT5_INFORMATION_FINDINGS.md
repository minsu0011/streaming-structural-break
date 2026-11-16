# NEXT5 동일 시간 순위의 보완성과 손상

2026-09-11 12:37:17 UTC에 `artifacts/next5/information/RESULTS.json`이 PASS로 생성됐다. 19개 full 후보의 5개 fold와 M2 screen의 2개 fold, 총 97개 후보·fold 범위를 분석했다. 이는 추가 학습 97회를 뜻하지 않는다. 새 fit과 bootstrap draw는 0이다. Full 후보 사이의 171개 대응 차이도 원래 고정된 2,000개 draw로 계산했다.

## 지표의 정의

각 fold 안에서 같은 online time의 모든 positive–negative pair를 비교했다. 올바른 순위 credit은 1, tie는 0.5, 반대 순위는 0이다. 각 모델의 credit 증가량과 감소량을 따로 합산해 TS-AUC 차이와 일치하는지 확인했다. Sampling한 pair로 추정한 결과가 아니다. 아래 credit 수치는 각 fold에서 전체 pair 수로 정규화한 뒤 다섯 fold를 동일 가중 평균한 값이다. M2만 screen의 두 fold 평균이다.

각 full 후보는 같은 425,658,131개 pair를 재사용했고 M2 screen은 26,443,176개 pair를 비교했다. 후보마다 독립적인 새 데이터나 pair를 확보한 것이 아니며, 불확실성의 resampling 단위는 pair가 아니라 원래 시계열이다.

상관 표의 Pearson은 fold·time 안에서 중심화한 출력의 cross-product를 모은 값이다. Spearman은 각 fold·time 안의 average-tie rank를 상대 순위로 정규화한 뒤 cross-product를 모은 값이다. 각각의 time별 상관계수를 단순 평균한 통계가 아니다. 동일 시간의 순위 관계를 보되, 상관이 낮다는 사실을 유용한 독립 신호의 증명으로 사용하지 않는다.

## 실제 결과

| 후보 | 범위 | BL80–expert rank 상관 | BL80–composite rank 상관 | Composite credit 증가 | Composite credit 감소 | 평균 TS-AUC Δ |
|---|---|---:|---:|---:|---:|---:|
| X2 | full, 5 folds | 0.6748937188 | 0.9964253232 | 0.01100469648 | 0.00980904646 | +0.001195650019 |
| S4B | full, 5 folds | 0.6592431053 | 0.9965257558 | 0.01072611734 | 0.00980880447 | +0.000917312871 |
| R2 | full, 5 folds | 0.6433792152 | 0.9958408335 | 0.01037649331 | 0.00948732133 | +0.000889171982 |
| P1 | full, 5 folds | 0.6169230720 | 0.9952660466 | 0.01100032755 | 0.01023972286 | +0.000760604686 |
| W2 | full, 5 folds | 0.5750546905 | 0.9897880807 | 0.01199752512 | 0.01127658071 | +0.000720944411 |
| E1 | full, 5 folds | 0.5187962832 | 0.9963369373 | 0.00996597005 | 0.00995145888 | +0.000014511168 |
| M2 | screen, 2 folds | −0.0264847268 | 0.9917940831 | 0.00924441584 | 0.01220895852 | −0.002964542688 |

X2의 0.01100469648 증가와 0.00980904646 감소는 약 1.10047%와 0.98090%의 pair credit에 해당한다. 차이가 +0.001195650019인 것이며 1.1%를 그대로 순수 개선으로 보고하면 안 된다. Tie의 half credit 때문에 이 값은 단순히 바뀐 pair 개수의 비율과도 구별한다.

X2 expert 자체의 standalone TS-AUC는 0.6060824242다. Expert 증가 credit 0.1092295429보다 감소 0.1235123288이 더 크다. 고정 혼합에서는 작은 순증가가 있지만, expert의 모든 rescue를 손상 없이 취할 수 있는 모델은 만들거나 검증하지 않았다. `expert_rescue_unrealized_by_fixed_blend_pair_fraction`의 X2 fold 평균 0.09846300042는 그런 실현 가능한 oracle의 점수가 아니다.

X2 composite의 B2와의 rank 상관은 0.9734608818, D2와는 0.9971237308이다. 이 높은 상관과 작은 winner gap을 포함하면 “새 독립 expert 확보”라는 확증보다 제한된 보완성이라는 해석이 맞다. 특히 M2는 거의 무상관이어도 standalone 0.4803233164, 혼합 Δ −0.0029645427로 screen에서 탈락했다. E1 역시 비교적 낮은 expert 상관에도 증가와 손상이 거의 상쇄됐다.

19개 full composite의 BL80와의 rank 상관은 0.9897880807–0.9976119785였다. 0.9 BL80 기반 조합이 대부분이라는 설계와 함께 읽어야 한다. S4A는 0.8 augmented J + 0.2 C205라는 다른 구성이다. 상관·credit 분해는 가중치 sweep 또는 새 router를 승인하거나 전체 적응적 선택을 교정하지 않는다.

## 원 자료

- `EXACT_PAIR_DECOMPOSITION.csv`: fold별 증가·감소 credit과 점수 parity.
- `EXACT_JOINT_CREDITS.csv`: BL80/expert/composite의 3×3×3 credit 조합별 정확한 pair 수.
- `SCORE_CORRELATIONS.csv`: 같은 fold·time 안의 pooled Pearson/rank 상관.
- `PAIR_COVERAGE.csv`: 각 후보에서 실제 비교한 pair 수와 full/screen 범위.
- `ALL_CANDIDATE_PAIRWISE_GAPS.csv`: 기존 2,000개 draw로 만든 171개 full 후보 간 차이. Selection-adjusted 구간이 아니다.

모든 파일은 `artifacts/next5/information/` 아래에 있으며 `RESULTS.json`에 SHA가 고정돼 있다. M2의 작은 screen 모집단을 full RP 결과와 같은 조건으로 비교하지 않는다. 상관, 훈련 tree 사용 비중, 동일 시간 pair-rescue, held-out Δ는 서로 다른 증거다.
