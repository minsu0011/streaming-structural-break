# 고정 감소 특성과 학습 점수의 방향 차이

별도 최종 감사에서 완료된 `scale_direction_response/complete_v1/`의 9,216개 특성 행과 8,160개 예측 반응 행을 읽었다. 아래는 사후 해석용 age 50 이상 구간의 요약이다. 모든 조건은 원래 256개 paired generating seed를 사용한다. 같은 seed의 변화 경로와 무변화 경로의 평균 차이를 보며, TS-AUC의 positive/negative 시계열 순위와는 다른 양이다.

V3에서 Gaussian lower w128의 raw/conditional 평균 반응은 +9.5514/+5.8772, Student-t5 lower는 +5.5974/+4.0554였다. 네 좌표 모두 각 seed의 반응이 양수였다. PIT lower EWMA128도 raw/conditional 모두 양수였고 upper는 모두 음수였다. V4와 CB_N3_050에서도 선택한 lower 좌표의 방향은 동일했다. 이 특성들은 단위가 다르므로 크기끼리 중요도를 비교하지 않는다.

그러나 학습한 expert의 반응은 달랐다. 다음 표는 다섯 고정 학습 파일의 평균 점수 반응(변화−무변화)이다.

| 조건 | BL80 | S4B expert | R2 expert | X2 expert |
|---|---:|---:|---:|---:|
| V3 | −0.0235088896 | −0.0461482666 | −0.0207893164 | −0.0476781990 |
| V4 | +0.0159405943 | −0.0188976186 | −0.0035942009 | −0.0300812904 |
| CB_N3_050 | −0.0098882228 | −0.0047940678 | +0.0166839324 | −0.0145100393 |

V3의 S4B/R2/X2 expert는 다섯 파일 모두 음수였다. CB_N3_050에서는 R2 expert가 다섯 파일 모두 양수였지만 X2 expert는 모두 음수였다. lower evidence의 반응이 존재한다는 사실만으로 learner가 이를 모든 배경에서 유리하게 사용한다고 결론낼 수 없다. 학습된 변환·특성 결합과 배경별 반응을 더 분리해야 한다는 관찰이며, 특정 특성이나 tree 사용 비중의 인과적 효과를 증명하지 않는다.

점수 반응과 TS-AUC를 혼동하면 안 된다. 예를 들어 R2의 V3 평균 score response는 음수지만 원래 full synthetic TS-AUC 차이는 평균 +0.0013012489였다. 고정 sample에서의 paired amplitude와 서로 다른 시계열 간 within-time ranking은 서로 다른 비교다. 기본 V3/V4의 X2 TS-AUC 손실도 원래 전수 점수표에서 별도로 확인했다.

원래 여섯 배경의 선택한 48개 특성 행과 102개 예측 집계, 각 파일 원값·입력 SHA는 `artifacts/next5/final_audit/interpretation_reviews/SCALE_DIRECTION_RESPONSE_READING.json`에 있다. 새 학습·예측·seed·신뢰구간을 만들지 않았다. lag 512는 관측 가능한 171개 seed에 한정되지만, 이 글의 age 50 이상 요약은 256개 모두를 사용한다. 이 검토에 승격 권한은 없다.
