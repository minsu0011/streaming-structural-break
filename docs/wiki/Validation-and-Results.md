# 평가와 결과의 의미

## TS-AUC의 단위

NEXT2의 고정 평가 정의는 within-online-time TS-AUC를 pair weight로 결합하고 5개 DEV fold를 평균하는 구조입니다. 시계열 전체 행을 섞은 일반 분류 AUROC와 같은 지표로 취급하지 않습니다.

모델 선택용 원본 fold, 후보 축소용 screen, 대체 split, 합성 stress는 역할이 다릅니다. 한 곳에서 유리한 수치를 다른 검증의 성능으로 옮기지 않습니다.

## 역할별 판단

[decision_gates.py](../../src/next2/decision_gates.py)는 단순화와 성능형의 조건을 나눕니다. 평균·중앙·최악 fold 변화, paired bootstrap, 대체 split과 H regime를 확인하며, engineering 조건도 별도로 반영합니다. 증거가 없을 때는 조건 실패와 구분합니다.

저장된 NEXT2 performance 설정은 serial variance와 dependence 보완을 고정 조합하는 연구 결과입니다. 설정 이름이나 파일 존재만으로 최종 독립 검증을 통과했다고 해석하지 않습니다. 세대 간 평가 조건을 전부 같다고 볼 수 없어 숫자 차이를 통합 개선율로 제시하지 않았습니다.

## 오경보와 지연 경보

점수의 순위가 좋다는 것과 낮은 오경보율로 빠르게 경보한다는 것은 다릅니다. 장기 null과 지연 경보를 별도로 점검해야 합니다. Serial GLR·Bayes는 근사 증거 특징이며 전체 기간의 오류확률을 보장하는 검정 통계가 아닙니다.

## 테스트 범위

합성 시계열 테스트는 상태 갱신과 특징 반응을, 수치 비교는 구현한 수식과 경량 runtime을 확인합니다. 공식 runner 기반 입력 프로토콜과 실제 독립 holdout의 성능은 별도입니다. 연구 후보를 운영 탐지기로 쓰려면 후자의 검증이 필요합니다.
