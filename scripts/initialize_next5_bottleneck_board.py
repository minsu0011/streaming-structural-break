"""Initial subjective priorities, separated from measured ResearchPool evidence."""
from pathlib import Path
import csv
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress


def main():
    proposals=[
      ('B1','scale generalization','ACTIVE',.002,4,5,3,3,2,'S',
       'NEXT4 B2 overall +0.00130367 near-miss; D2 SCALE +0.00893583; neither supports a new overall primary yet.'),
      ('B2','persistent versus transient shock','ACTIVE',.0015,4,5,2,3,2,'P',
       'NEXT4 D2 actual repeated-outlier mean excess +0.00476280; single-outlier improvement did not establish repeated-shock safety.'),
      ('B3','heavy-tail null calibration','PLANNED',.0012,3,5,3,3,2,'R',
       'B2 heavy-tail variance-decrease CB050 delta -0.05363612; robust H-only null transformations need real ResearchPool support.'),
      ('B4','long-null accumulation','PLANNED',.0007,3,4,3,3,2,'E,W',
       'D2 16k null alarm rate not a nominal alpha guarantee; bounded windows and fixed mixtures can be audited without claiming anytime validity.'),
      ('B5','independent signal discovery','PLANNED',.0015,2,4,4,3,2,'M,X',
       'Distinct bounded signed evidence is a finite low-budget question; past weak location variants remain closed.'),
      ('B6','NEXT5 selection uncertainty','MONITOR',.0007,5,5,4,4,2,'ALL',
       '8k DEV historically reused. Prospective Shadow reduces NEXT5 adaptivity but is not an independent original LOCAL SEAL.')]
    rows=[];now=utc_now()
    for code,name,status,impact,evidence,relevance,risk,complexity,runtime,families,reason in proposals:
        rows.append({'timestamp':now,'bottleneck':code,'question':name,'status':status,
          'estimated_TS_AUC_impact_subjective_prior':impact,'evidence_prior_1_to_5':evidence,
          'generalization_relevance_1_to_5':relevance,'overfit_risk_1_to_5':risk,
          'implementation_complexity_1_to_5':complexity,'runtime_cost_1_to_5':runtime,
          'priority_score':impact*evidence*relevance/(risk*complexity*runtime),
          'families':families,'evidence':reason,'NEXT5_measured_gain':None})
    assert sum(r['status']=='ACTIVE' for r in rows)==2
    record={'created_utc':now,'status':'INITIAL','rows':rows,
       'interpretation':'Impact and 1-5 ratings are subjective resource-allocation priors, not measured gains or probabilities.',
       'reference_source_sha256':digest(BASE/'preflight/FROZEN_REFERENCES.json'),
       'decision':'S and P first, then W/E/R and the finite M menu; X only after frozen prerequisites.',
       'next_review':'45-60 minutes after this timestamp, using completed ResearchPool results and failure ledger',
       'shadow_opened':False,'new_seal_accesses':0}
    atomic_json(BASE/'bottlenecks/history/001_INITIAL.json',record,immutable=True)
    atomic_json(BASE/'bottlenecks/CURRENT.json',record,immutable=True)
    with (ROOT/'BOTTLENECK_BOARD_INITIAL_NEXT5.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/'BOTTLENECK_BOARD_INITIAL_NEXT5.md').write_text(
       '# NEXT5 초기 병목 판단\n\n'+
       '우선 작업은 scale 일반화(S)와 지속 변화·일시 충격 구분(P) 두 가지다. NEXT4의 B2는 전체 개선이 작고 bootstrap 하한이 0을 넘지 못했다. D2는 SCALE 보조 신호로 남았지만 실제 배포 모델의 반복 이상치 반응은 개선되지 않았다.\n\n'+
       'CSV의 예상 TS-AUC와 1–5 평점은 자원 배분을 위한 사전 판단이며 측정 성과가 아니다. 실제 NEXT5 평가는 새 ResearchPool 분할에서 재학습한 기준 모델과 비교한다. Shadow와 원래 LOCAL SEAL은 닫혀 있다.\n\n'+
       '45–60분 간격으로 완료된 실험·실패·후보 예산을 검토한다. 구조적 가설 12개 이후 최선의 개선이 0.0007 미만이면 새 과학적 적합을 멈추고 검증·재현·분석을 수행한다.\n',encoding='utf-8')
    progress('Initial bottleneck priorities recorded','Two active questions, explicit subjective impact/risk/cost and historical evidence; next review in 45-60 minutes')


if __name__=='__main__':main()
