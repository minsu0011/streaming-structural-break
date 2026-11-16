"""Make the twenty handoff answers state measured results directly."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now


source=(ROOT/'scripts/next9/write_next9_scientific_reports_v2.py').read_text(encoding='utf-8')
insert="""        paired=rows('PAIRED_UNCERTAINTY_NEXT9.csv')
        def paired_result(candidate,reference):
            return next(r for r in paired if r['mode']=='NORMAL' and r['domain']=='ALL' and r['candidate']==candidate and r['reference']==reference)
        graph_tabular=paired_result('M6','TABULAR_SAME_INPUT');state_increment=paired_result('M3','M2')
        def self_count(mode,references):
            count=0
            for j in candidates:
                good=[]
                for fold in range(5):
                    matches=[r for r in self_rows if r['mode']==mode and r['domain']=='ALL' and int(r['feature'])==j and int(r['fold'])==fold
                             and r['contrast'] in ['T2_SELF_RUN_SLOPE_H__vs__'+ref for ref in references]]
                    good.append(len(matches)==len(references) and all(r['ci_low'] not in ('','None') and float(r['ci_low'])>0 for r in matches))
                count+=sum(good)>=4
            return count
        self_normal=self_count('NORMAL',['T0_PRIOR','T1_SELF_MARKOV'])
        self_early=self_count('EARLY',['T0_PRIOR','T1_SELF_MARKOV'])
        self_tree=self_count('NORMAL',['T3_SAME_INPUT_DEPTH3'])
        stress_failures=science['C']['promotion_score_gates']['major_domain_failures']
        stress_text='; '.join(f\"{r['mode']} {r['domain']}: 평균Δ={number(r['mean'])}, 최악foldΔ={number(r['worst'])}\" for r in sorted(stress_failures,key=lambda r:r['mean'])[:3]) or '사전고정 점수 기준을 위반한 지원 충분 TIME/H 영역 없음'
"""
assert "        answers=[" in source
source=source.replace("        answers=[",insert+"        answers=[",1)
replacements={
"'자기 상태전이 예측은 SELF_TRANSITION_RESULTS.csv의 T2 대 T0/T1 및 같은 입력 T3 비교로 판단한다. B 미통과로 M5 분기가 빠진 결과를 전이 모델의 용량 부족으로 해석하지 않는다.'":
"f'자기 상태전이 T2가 T0·T1을 모두 이긴 개별 95% CI가 4/5 fold 이상 나온 후보는 NORMAL {self_normal}개, EARLY {self_early}개다. 같은 입력 T3도 이긴 후보는 NORMAL {self_tree}개다. 이는 보조 상태 예측 결과이며 break 모델의 이득으로 바꾸어 해석하지 않는다.'",
"'state-only·고정 비선형 O1을 모두 이겨야 variable 효과다. 후보별 원시 방향 차이와 이 추가효과는 CONDITIONAL_ORIENTATION_RESULTS.csv에서 분리했다.'":
"f'조건부 variable 효과로 인증된 피처는 {cert[\"certified_variable_count\"]}개다. 상태 입력을 추가한 M3−M2 평균Δ={number(state_increment[\"point\"])}, 95% CI=[{number(state_increment[\"ci_low\"])}, {number(state_increment[\"ci_high\"])}]. 이 모델 차이만으로 개별 F의 방향 반전이나 인과 효과를 증명하지 않는다.'",
"'M6 대 TABULAR는 입력이 같다. M5 대 TABULAR는 M6=M5인 경우에만 같은 입력이다. 비유의는 동등성 증거가 아니며, 연구 signal 비열등 한계는 사전고정 .001이다.'":
"f'같은 입력 M6−TABULAR 평균Δ={number(graph_tabular[\"point\"])}, 95% CI=[{number(graph_tabular[\"ci_low\"])}, {number(graph_tabular[\"ci_high\"])}]. 인증 cross edge={graph[\"cross_edge_count\"]}개다. edge가 없으면 이 비교는 그래프의 추가 정보 효과를 추정한 것이 아니다. 비유의는 동등성 증거가 아니다.'",
"f'M7 평균 TS-AUC={normal_m7[\"mean_TS_AUC\"]}, BL80={normal_m0[\"mean_TS_AUC\"]}. 점수 gate 결과: {science[\"C\"][\"promotion_score_gates\"]}. 최종 runtime·누수·재현 승인은 별도다.'":
"f'M7 평균 TS-AUC={normal_m7[\"mean_TS_AUC\"]}, BL80={normal_m0[\"mean_TS_AUC\"]}. 사전고정 승격 점수 조건 통과={science[\"C\"][\"promotion_score_gates\"][\"score_gates_pass\"]}. 최종 runtime·누수·재현 승인은 별도다.'",
"f'EARLY→LATE 평균 M7−BL80 Δ={number(early[\"point\"])}. TIME/H의 지원 부족은 NA로 남기며 TIME_GENERALIZATION_NEXT9.csv, H_GENERALIZATION_NEXT9.csv와 점수 gate를 함께 본다.'":
"f'EARLY→LATE 평균 M7−BL80 Δ={number(early[\"point\"])}. 지원 충분 TIME/H 영역의 점수 기준 위반은 {len(stress_failures)}개다. {stress_text}. 지원 부족은 NA로 남겼다.'",
"'새 연구 역할은 B·같은 입력 비교·누수 감사가 모두 통과할 때만 부여한다. VARIABLE_CERTIFICATION_NEXT9.json과 FINAL_AUDIT_NEXT9.json을 함께 확인한다.'":
"('인증 B가 없으므로 새 movement-variable 연구 신호 역할은 없다. Primary BL80를 유지한다.' if not cert['certified_variable_count'] else '인증 B의 같은 입력 비교와 최종 누수 감사를 통과한 연구 신호 역할만 FINAL_AUDIT_NEXT9.json에서 부여한다.')"
}
for old,new in replacements.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
path='scripts/next9/write_next9_scientific_reports_v3.py';compile(source,path,'exec');atomic(ROOT/path,source)
queue=read_json(BASE/'runtime/TASK_QUEUE.json');output=[]
for item in queue:
    if item['id']=='WRITE_SCIENTIFIC_REPORTS_V2':
        assert not (BASE/'runtime/queue'/item['id']).exists()
        original=item;item={**item,'id':'WRITE_SCIENTIFIC_REPORTS_V3','script':path,'source_sha256':digest(ROOT/path)}
        atomic(BASE/'repairs/MEASURED_SCIENTIFIC_ANSWERS_V3_NEXT9.json',dict(status='PASS',timestamp=utc_now(),original=original,replacement=item,
            reason='Replace generic instructions in answers7,9,10,11,15,17 with actual counts, matched deltas and intervals; no scientific calculation or model selection changed.'))
    output.append(item)
atomic(BASE/'runtime/TASK_QUEUE.json',output,immutable=False)
