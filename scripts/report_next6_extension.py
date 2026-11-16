"""Build reviewable reports from completed receipts; pending work stays explicit."""
from pathlib import Path
import sys,csv,html,datetime,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.next6_extension.common import BASE,read_json,utc_now
from summarize_next6_extension import rows

def table(headers,items):return '\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in items)+'\n\n'
def pct(v):return f'{float(v):.1%}'
def main():
    baseline_complete=all((BASE/f'extended_synthetic/fold_{f}/DONE.json').exists() for f in range(5));bootstrap_complete=all((BASE/f'graph_bootstrap/fold_{f}/DONE.json').exists() for f in range(5));final=(BASE/'FINAL_AUDIT_NEXT6_EXTENSION.json').exists()
    status='연구·실행 최종 감사 완료' if final else '후속 연구 진행 중';stamp=utc_now();fresh=rows(BASE/'FRESH_SYNTHETIC_SUMMARY.csv');d=read_json(BASE/'FROZEN_EVIDENCE_DECOMPOSITION.json');rpdiag=read_json(BASE/'FRESH_RP_EVALUATION.json');runtime=read_json(BASE/'FUSED_OBSERVABLE_RUNTIME_PARITY.json');finite=read_json(BASE/'FINITE_INFLUENCE_AUDIT_DONE.json');nboot=sum(len(list((BASE/f'graph_bootstrap/fold_{f}').glob('replicate_*.json'))) for f in range(5));ncohort=sum(len(list((BASE/f'extended_synthetic/fold_{f}').glob('*_DONE.json'))) for f in range(5))
    time_text='요청한 추가 7시간 연구를 계속 진행하고 있습니다. 최소 종료 시점은 2026-09-13 01:20:09 UTC(한국 시각 10:20:09)입니다.'
    if final:
        a=read_json(BASE/'FINAL_AUDIT_NEXT6_EXTENSION.json');time_text=f"추가 작업 구간: {a['started_utc']} ~ {a['completed_utc']}. 실제 경과 {a['elapsed_hours']:.2f}시간. 자세한 작업·동시 실행 시간은 TIME_ACCOUNTING_NEXT6_EXTENSION.json을 확인합니다."
    intro=f'''# NEXT6 동적 상태전이망 후속 연구

**상태: {status} · 갱신 {stamp}**

충격과 새 관측의 지속 증거를 구분하는 상태망, 실제 자료의 전이확률 그림, 시간 재생 HTML을 만들었습니다. 기존 BL80/D2는 유지합니다. 새 설명 상태망을 기존 분류기의 대체 모델로 승격하지 않았습니다.

{time_text}

## 먼저 볼 파일

- [시간을 움직이는 동적 상태망](DYNAMIC_STATE_VIEWER.html): 11개 실제 계산 사례, 8개 노드, 두 조건의 전이확률, 학습 지원 수와 95% 범위.
- [상태전이망 이미지](FIGURES/02_OBSERVED_DYNAMIC_STATE_TRANSITION_NETWORK.png): 7개 실행 상태 중 학습 자료에서 관측된 전이.
- [그림 전체 보기](FIGURES/index.html): 확대 가능한 PNG/SVG.
- [상세 연구 결과](RESEARCH_REPORT.md), [계산 방법과 해석 범위](METHODS_AND_INTERPRETATION.md).

## 판단

1. **원래 NEXT6의 승격 판정은 유지합니다.** M3의 평균 TS-AUC는 0.620834, BL80(M0)는 0.620365였습니다. 차이 +0.000469는 사전 최소 +0.0012에 못 미쳤고 fold별 결과도 일관되지 않았습니다. TS-AUC는 승률이 아닙니다.
2. **예측 가능한 파생 연결과 변화 판별에 유용한 연결을 분리했습니다.** 원래 전체 전이 점수와 자기 상태 점수의 TS-AUC 차이는 {d['full_minus_own_TS_AUC']:+.6f}였습니다. 크기 변화를 통제한 새 교차 관계 검사에서는 원래 5개 fold 모두 채택 연결이 0개였습니다.
3. **한 번의 충격이 오래 남는 병목을 찾았습니다.** 기존 M3의 잔여 효과는 주로 J102의 누적 최대값에서 나왔습니다. 새 상태망은 서로 다른 네 완성 구간의 증거로 지속을 확인합니다.
4. **유한 영향과 실제 실행을 검증했습니다.** {finite['cases']:,}개 오염 경계 사례에서 27개 출력과 가변 상태가 오염 종료 후 +128 이내에 대응 경로와 정확히 합쳐졌습니다. RP 6,400개 전체의 빠른 실행 결과도 설명용 구현과 일치했습니다.
5. **아직 약점이 있습니다.** 새 지속 증거 단독의 RP 사건 판정 TS-AUC는 {rpdiag['mean_diagnostic_TS_AUC']:.4f}였습니다. 분산 감소를 잘 놓치고, H 길이와 관측 기간에 따라 오경보율이 달라집니다. 7개 상태 화살표는 관측 상황을 합친 평균 빈도이며 완전한 Markov 생성 모델이나 경제적 인과망이 아닙니다.

## 완료와 남은 작업

{table(['항목','현재 상태'],[['원래 그래프 학습 재표집',f'{nboot}/1,000개'],['장시간 M0/M3 대조',f'{ncohort}/115개 fold×시나리오'],['수정된 조건부 관계 검출력','42조건 × 40회 완료'],['같은 원시 엔진의 정상 대조','4배경 × 40개 그래프 완료'],['H 길이·H 오염 이동 실험','완료'],['전체 RP 실행·유효성·확률표 감사','완료'],['시차 경계·분산 감소의 계산 한계','완료'],['최종 무결성·지연시간', '완료' if final else '전체 장시간 실험 후 최종 확인']])}

전체 ZIP의 추출·실행 검사와 바탕화면 20개 사본의 결과는 별도 `HANDOFF_AUDIT_NEXT6_EXTENSION.json`에 기록합니다. 연구 감사와 패키지 검사를 구분합니다.

Shadow, Seal, public/cloud/reduced test를 사용하지 않았습니다. 후속 연구는 원래 outer 결과를 이미 본 뒤의 개발 분석이며 독립적인 최종 검증 성적이 아닙니다.
'''
    (BASE/'00_READ_FIRST_NEXT6_EXTENSION_KO.md').write_text(intro,encoding='utf-8')
    lines=[f'# NEXT6 후속 병목 연구 결과\n\n상태: **{status}** · {stamp}\n', '## 관측 상태의 정상 경보와 지속 변화 확인\n\n경보 비교에 쓰는 점수 임계값은 별도 calibration에서 고정했습니다. 노드 범주·충격 표시·구간 수는 미리 정한 상태 규칙입니다. 아래는 새 상태망의 512개 독립 합성 seed 결과입니다. 변화가 있는 사례의 값은 사전 경보 없이 처음으로 변화 이후에 확인한 비율이며, 정상·충격 사례는 경로 중 한 번 이상 경보한 비율입니다.\n']
    menu=[('N0','Gaussian 정상'),('N1','AR 정상'),('N2','ARCH 정상'),('N3','AR/ARCH/꼬리 정상'),('O1','단일 오염'),('OB16','16시점 오염'),('V2','분산 2배'),('V4','분산 절반'),('D1','자기상관 증가'),('D2','자기상관 부호 반전'),('RETURN_NORMAL','변화 후 정상 복귀')];items=[]
    for case,label in menu:
        rr=[next(r for r in fresh if r['scenario']==case and int(r['horizon'])==h) for h in (1024,4096)];metric='observed_null_alarm_rate' if case in ('N0','N1','N2','N3','O1','OB16') else 'first_detection_rate';items.append([label,pct(rr[0][metric]),pct(rr[1][metric]),'낮을수록 좋음' if metric.startswith('observed_null') else '높을수록 좋음'])
    lines.append(table(['시나리오','1,024시점','4,096시점','방향'],items));lines.append('\n95% Wilson 구간, 지연, 미검출·회복 검열은 [FRESH_SYNTHETIC_SUMMARY.csv](FRESH_SYNTHETIC_SUMMARY.csv)에 있습니다. 정상 복귀 뒤에도 공식 사건 정답은 1입니다. 긴 관측에서 확인율이 높아지는 효과와 오경보 증가를 함께 봅니다.\n')
    if (BASE/'FRESH_VS_FROZEN_PAIRED_SYNTHETIC.csv').exists():
        paired=rows(BASE/'FRESH_VS_FROZEN_PAIRED_SYNTHETIC.csv');items=[]
        def m3_delta(case,metric):
            return float(next(r for r in paired if r['scenario']==case and int(r['horizon'])==1024 and r['reference']=='M3' and r['metric']==metric)['FRESH_minus_reference_mean'])*100
        lines.append(f"\n전체 5개 고정 모델 비교에서 1,024시점의 새 상태망 경보율은 기존 M3보다 단일 충격 {-m3_delta('O1','alarm'):.1f}%p, 16시점 충격 {-m3_delta('OB16','alarm'):.1f}%p 낮았습니다. 자기상관 증가의 첫 확인율 차이는 {m3_delta('D1','first_detection'):+.1f}%p였지만, 분산 1.5배는 {m3_delta('V1','first_detection'):+.1f}%p였습니다. 충격 억제와 일부 의존성 변화에서 이득을 확인했으며, 분산 변화 전반의 우위를 확보하지 못했습니다. 이는 고정 합성 경로의 기전 비교이고 RP TS-AUC 개선이나 독립 검증을 뜻하지 않습니다. M0 비교와 조건별 구간도 아래에 모두 표시합니다.\n")
        for case,label in menu:
            metric='alarm' if case in ('N0','N1','N2','N3','O1','OB16') else 'first_detection'
            for ref in ('M0','M3'):
                r=next(r for r in paired if r['scenario']==case and int(r['horizon'])==1024 and r['reference']==ref and r['metric']==metric);items.append([label,ref,f"{float(r['FRESH_minus_reference_mean']):+.3f}",f"[{float(r['ci_lo']):+.3f}, {float(r['ci_hi']):+.3f}]"])
        lines.extend(['\n## 기존 동결 모델과 동일 seed 비교\n\n5개 fold의 예측을 seed 단위로 함께 재표집했습니다. 2,560개의 fold×seed를 독립 표본으로 세지 않았습니다. 다음 값은 1,024시점의 `새 상태망 − 기준` 차이입니다. 정상/오염의 경보 차이는 음수가 유리하고, 지속 변화 확인 차이는 양수가 유리합니다.\n',table(['조건','기준','비율 차이','95% paired 구간'],items)])
    else:lines.append('\n## 기존 모델과의 장시간 비교\n\n5개 fold 전체가 끝난 뒤 동일 seed paired 비교를 합칩니다. 현재 일부 fold 결과로 전체 성능 우위를 선언하지 않습니다.\n')
    lines.append(f"\n## 새 관측 상태와 과거 사건 판정의 차이\n\n새 상태망의 지속 증거 한 열을 원래 RP 6,400개의 공식 사건 정답으로 평가한 진단 TS-AUC는 **{rpdiag['mean_diagnostic_TS_AUC']:.6f}**, 고정 예측에 조건부인 95% 구간은 [{rpdiag['conditional_CI'][0]:.6f}, {rpdiag['conditional_CI'][1]:.6f}]였습니다. 별도 탐지 head를 학습하거나 승격 후보로 선택한 결과가 아닙니다. 합성 충격 억제와 현재 상태 설명에서 확인한 이득을 과거 사건 분류의 전반적 개선으로 해석하지 않습니다. [RP 진단 결과](FRESH_RP_EVALUATION.json)와 [fold별 표](FRESH_RP_DIAGNOSTIC_AUC.csv)를 함께 확인합니다.\n")
    original=rows(BASE.parent/'next6/MODEL_COMPARISON_NEXT6.csv');strict=rows(BASE.parent/'next6/STRICT_INPUT_PAIRED_NEXT6.csv');standard=rows(BASE.parent/'next6/PAIRED_UNCERTAINTY_NEXT6.csv')
    lines.append('\n## 원래 사건 탐지 모델의 승격 판정\n\n아래는 이번 후속 합성 비교와 별개인 원래 grouped 5-fold TS-AUC 결과입니다. 상태·시간·시차 입력을 빠짐없이 포함한 주 비교군은 M1_STRICT입니다. 원래 M1에는 명시적 상태·현재 시점이 빠졌고, 이 문제는 outer 점수 공개 전에 M1_STRICT로 수정·동결했습니다.\n')
    lines.append(table(['원래 모델','평균 TS-AUC','BL80 대비 평균 차이'],[[name,f"{float(next(r for r in original if r['method']==name)['mean_TS_AUC']):.6f}",f"{float(next(r for r in original if r['method']==name)['delta_M0']):+.6f}"] for name in ('M0','M1_STRICT','M2','M3','OWN','RANDOM')]))
    comparisons=[next(r for r in standard if r['candidate']=='M3' and r['reference']==ref) for ref in ('M0','OWN','RANDOM')]+[next(r for r in strict if r['candidate']=='M3')]
    lines.append(table(['원래 M3 비교 대상','평균 차이','95% whole-series paired 구간'],[[r['reference'],f"{float(r['estimate_mean_delta']):+.6f}",f"[{float(r['ci_2_5']):+.6f}, {float(r['ci_97_5']):+.6f}]"] for r in comparisons]))
    lines.append('\nM3는 random-edge 대조보다 나았지만 BL80·동일 입력 M1_STRICT·own-history 대비 구간의 하한이 양수가 아니었습니다. BL80 대비 평균 차이도 사전 +0.0012 기준에 못 미쳤습니다. 따라서 새로운 그래프의 성능 증분을 확증하거나 기존 champion을 교체하지 않습니다. M4는 일부 fold만 허용된 후보이므로 이 전체 5-fold 표에 포함하지 않았습니다. [원래 모델 표](../next6/MODEL_COMPARISON_NEXT6.csv), [기존 paired 결과](../next6/PAIRED_UNCERTAINTY_NEXT6.csv), [동일 입력 paired 결과](../next6/STRICT_INPUT_PAIRED_NEXT6.csv)를 그대로 보존했습니다. 구간은 고정 예측에 조건부이며 과거 DEV 적응성을 보정하지 않습니다.\n')
    lines.extend(['\n## 병목과 수정\n',table(['문제','확인한 근거','처리'],[
        ['파생 연결의 예측성과 사건 판별력 혼동','원래 전체 전이 − 자기 상태 TS-AUC = −0.007298','관계 추가 정보와 자기 상태 정보를 분해'],
        ['충격의 오래된 기억','J102 누적 최대값이 장기 점수 효과의 큰 부분','4개 새 구간 확인과 유한 영향 상태'],
        ['희소 문맥에서 null 성질 붕괴','128개 희소 표 중 로그 오즈 phase 차이 최대3.08725','자기 상태 층 전체를 두 phase에서 함께 backoff'],
        ['유한 극단 H의 축약 연산 overflow','상수 float32 최대값에서 기준 유효성 오판','float64 중앙값/표준편차 축약 어댑터'],
        ['자료 부족인데 확인 증거 출력','자료 부족 상태와 증거12·확인4가 공존','자료 부족이면 증거·확인·확률 사용을 무효화'],
        ['시차 입력의 phase 경계 혼합','45,044개 post 구간 쌍 중 671개가 경계 직전 최대4시점 참조','해당 구간 제외 민감도 검사: 5개 fold 연결·확률표 동일'],
        ['감소 방향의 표현 범위와 확인 손실','H의91.52%에서 꼬리 음의 확인 불가; V4 크기 한 구간77.73%→네 구간3.52%','고정 계산의 한계를 정량화; 임계값·모델은 재선택하지 않음'],
        ['스칼라 결과 조립 비용','설명용27열 어댑터 예비35μs/점','동일 계산을 합친 실행 경로; 예비3.55μs/점'],
        ['상태명만으로 Markov성을 가정','회복 체류 시간·구간 경계에 따라 전이 비중 변화','평균 관측 전이로 설명하고 시간·기억 제한 공개']])])
    if (BASE/'LATENCY_NEXT6_EXTENSION.csv').exists():
        latency=rows(BASE/'LATENCY_NEXT6_EXTENSION.csv');lines.append('\n## 최종 실제 스칼라 실행 비용\n\n대규모 연구 작업을 끝낸 뒤 동일 H와 4,096개 관측을 7회 순서 교대로 처리했습니다. 값은 실제 `predict_one` 호출의 마이크로초이며, 서로 다른 역할의 모델을 같은 성능으로 간주하지 않습니다.\n');lines.append(table(['실행기','중앙값 μs','p95 μs'],[[r['method'],f"{float(r['scalar_median_us']):.3f}",f"{float(r['scalar_p95_us']):.3f}"] for r in latency]));lines.append('\n첫 모델 로딩·JIT·H 준비 비용은 COLD_START_NEXT6_EXTENSION.csv에 따로 기록했습니다. 위 병목 표의 예비 지연시간 대신 이 최종 측정을 사용합니다.\n')
    else:lines.append('\n예비 지연시간은 다른 연구 작업이 실행 중인 측정입니다. 최종 비교는 별도 LATENCY 결과가 생성되면 그것을 사용합니다.\n')
    lines.append('\n원래 과학적 소스와 이전 결과를 보존하고 수정 어댑터를 별도 버전으로 남겼습니다. 원시 정상 대조 160개에서는 채택 연결이 0개였지만, 별도 범주형 대조의 64-series 희소 무관계 조건은 3/40(7.5%)에서 오연결이 나왔습니다. 두 실험을 합쳐 오탐이 없다고 표현하지 않습니다.\n')
    if (BASE/'BOOTSTRAP_EDGE_SELECTION_FREQUENCY.csv').exists():
        boot=rows(BASE/'BOOTSTRAP_EDGE_SELECTION_FREQUENCY.csv');selected=[r for r in boot if r['original_selected']=='True'];freq=[float(r['selection_frequency']) for r in selected];lines.append(f'\n## 원래 그래프의 재표집 안정성\n\n원래 채택 edge×fold {len(selected)}개에 대한 선택 빈도 범위는 {min(freq):.1%}~{max(freq):.1%}입니다. 전 연결의 표는 [BOOTSTRAP_EDGE_SELECTION_FREQUENCY.csv](BOOTSTRAP_EDGE_SELECTION_FREQUENCY.csv)를 확인합니다. 선택 빈도는 연결이 참일 확률이나 FDR 보장이 아닙니다.\n')
        if (BASE/'BOOTSTRAP_INTERPRETATION_SUMMARY.json').exists():
            detail=read_json(BASE/'BOOTSTRAP_INTERPRETATION_SUMMARY.json')
            lines.append(f"\n1,000개 그래프에서 채택된 연결의 발생 횟수 {detail['total_selected_link_occurrences']:,}개 중 {detail['mechanical_selected_link_occurrences']:,}개({detail['mechanical_share_of_link_occurrences']:.1%})가 원래 계산 계보 감사의 기계적 파생 연결이었습니다. 원래 45개 edge×fold는 **다시 선택된 경우에 한해** 모두 원래 계수 부호를 유지했습니다. 이는 부호의 조건부 일치이며 정확한 시차의 안정성이나 변화 판별력은 아닙니다.\n")
            labels={'energy':'에너지','conditional_energy':'조건부 에너지','tail':'꼬리','variance':'조건부 분산'}
            lines.append(table(['원래 특징 계열의 방향','어느 시차든 연결 1개 이상 선택: fold별 범위'],[[labels[r['source_family']]+' → '+labels[r['target_family']],f"{r['minimum_fold_frequency']:.1%}~{r['maximum_fold_frequency']:.1%}"] for r in detail['family_pairs']]))
            lines.append('\n계열 단위 연결은 자주 남아도 특정 노드·시차의 선택은 불안정할 수 있습니다. 특히 에너지/조건부 에너지 상방 → 조건부 분산 예측의 lag1 두 연결은 99.5~100%로 가장 안정적이었지만 계산 구조에서 비롯됐습니다. 기계적 표시가 없는 나머지 연결도 독립 원인으로 인증한 것은 아닙니다. [부호 표](BOOTSTRAP_SELECTED_COEFFICIENT_SIGNS.csv), [계열별 집계](BOOTSTRAP_FAMILY_SELECTION_FREQUENCY.csv), [해석 집계와 입력 해시](BOOTSTRAP_INTERPRETATION_SUMMARY.json)를 보존했습니다.\n')
    else:lines.append('\n## 원래 그래프의 재표집 안정성\n\n학습 그룹을 다시 뽑아 동일 선택 절차를 1,000회 수행 중입니다. 원래 내부 해시 절반, 후보, 기준은 바꾸지 않았습니다. 완료 뒤 선택 빈도·부호·기계적 특징 계열을 함께 보고합니다.\n')
    lines.extend(['\n## 재현과 범위\n\n[계산 방법](METHODS_AND_INTERPRETATION.md), 실행 소스와 감사 receipt를 함께 보존했습니다. 원래 6,400개 RP와 grouped 5-fold를 유지하며 새 분할과 새로운 학습 recipe를 만들지 않았습니다. 모든 추가 분석은 적응적 개발 연구입니다. 최종 ZIP은 원자료·전체 tau표·특징 캐시·가상환경을 제외합니다.\n',f'\n{time_text}\n'])
    (BASE/'RESEARCH_REPORT.md').write_text(''.join(lines),encoding='utf-8')
    cards=''.join(f'<article><h2>{html.escape(p.stem)}</h2><a href="{p.name}"><img loading="lazy" src="{p.name}" alt="{html.escape(p.stem)}"></a><a href="{p.with_suffix(".svg").name}">확대용 SVG</a></article>' for p in sorted((BASE/'FIGURES').glob('*.png')))
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NEXT6 후속 상태망 그림</title><style>body{max-width:1450px;margin:auto;padding:25px;background:#f2f6f8;color:#273d4b;font:16px/1.7 "Malgun Gothic",sans-serif}article{padding:22px;background:white;border-radius:14px;margin:24px 0}h2{font-size:17px;overflow-wrap:anywhere}img{max-width:100%;height:auto}a{color:#28758a}</style><h1>실제 계산 결과로 만든 동적 상태망과 병목 분석</h1><p><a href="../DYNAMIC_STATE_VIEWER.html">시간 재생 상태망</a> · <a href="../00_READ_FIRST_NEXT6_EXTENSION_KO.md">먼저 읽기</a> · <a href="../METHODS_AND_INTERPRETATION.md">계산과 해석</a></p><p>관측 전이 빈도와 검증 결과입니다. 경제적 인과망이나 보정된 변화 발생확률이 아닙니다. 기존 BL80/D2를 유지합니다.</p>'+cards+'</html>'
    (BASE/'FIGURES/index.html').write_text(page,encoding='utf-8');print(status,nboot,ncohort)
if __name__=='__main__':main()
