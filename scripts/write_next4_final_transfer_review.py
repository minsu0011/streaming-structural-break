"""Write the evidence-based scientific decision after actual-artifact validation."""
import argparse
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,digest,utc_now,progress


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);parser.add_argument('--gate-tag',required=True);args=parser.parse_args()
    cid=args.candidate;assert cid=='D2_DISCRETE_SCALE_MIXTURE'
    sources={}
    def read(path):
        sources[path.relative_to(ROOT).as_posix()]=digest(path);return read_json(path)
    def rows(path):
        sources[path.relative_to(ROOT).as_posix()]=digest(path)
        with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
    def n(value):return format(float(value),'.12g')
    def ci(r):return '['+n(r['ci_2_5'])+', '+n(r['ci_97_5'])+']'
    gate=read(BASE/'gate_reviews'/args.gate_tag/'RESULTS.json')
    assert cid in gate['robustness_deployment_validation_order'] and not gate['performance_admitted']
    full=read(BASE/'experiments'/cid/'full/RESULTS.json')
    boot=read(BASE/'bootstrap'/cid/'RESULTS.json')['paired_all_time']
    forensic=read(BASE/'forensic'/cid/'RESULTS.json');transfer=read(BASE/'robustness'/cid/'RESULTS.json')
    scale=read(BASE/'robustness_forensic'/cid/'RESULTS.json')
    nomination=read(BASE/'finalist_nominations'/(cid+'.json'))
    fitted=read(BASE/'refits'/cid/'nominated_all_dev_v1/RESULTS.json')
    api=read(BASE/'finalist_validation'/cid/('API_'+cid)/'RESULTS.json')
    cold=read(BASE/'finalist_validation'/cid/'COLD/RESULTS.json')
    latency=read(BASE/'latency'/(cid+'_DIRECT')/'RESULTS.json')
    syn=read(BASE/'synthetic/candidates'/(cid+'_all_dev')/'RESULTS.json')
    memory=read(BASE/'memory_diagnostics'/(cid+'_all_dev')/'RESULTS.json')
    reflection=read(BASE/'sign_reflection/RESULTS.json')
    for r in (fitted,api,cold,latency,syn,memory,reflection):assert r['status']=='PASS'
    assert latency['direct_runtime_within_3x'] and syn['ARCH_decrease_each_delta_positive'] and syn['heavy_tail_decrease_each_delta_positive']
    assert api['prefix_cases']>=300
    for path in (BASE/'synthetic_nominations').glob('*.json'):
        nom=read(path);assert read(BASE/'synthetic/candidates'/nom['id']/'RESULTS.json')['status']=='PASS'
    text='# NEXT4 최종 과학적 채택 검토\n\n'
    text+='작성 UTC '+utc_now()+'. 연구 단계의 실제 파일 검증을 완료한 시점의 판단이다. **별도 final audit와 fresh raw 재현은 이후 실제 수행해야 하며, 이 문서는 감사 완료를 선행 주장하지 않는다.**\n\n'
    text+='주 성능 모델 BL2_C205_DEPENDENCE_80의 0.6232684128173095를 유지한다. 탐색 최고 B2의 원래 paired CI 하한 -0.000018936916538은 실패로 남긴다. '+cid+'는 제한된 SCALE 강건성 역할의 최종 감사 대상으로 지정한다. 새 performance champion이나 generation은 아니다.\n\n'
    text+='후보 평균 '+repr(full['mean'])+', 기준 대비 Δ '+repr(full['mean_delta'])+', 전체 CI '+ci(boot)+', P(Δ>0) '+n(boot['probability_delta_positive'])+'. 전체 CI가 0을 포함하므로 평균 개선을 확증하지 않았다.\n\n'
    text+='고정 우선순위는 original SCALE CI 하한, original 전체 gain, ID 순이다. 사후 가장 좋은 ALT나 합성 점수로 순서를 바꾸지 않았다. A2/C3/D3는 고정 네 분산 감소 반례를 모두 통과하지 못했다. B1은 이 네 반례를 통과했지만 original SCALE CI 하한이 더 낮고 H 제외 SCALE 점 추정치도 음수였다.\n\n'
    text+='| 검증 | 전체 Δ | 전체 95% CI | SCALE Δ | SCALE 95% CI |\n|---|---:|---|---:|---|\n'
    fs=forensic['paired_scale'];text+='| PRIMARY | '+n(boot['estimate_mean_delta'])+' | '+ci(boot)+' | '+n(fs['estimate_mean_delta'])+' | '+ci(fs)+' |\n'
    for t in transfer['summaries']:
        s=next(r for r in scale['summaries'] if r['split']==t['split'] and r['scope']=='SCALE')
        text+='| '+t['split']+' | '+n(t['mean_delta'])+' | '+ci(t['paired_all_time'])+' | '+n(s['estimate_mean_delta'])+' | '+ci(s)+' |\n'
    text+='\nSCALE에는 변화 시계열 132개와 null 4056개가 있다. ALT와 H 제외 학습은 같은 DEV 8000개의 다른 grouped partition이다. 독립 데이터가 아니다. H 전체의 matched 최소 Δ는 '+n(transfer['H_minimum_matched_delta'])+'다.\n\n'
    mechanisms=rows(BASE/'forensic'/cid/'PAIRED_FRACTURES.csv')
    text+='| 원래 기전 | Δ | 95% CI |\n|---|---:|---|\n'
    for r in mechanisms:
        if r['scope'] in ['mechanism_'+m for m in ('DEPENDENCE_DOMINANT','MIXED','WEAK_OR_AMBIGUOUS','observed_scale_increase')]:
            text+='| '+r['scope']+' | '+n(r['estimate_mean_delta'])+' | '+ci(r)+' |\n'
    text+='\nraw SD 증가/감소 subset은 53/79개여서 각 100개 미만이다. 이 방향 구분은 latent conditional variance의 정답이 아니다. MEAN 21, TAIL 86도 지원 표본 기준에 못 미친다.\n\n'
    text+='실제 all-DEV 파일의 고정 반례 확인은 512개 matched generating-seed pair를 사용했다. 각 시나리오와 seed, null 보정 방식, 기존 learner/혼합은 그대로이며 새 fit은 해당 후보의 전체 DEV 학습 1회뿐이다. 다음 Δ는 합성 TS-AUC로 실제 DEV 점수와 합산하지 않는다.\n\n'
    paired=rows(BASE/'synthetic/candidates'/(cid+'_all_dev')/'PAIRED_SYNTHETIC.csv')
    text+='| 실제 파일 반례 | Δ | 95% CI |\n|---|---:|---|\n'
    for r in paired:
        if r['scenario'] in ('V3','V4','CB_N3_070','CB_N3_050'):
            text+='| '+r['scenario']+' | '+n(r['estimate_mean_delta'])+' | '+ci(r)+' |\n'
    text+='\n| 실제 파일 이상치 후 지표 | 후보−BL80 | 95% CI |\n|---|---:|---|\n'
    for r in rows(BASE/'memory_diagnostics'/(cid+'_all_dev')/'PAIRED_OUTLIER_EXCESS.csv'):
        if r['metric'] in ('mean_signed_excess','mean_signed_excess_final_64'):
            text+='| '+r['scenario']+' '+r['metric']+' | '+n(r['estimate_mean_delta'])+' | '+ci(r)+' |\n'
    text+='\n음수 excess 차이는 해당 고정 반례에서 score 오염이 줄었다는 뜻이다. 검출력 보존이나 모든 이상치에 대한 보장을 뜻하지 않는다. fold-0의 256쌍과 실제 all-DEV 파일의 512쌍 통계를 혼합하지 않았다. 128개 TreeSHAP seed의 분해는 별도 진단이며 인과적 feature 제거 실험이 아니다.\n\n'
    text+='실제 파일의 반복 이상치 OB16은 평균 excess Δ +0.00476280282419, 마지막 64점 +0.00691903814368로 악화됐다. 두 CI도 양수다. fold-0에서 평균 Δ -0.00367609088426이었던 개선이 전체 학습 파일에서는 유지되지 않았다. 단발 이상치 개선과 반복 이상치 악화를 함께 보고하며, 전반적인 outlier repair라고 부르지 않는다. α=.01의 OB16 outlier alarm fraction도 0.650390625 대 BL80 0.525390625였고, α=.05에서는 0.986328125 대 0.939453125였다.\n\n'
    text+='고정 합성 TS-AUC에서도 모든 조건이 좋아진 것은 아니다. 실제 파일은 AR 변화 D1에서 Δ -0.00717549926562, mixed X1에서 -0.00466391236264, ARCH A1에서 -0.00176538745381, tail T1에서 -0.00131586908997이었다. 강건성 역할의 근거는 미리 고정한 실제 SCALE 기준과 감소 반례 확인이며, 새로운 사후 gate를 만들거나 이 손실을 숨기지 않는다.\n\n'
    text+='| 16384점 null | α | 후보 alarm fraction | BL80 alarm fraction |\n|---|---:|---:|---:|\n'
    for r in rows(BASE/'synthetic/candidates'/(cid+'_all_dev')/'LONG_NULL.csv'):
        if int(r['horizon'])==16384:text+='| '+r['regime']+' | '+r['alpha']+' | '+r['eventual_alarm_rate']+' | '+r['reference_eventual_alarm_rate']+' |\n'
    text+='\n독립 1024점 null maximum으로 정한 threshold를 긴 horizon에 고정했으므로 anytime 보장은 없다. 부호 반전 결과도 기존 DEV stress이며 새로운 ensemble/augmentation 후보가 아니다.\n\n'
    text+='실제 entry는 300개 prefix 및 DEV 전체 현재값 추론 대조, 20개 역순 reset, 첫 None, 미래 길이/미래 suffix trap을 통과했다. cold read-only import는 2개 새 Python 프로세스에서 확인했다. 고유 mutable NumPy state 배열은 '+str(api['state_array_bytes_min'])+' bytes다. 연구 단계 2회 paired 직접 runtime 중앙 비율은 '+repr(latency['summary_median_ratios'])+'이며 동일 시점 BL80과 비교했다. Final audit에서 직접 측정과 raw→새 feature→동일 fit→추론 재현을 별도로 요구한다.\n\n'
    text+='SEAL 네 개는 미개봉이며 NEXT4 reduced/public/cloud 사용은 0이다. DEV saturation은 HIGH로 유지한다. 이 채택은 SCALE의 제한된 support와 고정 진단 범위에 한정한다.\n\n근거 SHA-256:\n\n'
    text+='\n'.join('- `'+name+'`: `'+sha+'`' for name,sha in sources.items())+'\n'
    path=ROOT/'docs/NEXT4_FINAL_TRANSFER_REVIEW.md';assert not path.exists();path.write_text(text,encoding='utf-8')
    progress('Final transfer decision reviewed',cid+' limited SCALE robustness; primary unchanged; actual artifact verified and separate audit still required')
    print('NEXT4 FINAL SCIENTIFIC TRANSFER REVIEW WRITTEN',cid,flush=True)


if __name__=='__main__':main()
