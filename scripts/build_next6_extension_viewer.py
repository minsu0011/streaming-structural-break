from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/next6_extension'
def read(p):return json.loads((BASE/p).read_text(encoding='utf-8'))
def compact(curve):
    x=np.asarray(curve);a=x[:,[0,1,2,3,5,12,13,14,15,16,17,18,19]];a=np.round(a,6);changed=np.r_[True,np.any(a[1:]!=a[:-1],axis=1)];return np.column_stack((np.flatnonzero(changed),a[changed])).tolist()
def main():
    cases=[];names={'O1':'합성 · 단일 충격','OB16':'합성 · 16시점 연속 충격','D2':'합성 · 자기상관 부호 반전','V2':'합성 · 분산 2배','V4':'합성 · 분산 절반','RETURN_NORMAL':'합성 · 변화 후 정상 복귀'}
    for case,name in names.items():
        r=read(f'extended_synthetic/fresh/{case}_EXAMPLES.json')['curves'][0];a=r['treatment'];c={'name':name,'length':len(a),'tau':r['tau'],'event_time':r['scheduled'],'treatment':compact(a),'control':compact(r['control']),
            'support':'독립 합성 평가 partition31, 사전에 고정한 seed0. 전체 사례별 512개 경로에서 별도 검증했습니다.','baseline':None}
        hp=BASE/f'head_memory/{case}/EXAMPLES.json'
        if hp.exists():
            old=json.loads(hp.read_text(encoding='utf-8'))['cases'][0];c['baseline']={'treatment':np.round(old['treatment'],7).tolist(),'control':np.round(old['control'],7).tolist(),'threshold':read('calibration/fold_0/DONE.json')['thresholds']['M3']}
        cases.append(c)
    labels={'success':'빠른 확인','delayed':'늦은 확인','null_false_alarm':'오경보','miss':'누락'}
    for r in read('FRESH_RP_CASES.json')['cases']:
        if r['status']=='NO_CASE':continue
        cases.append({'name':'실제 RP · '+labels[r['category']],'length':len(r['curve']),'tau':r['tau'],'event_time':r['tau'],'treatment':compact(r['curve']),'control':None,'support':f"원래 outer fold0의 held-out series {r['dataset_id']}. 사전에 고정한 해시 순서로 선택했으며 tau는 사후 표시입니다.",'baseline':None})
    invalid=read('INVALID_REFERENCE_EXAMPLE.json')
    cases.append({'name':'수치 검증 · 기준 자료 부족','length':len(invalid['curve']),'tau':-1,'event_time':invalid['scheduled'],'treatment':compact(invalid['curve']),'control':None,'support':'상수 H의 입력 유효성 검증 사례입니다. 자료 부족일 때 증거와 확인은 0, 조건부 확률은 사용 불가로 반환합니다. 실제 RP 성적 사례가 아닙니다.','baseline':None,'referenceValid':False})
    with (BASE/'MACRO_STATE_TRANSITIONS.csv').open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
    macro=[{'phase':int(r['phase']),'a':int(r['previous']),'b':int(r['current']),'jump':float(r['jump_probability_unsmoothed'] or 0),'prob':float(r['ordinary_probability_smoothed']),'support':int(r['unique_training_series']),'sourceSupport':int(r['source_state_training_series'])} for r in rr if int(r['fold'])==0]
    with (BASE/'MACRO_TRANSITION_UNCERTAINTY.csv').open(encoding='utf-8-sig',newline='') as f:uncertainty={(int(r['phase']),int(r['previous']),int(r['current'])):r for r in csv.DictReader(f) if int(r['fold'])==0}
    for edge in macro:
        r=uncertainty[edge['phase'],edge['a'],edge['b']]
        edge['probCI']=[float(r[k]) if r[k] else None for k in ('bootstrap_lo','bootstrap_hi')];edge['jumpCI']=[float(r[k]) if r[k] else None for k in ('jump_bootstrap_lo','jump_bootstrap_hi')]
    data={'cases':cases,'macro':macro,'threshold':read('calibration/fresh/DONE.json')['thresholds']['FRESH'],'display_rounding_decimals':6,'source':'Actual frozen results; no raw observations'}
    template=(ROOT/'src/next6_extension/viewer/template.html').read_text(encoding='utf-8');page=template.replace('__PAYLOAD_JSON__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    out=BASE/'DYNAMIC_STATE_VIEWER.html';out.write_text(page,encoding='utf-8');print(out,out.stat().st_size,'bytes',len(cases),'cases')
if __name__=='__main__':main()
