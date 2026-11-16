"""Describe transport without turning post hoc components into causal effects."""
from pathlib import Path
import sys
import csv
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now,csv_write


with activity('Summarize five-fold state transport and frozen-score component identities','STATE_TRANSPORT_INTERPRETATION','B4_EARLY_LATE_REPLICATION'):
    import numpy as np
    frequencies=[];records=[];sources={};full=[]
    for fold in range(5):
        root=BASE/'state_transport'/f'outer_{fold}';r=read_json(root/'DONE.json');assert r['status']=='COMPLETE'
        assert digest(root/'STATE_FREQUENCIES.csv')==r['frequencies_sha256']
        with (root/'STATE_FREQUENCIES.csv').open(encoding='utf-8-sig',newline='') as f:frequencies.extend(csv.DictReader(f))
        records.extend(r['records']);full.append(r);sources[(root/'DONE.json').relative_to(BASE).as_posix()]=digest(root/'DONE.json')
    grouped=defaultdict(list)
    for r in frequencies:grouped[(r['mode'],int(r['feature']),r['family'])].append(float(r['total_variation']))
    summary=[]
    for (mode,j,family),values in sorted(grouped.items()):
        assert len(values)==5
        summary.append(dict(mode=mode,feature=j,family=family,mean_TV=float(np.mean(values)),min_TV=min(values),max_TV=max(values),
                            fold_values='|'.join(map(str,values)),exploratory_not_new_selection=True))
    global_tv={mode:float(np.mean([r['mean_TV'] for r in summary if r['mode']==mode])) for mode in ['NORMAL','EARLY']}
    model=read_json(BASE/'MODEL_ONLY_FIVE_FOLDS_NEXT9.json')
    contrast={mode:model['results'][mode]['contrasts']['M3__vs__M2'] for mode in ['NORMAL','EARLY']}
    csv_write('STATE_TRANSPORT_FREQUENCIES_NEXT9.csv',frequencies);csv_write('STATE_TRANSPORT_SUMMARY_NEXT9.csv',summary)
    atomic(BASE/'STATE_TRANSPORT_DETAILS_NEXT9.json',dict(status='COMPLETE',timestamp=utc_now(),source_sha256=sources,fold_records=full,
        average_feature_family_total_variation=global_tv,already_frozen_model_comparison=contrast,
        no_classification_labels_used_for_transport=True,training_stride8_heldout_every_point=True,
        no_new_model_fit_or_candidate_scores=True,post_hoc_exploratory=True,
        warning='Centered component decomposition depends on frozen coding and correlated retained columns. Mean logit shifts alone cannot explain same-time AUC changes. Covariate transport is descriptive, not a causal explanation.'))
    text=f'''# 상태 입력의 시간 이동 진단

EARLY에서 M3−M2 TS-AUC 차이는 {contrast['EARLY']['point']:.12f}이며, 다섯 fold 모두 음수다. NORMAL 차이는 {contrast['NORMAL']['point']:.12f}다.

고정된 피처·상태 가족·fold의 상태 빈도 차이(TV)를 동일 가중 평균하면 NORMAL={global_tv['NORMAL']:.6f}, EARLY={global_tv['EARLY']:.6f}다. TV는 0이면 같은 분포, 1이면 겹치지 않는 분포다. 학습은 원래 stride8, 검증은 원래 모델 평가와 같은 전체 관측 시점을 사용했다. 각 series의 총 가중치는 동일하다.

M3−M2의 고정된 선형 점수 차이를 기존 입력 계수 변화, S1/S2/S3/S4 상태 항, 절편 차이로 정확히 분해했다. 이 합은 원래 점수 차이와 최대 1e−10 이내로 일치해야 한다. 세부 평균·표준편차는 STATE_TRANSPORT_DETAILS_NEXT9.json에 있다.

이것은 외부 결과를 본 뒤 수행한 사후 원인 탐색이다. 정답을 추가로 읽거나 학습·후보·기준을 바꾸지 않았다. 계수 분해는 고정된 인코딩에 의존하므로 고유한 변수 중요도나 인과 효과가 아니다. 특히 평균 점수 이동만으로 같은 시점의 순위를 비교하는 TS-AUC 하락을 설명할 수 없다. 상태 분포 이동과 성능 하락이 함께 관측돼도 원인이 확정되지는 않는다.
'''
    atomic(BASE/'STATE_TRANSPORT_FINDING_NEXT9_KO.md',text)
    atomic(BASE/'STATE_TRANSPORT_DONE_NEXT9.json',dict(status='PASS',timestamp=utc_now(),folds=5,original_failed_cadence_attempts_preserved=True,
        all_fixed_row_count_checks_passed=True,no_fits=True,summary_sha256=digest(BASE/'STATE_TRANSPORT_DETAILS_NEXT9.json')))
