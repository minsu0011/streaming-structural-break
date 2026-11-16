"""Summarize completed diagnostic evidence without selecting or fitting models."""
from pathlib import Path
import json
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now
from scripts.write_next_final_reports import md_table


def main():
    out = ROOT/'artifacts/next/research_synthesis'
    if out.exists():
        raise RuntimeError('Preserve the first completed secondary-evidence synthesis')
    sources = {}
    def study(relative):
        directory = ROOT/relative
        report = json.loads((directory/'RESULTS.json').read_text(encoding='utf-8'))
        if report['status'] != 'PASS':
            raise RuntimeError('Incomplete diagnostic: '+relative)
        sources[(directory/'RESULTS.json').relative_to(ROOT).as_posix()] = sha256(directory/'RESULTS.json')
        for name, digest in report['files_sha256'].items():
            if sha256(directory/name) != digest:
                raise RuntimeError('Diagnostic payload changed: '+name)
            sources[(directory/name).relative_to(ROOT).as_posix()] = digest
        return directory, report
    long, long_report = study('artifacts/next/long_null_stress')
    directions, direction_report = study('artifacts/next/synthetic_variance_stress/direction_fractures')
    late, late_report = study('artifacts/next/late_contribution_contrasts')
    fixed, fixed_report = study('artifacts/next/fixed_ablation_contrasts/research_v1')
    if long_report['long_scalar_predictions'] != 327680000 or direction_report['paired_scale_contrasts'] != 20 or late_report['contrasts'] != 9 or fixed_report['contrasts'] != 16:
        raise RuntimeError('A planned diagnostic comparison is missing')
    horizon = pd.read_csv(long/'HORIZON_FALSE_ALARMS.csv')
    if len(horizon) != 200 or not (horizon.series == 1000).all():
        raise RuntimeError('Long-null cell coverage differs')
    keys = ['model', 'regime', 'nominal_alpha_at_1024']
    start = horizon[horizon.horizon == 1024].set_index(keys)
    end = horizon[horizon.horizon == 16384].set_index(keys)
    paired_horizons = start[['false_alarms', 'false_alarm_fraction']].join(
        end[['false_alarms', 'false_alarm_fraction', 'wilson_95_low', 'wilson_95_high']], lsuffix='_at_1024', rsuffix='_at_16384').reset_index()
    paired_horizons['additional_false_alarm_fraction'] = paired_horizons.false_alarm_fraction_at_16384-paired_horizons.false_alarm_fraction_at_1024
    if len(paired_horizons) != 40 or (paired_horizons.additional_false_alarm_fraction < 0).any():
        raise RuntimeError('A running maximum alarm rate decreased with horizon')
    ranges = []
    for (model, alpha), frame in paired_horizons.groupby(['model', 'nominal_alpha_at_1024'], sort=False):
        ranges.append({'model': model, 'alpha_at_1024': alpha,
            'min_regime_fpr_at_1024': frame.false_alarm_fraction_at_1024.min(),
            'max_regime_fpr_at_1024': frame.false_alarm_fraction_at_1024.max(),
            'min_regime_fpr_at_16384': frame.false_alarm_fraction_at_16384.min(),
            'max_regime_fpr_at_16384': frame.false_alarm_fraction_at_16384.max(),
            'minimum_horizon_increase': frame.additional_false_alarm_fraction.min(),
            'maximum_horizon_increase': frame.additional_false_alarm_fraction.max()})
    ranges = pd.DataFrame(ranges)
    scale = pd.read_csv(directions/'SCALE_PAIRED_BOOTSTRAP.csv')
    contributions = pd.read_csv(late/'PAIRED_CONTRIBUTIONS.csv')
    interventions = pd.read_csv(fixed/'PAIRED_INTERVENTIONS.csv')
    out.mkdir(parents=True)
    paired_horizons.to_csv(out/'HORIZON_1024_VS_16384.csv', index=False)
    ranges.to_csv(out/'HORIZON_REGIME_RANGES.csv', index=False)
    summary_columns = ['candidate', 'reference', 'estimate_mean_delta', 'ci_2_5', 'ci_97_5', 'probability_delta_positive']
    text = '# NEXT 보조 연구: 얻은 근거와 남은 한계\n\n'
    text += ('아래 수치는 최종 역할을 새로 정하는 기준이 아니다. 고정된 DEV 또는 합성 시계열에서 수행한 보조 비교이며, '
        '모든 bootstrap 구간은 해당 모델·예측·선택·유한 calibration에 조건부이다. 다중 탐색을 보정하지 않았다.\n\n')
    text += ('## 관측 기간과 누적 오경보\n\n1,024개 관측값으로 정한 모델별 임계값을 고정하고, '
        '새 무변화 시계열 5,000개를 최대 16,384개까지 관측했다. 4개 CV fold-0 모델에서 '
        '327,680,000개 scalar 예측을 계산했다. 아래 min/max는 다섯 합성 regime 사이의 범위이고 신뢰구간이 아니다.\n\n')
    text += md_table(ranges.round(6))+'\n\n'
    text += ('PRIMARY/V007/Q005/Y001만 이 실험의 대상이다. 최종 J102/B104/50-tree/all-DEV 모델에서 '
        '같은 비율을 직접 측정했다고 해석할 수 없다. 범위의 변화는 짧은 horizon용 임계값을 장기간 '
        '그대로 쓸 때의 민감도이며 실제 경쟁 데이터의 오경보 보장은 아니다. 각 regime의 사건 수와 '
        'Wilson 구간은 HORIZON_1024_VS_16384.csv, 전체 horizon과 paired 모델 비교는 원본 long_null_stress에 있다.\n\n')
    text += '## 증가와 감소 방향을 모두 남긴 scale 반례\n\n'+md_table(scale[['regime', 'polarity', *summary_columns]].round(7))+'\n\n'
    text += ('이 비교는 합성 aggregate 결과에서 약점이 보인 뒤 수행한 사후 진단이다. '
        '모든 다섯 regime에서 scale 증가·감소를 모두 보고하며, 결과를 보고 모델이나 임계값을 수정하지 않았다. '
        'MEAN/SCALE/MIXED 방향별 90개 점 추정치는 direction_fractures/DIRECTION_SCORES.csv에 있다.\n\n')
    text += '## 고정 앙상블의 추가 기여\n\n'+md_table(contributions[summary_columns].round(9))+'\n\n'
    text += ('L101/L102/L103은 각각 V007:Q005 = 0.50:0.50, 0.67:0.33, 0.33:0.67이다. '
        '미리 고정한 세 가중치를 그대로 사용했다. 점 추정치가 구성 모델보다 높다는 것과 '
        '그 차이의 불확실성이 충분히 작다는 것은 별개의 결과이다. Y001 대비 평균은 세 경우 모두 낮았다.\n\n')
    text += '## 고정된 16개 단일 변경 비교\n\n'+md_table(interventions[summary_columns].round(9))+'\n\n'
    text += ('J102−Y001의 평균 차이는 약 2.8e-7이며 자기상관 보정의 독립적인 개선 증거가 아니다. '
        'B104는 explicit GLR와 GLR-memory 열을 모두 제거했고 local/Bayes 열은 남겼다. '
        '열 제거와 runtime 계산 제거를 혼동하지 않는다. 학습량·nested 메뉴·nested alarm의 결과도 '
        '별도 보고서에 보존했다. 이 문서의 표로 새 후보, 새 threshold 또는 seal 개봉을 정하지 않는다.\n')
    (out/'ADDITIONAL_DIAGNOSTICS.md').write_text(text, encoding='utf-8')
    paths = [out/'HORIZON_1024_VS_16384.csv', out/'HORIZON_REGIME_RANGES.csv', out/'ADDITIONAL_DIAGNOSTICS.md']
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'source_files_sha256': sources, 'script_sha256': sha256(Path(__file__)),
        'files_sha256': {path.name: sha256(path) for path in paths},
        'new_fits': 0, 'new_predictions': 0, 'new_bootstrap_draws': 0, 'new_selection_rules': 0,
        'seal_rows': 0, 'reduced_new_uses': 0})
    print('SECONDARY EVIDENCE SYNTHESIS PASS', flush=True)


if __name__ == '__main__':
    main()
