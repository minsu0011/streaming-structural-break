"""Draw completed first-event and bounded-scale diagnostics from exact tables."""
from pathlib import Path
import argparse
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.next7.common import BASE, read_json, save, digest, utc_now, activity

CATEGORIES = ('correct_first_qualification', 'prebreak_qualified_memory',
              'postbreak_raw_confirmation_rejected', 'no_postbreak_raw_confirmation')
LABELS = ('올바른 최초 확인', '변화 전 오기억', '원시 확인 뒤 모두 거절', '변화 뒤 원시 확인 없음')
COLORS = ('#16846b', '#ba4646', '#d89530', '#748499')


def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main(snapshot):
    assert snapshot and all(c.isalnum() or c in '_-' for c in snapshot)
    relative = 'visuals/scale_evidence/' + snapshot
    destination = BASE / relative
    assert not destination.exists()
    inputs = {}
    for name in ('SCALE_EVIDENCE_PATH_AUDIT_DONE_NEXT7.json', 'OBSERVED_SCALE_ADMISSION_DONE_NEXT7.json'):
        receipt = read_json(BASE / name)
        assert receipt['status'] == 'COMPLETE'
        inputs[name] = digest(BASE / name)
        for path, sha in receipt['outputs_sha256'].items():
            assert digest(BASE / path) == sha
            inputs[path] = sha
    first = rows(BASE / 'SCALE_FIRST_EVENT_FAILURE_PARTITION_NEXT7.csv')
    rp = [r for r in rows(BASE / 'RP_SCALE_ADMISSION_PARTITION_NEXT7.csv') if r['group'] == 'ALL_RP']
    lookup = {(r['case'], int(r['horizon']), r['category']): r for r in first}
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
        'font.size': 12, 'figure.facecolor': '#f5f7fb', 'axes.facecolor': 'white', 'savefig.facecolor': '#f5f7fb'})
    with activity('Visualize completed scale failure partitions', 'VISUALIZATION_ENGINEERING', 'F3_SYMMETRIC_SCALE'):
        destination.mkdir(parents=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.3), sharey=True)
        for ax, horizon in zip(axes, (1024, 4096)):
            left = np.zeros(4)
            for category, color in zip(CATEGORIES, COLORS):
                values = np.array([float(lookup[case, horizon, category]['fraction']) for case in ('V1', 'V2', 'V3', 'V4')]) * 100
                ax.barh(range(4), values, left=left, color=color, height=.58, edgecolor='white')
                for i, value in enumerate(values):
                    if value >= 7:
                        ax.text(left[i] + value / 2, i, f'{value:.1f}%', ha='center', va='center', color='white', fontsize=11)
                left += values
            np.testing.assert_allclose(left, 100, atol=1e-12)
            ax.set_yticks(range(4), ('V1 확장', 'V2 큰 확장', 'V3 축소', 'V4 큰 축소'))
            ax.set_xlim(0, 100)
            ax.set_xticks((0, 25, 50, 75, 100), ('0%', '25%', '50%', '75%', '100%'))
            ax.set_title(f'{horizon:,}개 관측까지의 최초 사건 결과', loc='left', pad=19, fontsize=14)
            ax.set_xlabel('512개 원래 경로의 비율')
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0)
        axes[0].invert_yaxis()
        fig.suptitle('축소를 놓친 위치: 증거 확인 전인가, 보정 뒤인가', x=.06, ha='left', y=.98, fontsize=21, weight='bold')
        fig.legend([Patch(facecolor=c) for c in COLORS], LABELS, ncol=4, frameon=False,
                   loc='lower center', bbox_to_anchor=(.51, .155), fontsize=11)
        fig.text(.06, .09, 'G2 사건 기억 기준 · 네 결과는 서로 배타적이며 합계100% · 변화 전 오기억이 있으면 이후 확인도 올바른 최초 검출로 세지 않습니다.', fontsize=10)
        fig.text(.06, .045, '같은 시드와 고정 기준의 재실행입니다. 분류기의 경보율·TS-AUC와는 다른 결과이며, 95% 구간과 실제 개수는 원본 표에 있습니다.', fontsize=10, color='#475569')
        fig.subplots_adjust(left=.10, right=.975, bottom=.31, top=.82, wspace=.16)
        fig.savefig(destination / 'SCALE_FIRST_EVENT_FAILURE_PARTITION.png', dpi=160)
        fig.savefig(destination / 'SCALE_FIRST_EVENT_FAILURE_PARTITION.svg')
        plt.close(fig)
        metric = {r['metric']: r for r in rp}
        chosen = ('raw_scale_confirmation_exceeds_admission_cut',
                  'raw_scale_confirmation_subceiling_rejected', 'raw_scale_confirmation_hard_ceiling_rejected')
        labels = ('scale 증거가 G2 문턱을 넘음', '증거 상한보다 낮은 문턱에서 거절', '증거 상한으로도 넘지 못하는 문턱')
        values = np.array([int(metric[name]['block_events']) for name in chosen])
        assert values.sum() == 2275
        fig, ax = plt.subplots(figsize=(13.5, 6.6))
        ax.barh(range(3), values, color=(COLORS[0], COLORS[2], COLORS[1]), height=.55)
        for i, count in enumerate(values):
            ax.text(count + 17, i, f'{count:,}건 ({100 * count / 2275:.1f}%)', va='center', fontsize=13)
        ax.set_yticks(range(3), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1510)
        ax.set_xlabel('지원 가능한 원시 scale 확인 2,275건의 분류')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        fig.suptitle('실제 RP에서 scale 확인이 통과하거나 거절된 이유', x=.035, ha='left', y=.965, fontsize=19, weight='bold')
        fig.text(.035, .11, 'RP 6,400개 시계열 · 3,203,157개 저장 상태 전수 검사 · 기준 변경이나 새로운 예측 없음', fontsize=11)
        fig.text(.035, .065, '거절1,052건 중322건은 core 증거가 대신 확인했습니다. 반복 블록 건수를 독립 표본 수나 사건 검출률로 해석하지 않습니다.', fontsize=10, color='#475569')
        fig.text(.035, .025, 'scale 상한=4 / 원시 확인 기준. G2 문턱이 이 값 이상이면 scale 단독의 새 확인은 불가능하지만, 이전 사건 기억은 남을 수 있습니다.', fontsize=10, color='#475569')
        fig.subplots_adjust(left=.33, right=.965, bottom=.23, top=.84)
        fig.savefig(destination / 'RP_SCALE_ADMISSION_CENSUS.png', dpi=160)
        fig.savefig(destination / 'RP_SCALE_ADMISSION_CENSUS.svg')
        plt.close(fig)
        save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'source_sha256': digest(__file__), 'inputs_sha256': inputs,
            'outputs_sha256': {path.name: digest(path) for path in destination.iterdir() if path.is_file()},
            'new_statistics_or_calibration': False, 'visual_review_required': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    main(parser.parse_args().snapshot)
