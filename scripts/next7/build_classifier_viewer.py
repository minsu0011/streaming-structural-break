"""Offline dynamic network and static figures from the same paired trace data."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from src.next7.common import BASE, read_json, save, digest, utc_now, activity
from scripts.next7.build_network_visuals import diagram


def plot(data, tr, directory):
    t = np.arange(len(tr['raw']))
    selected = tr['initial_time']
    fig = plt.figure(figsize=(15, 10.8), facecolor='#f5f8fa')
    grid = fig.add_gridspec(4, 3, width_ratios=[1, 1, 1.06], left=.06, right=.975,
        top=.88, bottom=.125, wspace=.27, hspace=.3)
    axes = [fig.add_subplot(grid[i, :2]) for i in range(4)]
    graph = fig.add_subplot(grid[:3, 2])
    note = fig.add_subplot(grid[3, 2])
    fig.suptitle(tr['title'], x=.04, ha='left', y=.965, fontsize=22, color='#172a38', weight='bold')
    fig.text(.04, .925, f"{tr['case']} · 고정 seed {tr['seed']} · H {tr['H_length']:,} · 그림의 선택 시점 t={selected}",
        fontsize=12, color='#506773')
    axes[0].plot(t, tr['raw'], color='#277b89', lw=.85, label='변화 / 충격 관측')
    axes[0].plot(t, tr['control_raw'], color='#9eaeb8', lw=.75, ls='--', label='대응 정상 관측')
    axes[0].set_ylabel('관측값')
    v = tr['values']
    axes[1].plot(t, v['ratio'], color='#497d99', lw=1.1, label='현재 확정 증거')
    axes[1].plot(t, v['admission_cut'], color='#ad783a', lw=1.2, ls='--', label='관측 횟수별 H 기준')
    axes[1].set_ylabel('확정 증거 / H 기준')
    axes[2].plot(t, v['raw_event'], color='#a5809e', lw=1.1, label='원래 사건 기억')
    axes[2].plot(t, v['qualified_event'], color='#048776', lw=1.6, label='H 기준 통과 기억')
    axes[2].plot(t, tr['control_values']['qualified_event'], color='#9eaeb8', lw=1, ls='--', label='대응 정상의 통과 기억')
    axes[2].set(ylabel='사건 증거 · 확률 아님', ylim=(-.03, 1.03))
    for method, color, label in [('M0', '#4b6d91', 'BL80'), ('G2_FULL', '#c36d4e', 'G2 전체 보정')]:
        value = tr['classifiers'][method]
        axes[3].plot(t, np.array(value['score']) - value['cut'], color=color, lw=1.1, label=label+' 점수 - 자체 기준')
    axes[3].axhline(0, color='#71848f', lw=1, ls='--')
    axes[3].set(ylabel='양수일 때 분류기 경보', xlabel='관측 시점 t · 전체 경로는 사후 설명용')
    for axis in axes:
        at, end = tr['info']['scheduled'], tr['info']['intervention_end']
        if not tr['case'].startswith('N') and at < len(t):
            axis.axvspan(at, len(t) - 1 if end is None else min(end, len(t) - 1), color='#f3d7a8', alpha=.22, lw=0)
        axis.axvline(selected, color='#213a48', lw=1)
        axis.set_xlim(0, len(t) - 1)
        axis.legend(loc='upper left', fontsize=8.5, ncol=2, frameon=False)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', color='#e0e8eb', lw=.7)
        axis.set_facecolor('white')
    graph.set_title('선택 시점까지 실제 상태 전이', fontsize=13, color='#26414d')
    diagram(graph, v['current_state'], data['state_names'], selected)
    note.axis('off')
    score = tr['classifiers']['G2_FULL']
    current = data['state_names'][int(v['current_state'][selected])]
    truth = int(tr['tau_offline_only'] >= 0 and selected >= tr['tau_offline_only'])
    text = (f"현재 상태: {current}\nH 기준 통과 기억: {v['qualified_event'][selected]:.3f}\n"
        f"G2 분류기 경보: {'예' if score['alarm'][selected] else '아니요'}\n사후 정답 target: {truth}")
    note.text(.03, .9, text, transform=note.transAxes, va='top', fontsize=13, color='#254756', linespacing=1.7,
        bbox=dict(facecolor='white', edgecolor='#cfdde3', boxstyle='round,pad=.7'))
    fig.text(.04, .035,
        '화살표 숫자는 선택 시점까지 상태가 바뀐 횟수입니다. 학습된 인과관계가 아닙니다.\n'
        '기억과 최종 분류기 경보는 서로 다른 출력입니다. 고정 fold 0의 연구 모델이며 주력 BL80 역할은 유지합니다.\n'
        '성공·실패 예시는 원래 선택을 그대로 사용했습니다. 이 한 경로를 모집단 성능으로 해석하지 않습니다.',
        fontsize=10, color='#566c79')
    for extension in ('png', 'svg'):
        fig.savefig(directory / f'NETWORK_CLASSIFIER_{tr["id"]}.{extension}', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    directory = BASE / 'visuals/v3'
    receipt = read_json(directory / 'DATA_EXPORT_DONE.json')
    assert receipt['status'] == 'PASS' and digest(directory / 'STATE_NETWORK_TRACES.json') == receipt['trace_sha256']
    data = read_json(directory / 'STATE_NETWORK_TRACES.json')
    template = ROOT / 'src/next7/viewer_v3.html'
    font = Path('C:/Windows/Fonts/malgun.ttf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
        'axes.unicode_minus': False, 'svg.fonttype': 'none'})
    with activity('Build offline paired classifier-state network and eight images', 'SCIENTIFIC_VISUALIZATION', 'F5_CAUSAL_NESTED_INTEGRATION'):
        content = template.read_text(encoding='utf-8')
        assert content.count('__TRACE_DATA__') == 1
        content = content.replace('__TRACE_DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))
        html_path = directory / 'DYNAMIC_STATE_TRANSITION_NEXT7.html'
        if html_path.exists():
            assert html_path.read_text(encoding='utf-8') == content
        else:
            save('visuals/v3/DYNAMIC_STATE_TRANSITION_NEXT7.html', content)
        for tr in data['traces']:
            plot(data, tr, directory)
        save('visuals/v3/BUILD_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'source_sha256': digest(__file__), 'template_sha256': digest(template),
            'data_export_sha256': digest(directory / 'DATA_EXPORT_DONE.json'),
            'images': 8, 'svg_images': 8, 'standalone_HTML': True, 'external_network_dependency': False,
            'files_sha256': {p.name: digest(p) for p in directory.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
