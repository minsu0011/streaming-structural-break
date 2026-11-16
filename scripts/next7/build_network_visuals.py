"""Standalone HTML plus computed PNG/SVG figures. No generative artwork."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib import font_manager
from src.next7.common import BASE, save, read_json, digest, utc_now, activity

COLORS = ['#88959f', '#349781', '#de933e', '#73879a', '#c36851', '#329bac', '#9171a1']
POSITIONS = [(0.5, .88), (.24, .64), (.77, .64), (.5, .45), (.77, .17), (.24, .17), (.06, .88)]


def diagram(axis, states, names, selected):
    state = np.asarray(states, dtype=int)
    counts = np.zeros((7, 7), dtype=int)
    for a, b in zip(state[:selected], state[1:selected + 1]):
        if a != b:
            counts[a, b] += 1
    for a, b in zip(*np.nonzero(counts)):
        x1, y1 = POSITIONS[a]
        x2, y2 = POSITIONS[b]
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12,
            shrinkA=23, shrinkB=25, connectionstyle='arc3,rad=.13', color='#9cabb4',
            linewidth=min(3., .7 + np.log1p(counts[a, b]) * .5))
        axis.add_patch(arrow)
        distance = np.hypot(x2 - x1, y2 - y1)
        label_x = (x1 + x2) / 2 - (y2 - y1) / distance * .055
        label_y = (y1 + y2) / 2 + (x2 - x1) / distance * .055
        axis.text(label_x, label_y, str(counts[a, b]),
            fontsize=8, color='#5e7481', ha='center', bbox=dict(facecolor='white', edgecolor='none', pad=1))
    active = int(state[selected])
    for j, (x, y) in enumerate(POSITIONS):
        axis.add_patch(Circle((x, y), .084 if j == active else .072,
            facecolor=COLORS[j] if j == active else '#f1f5f7', edgecolor=COLORS[j], linewidth=2 if j == active else 1))
        axis.text(x, y, names[j], ha='center', va='center', fontsize=9,
            weight='bold' if j == active else 'normal', color='white' if j == active else '#314a59')
    axis.set(xlim=(-.05, 1), ylim=(0, 1), aspect='equal')
    axis.axis('off')


def plot_trace(data, trace, directory):
    v = trace['values']
    n, selected = len(trace['raw']), trace['initial_time']
    times = np.arange(n)
    fig = plt.figure(figsize=(14.2, 8.8), facecolor='#f5f8fa', layout='constrained')
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.05], height_ratios=[1, 1, 1])
    axes = [fig.add_subplot(grid[j, :2]) for j in range(3)]
    graph = fig.add_subplot(grid[:2, 2])
    memory = fig.add_subplot(grid[2, 2])
    fig.suptitle(trace['title'] + f"   ·   {trace['case']} / seed {trace['seed']}",
        fontsize=18, color='#172a38', weight='bold')
    axes[0].plot(times, trace['raw'], lw=.8, color='#267a88')
    axes[0].set_ylabel('관측값')
    axes[1].plot(times, v['ratio'], lw=1.2, color='#3b728e', label='현재 확정 증거')
    axes[1].plot(times, v['admission_cut'], lw=1.3, color='#ad7431', label='관측 횟수별 H 기준')
    axes[1].set_ylabel('확정 증거 / 기준')
    axes[1].legend(loc='upper left', fontsize=9, frameon=False)
    axes[2].plot(times, v['raw_event'], lw=1.2, color='#9a708a', label='원래 사건 기억')
    axes[2].plot(times, v['qualified_event'], lw=1.7, color='#008475', label='H 기준 통과 기억')
    axes[2].set(ylabel='사건 증거 · 확률 아님', xlabel='관측 시점 t', ylim=(-.02, 1.02))
    axes[2].legend(loc='upper left', fontsize=9, frameon=False)
    for axis in axes:
        axis.set_facecolor('white')
        at, end = trace['info']['scheduled'], trace['info']['intervention_end']
        if at < n and trace['case'] != 'N0':
            axis.axvspan(at, n - 1 if end is None else min(end, n - 1), color='#efd6ac', alpha=.3, lw=0)
        axis.axvline(selected, lw=1, color='#172a38')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', color='#e4ebef', linewidth=.7)
        axis.set_xlim(0, n - 1)
        axis.tick_params(labelsize=9)
    diagram(graph, v['current_state'], data['state_names'], selected)
    graph.set_title(f'현재 상태 전이 · t={selected}까지\n화살표 숫자는 관측된 이동 횟수', fontsize=11, color='#304959')
    memory.axis('off')
    first = int(v['qualified_first'][selected])
    target = int(trace['tau_offline_only'] >= 0 and selected >= trace['tau_offline_only'])
    lines = [f"선택 시점 현재 상태: {data['state_names'][int(v['current_state'][selected])]}",
        f"원래 사건 증거: {v['raw_event'][selected]:.3f}",
        f"H 기준 통과 증거: {v['qualified_event'][selected]:.3f}",
        f"첫 통과 시점: {'없음' if first < 0 else first}", f'공식 target: {target} · 사후 표시',
        f"H: {trace['H_length']}개 / 기준 {'지원' if v['admission_support'][selected] else '부족'}", '',
        '관측된 상태 경로 · 인과관계 의미 없음', '성공·실패 예시이며 전체 성능을 대표하지 않음']
    memory.text(.02, .96, '\n'.join(lines), va='top', fontsize=10, color='#304959', linespacing=1.55)
    for suffix in ('png', 'svg'):
        path = directory / (trace['id'] + '.' + suffix)
        assert not path.exists()
        fig.savefig(path, dpi=175, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(version):
    directory = BASE / 'visuals' / version
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError('Use a new visual revision directory; preserve previous review images')
    directory.mkdir(parents=True, exist_ok=True)
    data = read_json(BASE / 'visuals/STATE_NETWORK_TRACES.json')
    template = ROOT / 'src/next7/viewer.html'
    html = template.read_text(encoding='utf-8')
    assert html.count('__TRACE_DATA__') == 1
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    with activity('render actual dynamic state network figures ' + version, 'VISUAL_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        save(f'visuals/{version}/DYNAMIC_STATE_TRANSITION_NEXT7.html', html.replace('__TRACE_DATA__', payload))
        font = Path('C:/Windows/Fonts/malgun.ttf')
        assert font.exists()
        font_manager.fontManager.addfont(str(font))
        plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
            'svg.fonttype': 'none', 'savefig.bbox': 'tight'})
        for trace in data['traces']:
            plot_trace(data, trace, directory)
        save(f'visuals/{version}/BUILD.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'trace_sha256': digest(BASE / 'visuals/STATE_NETWORK_TRACES.json'),
            'template_sha256': digest(template), 'builder_sha256': digest(__file__),
            'files_sha256': {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
            'computed_from_actual_synthetic_outputs': True, 'learned_graph_edges_invented': False,
            'raw_competition_data_included': False, 'browser_review_pending': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--version', required=True)
    main(p.parse_args().version)
