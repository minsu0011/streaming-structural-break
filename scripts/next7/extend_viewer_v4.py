"""Add exact confirmation explanations to the preserved eight-example viewer."""
from pathlib import Path
import copy
import json
import os
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, read_json, save, digest, utc_now, activity, note
from src.next7.calibration import load_calibration, cutoff
from src.next7.viewer_evidence import evidence_flags, FLAG_COLUMNS, STATUS_NAMES

CSS = '''
.evidence-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:16px 0}
.evidence-node{border:1px solid #d7e3e6;background:#f6f9fa;border-radius:10px;padding:13px}
.evidence-node strong{display:block;font-size:18px;margin:6px 0}.evidence-node[data-pass="true"]{border-color:#60a58c;background:#edf8f2}
.evidence-node[data-blocked="true"]{border-color:#d6a66b;background:#fff7ed}
#evidenceStatus{font-size:17px;font-weight:700;color:#255d6a}.evidence-control{border-top:1px solid #d7e3e6;padding-top:10px}
@media(max-width:1000px){.evidence-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:420px){.evidence-grid{grid-template-columns:1fr}}
'''
CARD = '''
<section class="box classifier-card" id="evidenceCard" aria-labelledby="evidenceHeading">
<h2 id="evidenceHeading">새 확인은 어디에서 멈췄나요?</h2>
<p class="muted">선택 시점까지 완료된 최근 네 블록만 읽습니다. 반대 방향의 블록도 확인 강도의 평균에 포함됩니다.</p>
<p id="evidenceClock" class="small"></p><p id="evidenceStatus"></p>
<div class="evidence-grid">
<div class="evidence-node"><span class="small">최근 scale 블록의 방향</span><strong id="evidenceSigns"></strong><span id="evidenceSignNote" class="small"></span></div>
<div class="evidence-node" id="evidenceRawNode"><span class="small">scale의 원시 확인</span><strong id="evidenceRaw"></strong><span class="small">확인 비율이 1보다 커야 합니다.</span></div>
<div class="evidence-node" id="evidenceAdmissionNode"><span class="small">공통 H 문턱에서의 scale 확인</span><strong id="evidenceCut"></strong><span id="evidenceCeiling" class="small"></span></div>
<div class="evidence-node" id="evidenceMemoryNode"><span class="small">이번 관측과 과거 기록</span><strong id="evidenceFresh"></strong><span id="evidenceRemembered" class="small"></span></div>
</div>
<p id="evidenceCore" class="small"></p><p id="evidenceControl" class="evidence-control small"></p>
<p class="muted">scale가 거절돼도 core 증거가 대신 확인할 수 있습니다. 이 설명과 상태망의 선은 실제 구조변화나 인과관계를 보증하지 않습니다.</p>
</section>
'''
JAVASCRIPT = r'''
<script>
'use strict';
const EVIDENCE_LABELS=['H 보정 지원이 부족해 새 사건 확인을 보류합니다.','네 블록이 완성되기 전입니다.',
 'scale 증거가 원시 확인 기준을 넘지 못했습니다.','공통 문턱이 scale 증거 상한 이상이라 scale 단독 확인이 불가능합니다.',
 'scale는 원시 확인을 했지만 공통 H 문턱에서 거절됐습니다.','최근 블록의 scale 증거가 공통 H 문턱을 통과했습니다.'];
const renderBeforeEvidence=render;
render=function(){
 renderBeforeEvidence();const tr=trace(),d=tr.confirmation_diagnostics,f=d.observed[time],c=d.control[time];
 const raw=val('scale_ratio'),cut=val('admission_cut'),status=f[0];
 $('evidenceClock').textContent=f[1]<0?'아직 완료된 블록이 없습니다.':('최근 완료 블록: t='+f[1]+' · '+(f[2]?'이번 관측에서 새 블록 완성':'이번 관측에서는 기존 블록 증거를 유지'));
 $('evidenceStatus').textContent=EVIDENCE_LABELS[status];
 $('evidenceSigns').textContent='음 '+f[3]+' · 양 '+f[4]+' / '+f[9]+'블록';
 $('evidenceSignNote').textContent=f[9]<4?'네 완료 블록을 기다립니다.':'같은 방향이 3개 이상이어도 평균 강도가 부족하면 미확정입니다.';
 $('evidenceRaw').textContent=raw.toFixed(4)+' / 기준 1';
 $('evidenceCut').textContent=cut.toFixed(4)+' · '+(f[6]?'scale 통과':'scale 미통과');
 $('evidenceCeiling').textContent='scale 비율의 최대값 '+d.scale_ratio_ceiling.toFixed(4)+' · 원시 확인 강도 기준 '+d.scale_confirmation_cut.toFixed(4);
 $('evidenceFresh').textContent=f[7]?'새 G2 확인 있음':'새 G2 확인 없음';
 $('evidenceRemembered').textContent=f[8]?'과거 또는 현재의 통과 기록은 유지됩니다.':'아직 H 기준을 통과한 사건 기록이 없습니다.';
 $('evidenceCore').textContent='최근 완료 블록의 core 경로: '+(f[5]?'공통 H 문턱 통과':'공통 H 문턱 미통과')+' · 분류기 경보는 아래에서 별도로 확인합니다.';
 $('evidenceControl').hidden=!$('showControl').checked;
 $('evidenceControl').textContent='대응 정상 경로의 scale: '+EVIDENCE_LABELS[c[0]]+' 현재까지 사건 기록 '+(c[8]?'있음':'없음');
 $('evidenceRawNode').dataset.pass=String(status>=3);$('evidenceAdmissionNode').dataset.pass=String(f[6]===1);
 $('evidenceAdmissionNode').dataset.blocked=String(status===3||status===4);$('evidenceMemoryNode').dataset.pass=String(f[8]===1);
 window.NEXT7_VIEW_STATE={...window.NEXT7_VIEW_STATE,evidenceFlags:f.slice(),controlEvidenceFlags:c.slice(),scaleStatus:DATA.evidence_status_names[status]};
};
render();
</script>
'''


def main():
    previous = BASE / 'visuals/v3'
    target = BASE / 'visuals/v4'
    assert not target.exists()
    original = read_json(previous / 'BUILD_DONE.json')
    for name in ('DYNAMIC_STATE_TRANSITION_NEXT7.html', 'STATE_NETWORK_TRACES.json', 'EXACT_SYNTHETIC_CURVES.npz'):
        assert digest(previous / name) == original['files_sha256'][name]
    tests = BASE / 'audit/VIEWER_EVIDENCE_TESTS_001.xml'
    xml = ET.parse(tests)
    assert len(xml.findall('.//testcase')) == 4 and not xml.findall('.//failure') and not xml.findall('.//error')
    assert read_json(BASE / 'audit/VIEWER_EVIDENCE_TEST_ACCESS_001.json')['status'] == 'PASS'
    save('visuals/v4/EXTENSION_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'source_sha256': {name: digest(ROOT / name) for name in ('scripts/next7/extend_viewer_v4.py', 'src/next7/viewer_evidence.py')},
        'original_v3_build_sha256': digest(previous / 'BUILD_DONE.json'), 'tests_sha256': digest(tests),
        'scope': 'Display-only clarification using the original exact synthetic state arrays. Same eight examples and every original classifier curve/boolean alarm. No new predictions, calibration, path selection or population estimate.',
        'flag_columns': FLAG_COLUMNS, 'status_names': STATUS_NAMES,
        'future_truth_used_in_explanation': False, 'status_decisions_before_display_rounding': True})
    data = read_json(previous / 'STATE_NETWORK_TRACES.json')
    old_data = copy.deepcopy(data)
    calibration = load_calibration()
    checks = []
    with activity('Extend dynamic viewer with exact confirmation bottlenecks', 'VISUALIZATION_ENGINEERING', 'F3_SYMMETRIC_SCALE'):
        with np.load(previous / 'EXACT_SYNTHETIC_CURVES.npz', allow_pickle=False) as curves:
            for trace in data['traces']:
                scale_cut = cutoff(calibration, trace['H_length'])
                traces = []
                for suffix, values_key in (('qualified_state', 'values'), ('control_qualified_state', 'control_values')):
                    values = curves[trace['id'] + '_' + suffix]
                    assert values.shape == (trace['horizon'], 45)
                    for key, col in (('scale_ratio', 11), ('qualified_event', 30), ('admission_cut', 42), ('qualified_fresh', 40)):
                        np.testing.assert_array_equal(np.round(values[:, col], 6), trace[values_key][key])
                    flags = evidence_flags(values, scale_cut)
                    for prefix in (1, 16, 64, 257):
                        np.testing.assert_array_equal(evidence_flags(values[:prefix], scale_cut), flags[:prefix])
                    traces.append(flags.tolist())
                    checks.append({'trace': trace['id'], 'path': suffix, 'points': len(values),
                        'exact_stored_state_used': True, 'all_latest_block_scale_core_admission_decisions_match': True,
                        'prefixes_checked': [1, 16, 64, 257], 'status_counts': np.bincount(flags[:, 0], minlength=6).tolist()})
                trace['confirmation_diagnostics'] = {'scale_confirmation_cut': scale_cut,
                    'scale_ratio_ceiling': 4 / scale_cut, 'observed': traces[0], 'control': traces[1]}
        for before, after in zip(old_data['traces'], data['traces']):
            assert before == {key: value for key, value in after.items() if key != 'confirmation_diagnostics'}
        data.update(viewer_version=4, evidence_flag_columns=FLAG_COLUMNS, evidence_status_names=STATUS_NAMES,
                    explanation_uses_exact_state_before_rounding=True, explanation_uses_future_truth=False)
        save('visuals/v4/STATE_NETWORK_TRACES.json', data)
        html = (previous / 'DYNAMIC_STATE_TRANSITION_NEXT7.html').read_text(encoding='utf-8')
        start_marker = '<script id="traceData" type="application/json">'
        start = html.index(start_marker) + len(start_marker)
        end = html.index('</script>', start)
        html = html[:start] + json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/') + html[end:]
        html = html.replace('</style>', CSS + '</style>', 1)
        marker = '<section class="box classifier-card" id="classifierCard">'
        assert html.count(marker) == 1
        html = html.replace(marker, CARD + marker, 1).replace('</html>', JAVASCRIPT + '\n</html>')
        save('visuals/v4/DYNAMIC_STATE_TRANSITION_NEXT7.html', html)
        copied = {}
        for name in ['EXACT_SYNTHETIC_CURVES.npz', *[path.name for path in previous.glob('NETWORK_CLASSIFIER_*')]]:
            source = previous / name
            assert digest(source) == original['files_sha256'][name]
            with source.open('rb') as input_file, (target / name).open('xb') as output_file:
                output_file.write(input_file.read())
            assert digest(target / name) == digest(source)
            copied[name] = digest(source)
        save('visuals/v4/BUILD_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'source_sha256': digest(__file__), 'extension_lock_sha256': digest(target / 'EXTENSION_LOCK.json'),
            'original_v3_build_sha256': digest(previous / 'BUILD_DONE.json'), 'checks': checks,
            'paired_state_paths': 16, 'all_original_classifier_curves_and_alarms_unchanged': True,
            'exact_flag_decisions_before_rounding': True, 'new_models_or_thresholds': False,
            'original_static_images_copied_unchanged': copied, 'standalone_HTML': True,
            'files_sha256': {p.name: digest(p) for p in target.iterdir() if p.is_file()},
            'browser_and_visual_review_required': True})
    note('Dynamic viewer confirmation panel built', 'Same eight exact examples; scale evidence, H rejection, core fallback and retained memory separated', 'F3_SYMMETRIC_SCALE')


if __name__ == '__main__':
    main()
