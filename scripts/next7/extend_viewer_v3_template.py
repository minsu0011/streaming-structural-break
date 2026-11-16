"""Extend the preserved state viewer with paired classifiers and causal display."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import atomic

CSS = '''
.viewOptions{display:flex;gap:20px;flex-wrap:wrap;margin:12px 0;font-size:13px}.viewOptions label{display:flex;gap:7px;align-items:center}.viewOptions input{accent-color:var(--teal)}.classifier-card{margin-top:16px}.classifier-head{display:flex;align-items:end;gap:18px;justify-content:space-between}.classifier-head label{min-width:260px;font-size:13px}.classifier-grid{display:grid;grid-template-columns:minmax(400px,1.8fr) minmax(250px,1fr);gap:22px;margin-top:12px}.classifier-plot{height:185px}.classifier-readout{background:#f4f8fa;border:1px solid var(--line);border-radius:10px;padding:12px 16px}.classifier-readout strong{font-size:19px;display:block}.classifier-readout p{font-size:13px}#displayMode{font-weight:700;color:#267481}.controlLegend[hidden]{display:none}@media(max-width:800px){.classifier-grid{grid-template-columns:1fr}.classifier-head{display:block}.classifier-head label{display:block;min-width:0;margin-top:12px}.classifier-plot{height:155px}}@media(max-width:520px){header{display:block}.badge{display:inline-block;margin-top:10px}.toolbar{gap:10px}.controls button{padding:8px}.viewOptions{gap:10px}}
'''

OPTIONS = '''
<div class="viewOptions"><label><input id="showControl" type="checkbox" checked>대응 정상 경로 함께 보기</label><label><input id="showFuture" type="checkbox">전체 미래 경로 보기 · 사후 설명용</label><span id="displayMode">선택 시점까지 표시</span></div>
'''

CARD = '''
<section class="box classifier-card" id="classifierCard"><div class="classifier-head"><div><h2>최종 분류기 점수와 경보</h2><p class="muted">사건 기억과 구분해서 읽습니다. 기존 outer fold 0의 고정 모델과 각 모델의 고정 경보 기준을 사용합니다.</p></div><label>분류기 선택<select id="classifierMethod" aria-label="고정 분류기 선택"></select></label></div><div class="classifier-grid"><div><div class="legend"><span><i class="dot" style="background:#237687"></i>변화 / 충격 경로 점수</span><span class="controlLegend"><i class="dot" style="background:#99a8b4"></i>대응 정상 경로 점수</span><span><i class="dot" style="background:#b47c30"></i>해당 분류기 자체 기준</span></div><svg id="classifierPlot" class="classifier-plot" viewBox="0 0 730 145" role="img" aria-label="최종 분류기와 대응 정상 경로 및 고정 기준"></svg><p class="muted">경보는 반올림 전 점수로 계산했습니다. 점수는 정렬을 위한 값이며 보정된 변화 확률이 아닙니다.</p></div><div class="classifier-readout"><span class="small">선택한 분류기 · 현재 관측</span><strong id="classifierCurrent"></strong><p id="classifierAlarm"></p><p id="classifierFirst"></p><p id="classifierControl"></p><p id="classifierMemoryContrast"></p><button id="classifierJump">분류기 첫 경보로</button></div></div></section>
'''

JAVASCRIPT = r'''
<script>
'use strict';
const METHOD_LABELS={M0:'BL80 · 기존 주력',M3:'M3 · NEXT6 상태 참조',M_EVENT:'G1 · 원래 사건 기억 결합',G2_EVENT:'G2 · H 기준 통과 기억 결합',I4_EVENT_SCALE_TREE:'I4 · 사건·규모 보정',G2_FULL:'G2 · 전체 보정',G3_PAIRWISE_FULL:'G3 · 같은 시점 순위 보정',AMPLITUDE_0_9_M0:'0.9 × BL80 · 크기 대조군'};
$('classifierMethod').innerHTML=DATA.classifier_methods.map(m=>'<option value="'+esc(m)+'">'+esc(METHOD_LABELS[m])+'</option>').join('');
$('classifierMethod').value=DATA.classifier_default;
for(const [id,label] of [['rawPlot','대응 정상 관측'],['ratioPlot','대응 정상 확인 증거'],['memoryPlot','대응 정상의 통과 기억']])$(id).previousElementSibling.insertAdjacentHTML('beforeend','<span class="controlLegend"><i class="dot" style="background:#99a8b4"></i>'+label+'</span>');
plot=function(id,series,colors,range){
 const tr=trace(),w=730,h=145,l=44,r=14,top=9,bottom=22,n=tr.raw.length,control=$('showControl').checked,future=$('showFuture').checked;
 series=series.slice();colors=colors.slice();
 if(control){let extra=id==='rawPlot'?tr.control_raw:id==='ratioPlot'?tr.control_values.ratio:id==='memoryPlot'?tr.control_values.qualified_event:null;if(extra){series.push(extra);colors.push('#99a8b4')}}
 const visible=series.map(values=>future?values:values.slice(0,time+1));
 let min=range?range[0]:Math.min(...visible.flat()),max=range?range[1]:Math.max(...visible.flat());
 if(max===min){max+=1;min-=1}const pad=(max-min)*.08;if(!range){min-=pad;max+=pad}
 const x=t=>l+(w-l-r)*t/(n-1),y=v=>top+(h-top-bottom)*(max-v)/(max-min);let s='';
 const at=tr.info.scheduled,end=tr.info.intervention_end,displayEnd=future?n-1:time;
 if(at<=displayEnd&&!tr.case.startsWith('N'))s+='<rect x="'+x(at)+'" y="'+top+'" width="'+Math.max(1,x(Math.min(end===null?displayEnd:end,displayEnd))-x(at))+'" height="'+(h-top-bottom)+'" fill="#f7dfb5" opacity=".35"/>';
 for(let j=0;j<3;j++){let v=min+(max-min)*j/2;s+='<path d="M'+l+','+y(v)+'H'+(w-r)+'" stroke="#e3eaee"/><text x="'+(l-6)+'" y="'+(y(v)+4)+'" text-anchor="end" font-size="10" fill="#61717b">'+v.toFixed(1)+'</text>'}
 for(let j=0;j<visible.length;j++){let dashed=colors[j]==='#99a8b4'?' stroke-dasharray="4 3"':colors[j]==='#b47c30'||(id==='ratioPlot'&&j===1)?' stroke-dasharray="6 3"':'';s+='<path d="'+path(visible[j],x,y)+'" fill="none" stroke="'+colors[j]+'" stroke-width="1.65"'+dashed+'/>'}
 s+='<path d="M'+x(time)+','+top+'V'+(h-bottom)+'" stroke="#172a38" stroke-width="1.2"/>';
 for(let t of [0,Math.floor((n-1)/2),n-1])s+='<text x="'+x(t)+'" y="'+(h-4)+'" text-anchor="middle" font-size="10" fill="#61717b">'+t+'</text>';
 $(id).innerHTML=s;
};
const renderStateOnly=render;
render=function(){
 renderStateOnly();const tr=trace(),method=$('classifierMethod').value,c=tr.classifiers[method],control=$('showControl').checked;
 let series=[c.score],colors=['#237687'];if(control){series.push(c.control_score);colors.push('#99a8b4')}series.push(Array(tr.raw.length).fill(c.cut));colors.push('#b47c30');
 plot('classifierPlot',series,colors,[0,1]);
 const first=c.alarm.slice(0,time+1).findIndex(x=>x>0),controlFirst=c.control_alarm.slice(0,time+1).findIndex(x=>x>0);
 $('classifierCurrent').textContent=c.score[time].toFixed(4)+' / 기준 '+c.cut.toFixed(4);
 $('classifierAlarm').textContent='현재 경보: '+(c.alarm[time]?'예':'아니요');
 $('classifierFirst').textContent='선택 시점까지 첫 경보: '+when(first);
 $('classifierControl').textContent='대응 정상 경로: '+c.control_score[time].toFixed(4)+' · 첫 경보 '+when(controlFirst);
 $('classifierMemoryContrast').textContent='현재 사건 기억 '+fmt(val('qualified_event'))+' · 기억이 있어도 분류기 기준을 넘지 않을 수 있습니다.';
 $('displayMode').textContent=$('showFuture').checked?'전체 경로 · 사후 표시':'선택 시점까지 표시';
 document.querySelectorAll('.controlLegend').forEach(e=>e.hidden=!control);
 window.NEXT7_VIEW_STATE={...window.NEXT7_VIEW_STATE,classifier:method,classifierScore:c.score[time],classifierCut:c.cut,classifierAlarm:c.alarm[time],controlClassifierAlarm:c.control_alarm[time],classifierFirst:first,controlClassifierFirst:controlFirst,showFuture:$('showFuture').checked,showControl:control};
};
for(const id of ['classifierMethod','showControl','showFuture'])$(id).addEventListener('change',()=>{stop();render()});
$('classifierJump').addEventListener('click',()=>{stop();const c=trace().classifiers[$('classifierMethod').value],first=c.alarm.findIndex(x=>x>0);time=first>=0?first:trace().raw.length-1;render()});
render();
</script>
'''


def main():
    source = (ROOT / 'src/next7/viewer.html').read_text(encoding='utf-8')
    target = ROOT / 'src/next7/viewer_v3.html'
    assert target.read_text(encoding='utf-8') == source
    source = source.replace('</style>', CSS + '</style>', 1)
    source = source.replace('<div class="layout">', OPTIONS + '<div class="layout">', 1)
    source = source.replace('<div class="footer">', CARD + '<div class="footer">', 1)
    source = source.replace('</html>', JAVASCRIPT + '</html>', 1)
    source = source.replace('<title>NEXT7 · 동적 상태와 사건 기억</title>', '<title>NEXT7 · 상태·사건 기억·분류기 경보</title>')
    atomic(target, source, immutable=False)


if __name__ == '__main__':
    main()
