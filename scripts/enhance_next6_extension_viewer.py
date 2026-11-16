"""Apply a one-time, source-preserving UI enhancement using exact UTF-8 text."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'src/next6_extension/viewer/template.html';s=p.read_text(encoding='utf-8')
assert 'id="transitionReadout"' not in s
s=s.replace('</style>', '.transition-readout{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}.transition-readout>div{padding:10px;background:#f3f7f9;border-radius:8px;font-size:12px}.transition-readout strong{display:block;font-size:18px}.edge-detail{min-height:76px;padding:12px;border:1px solid #dce5e9;border-radius:8px;font-size:12px}.edge-target{cursor:pointer}.edge-target:focus{outline:none;stroke:#182f3e;stroke-width:5}</style>')
s=s.replace('<p class="subtle">화살표는 상태를 떠날 때의 비중입니다.', '<div class="transition-readout" id="transitionReadout"></div><div class="edge-detail" id="edgeDetail" aria-live="polite">화살표나 상태 원을 선택하면 확률의 오차 범위와 지원 자료를 볼 수 있습니다.</div><p class="subtle">화살표는 상태를 떠날 때의 비중입니다.')
s=s.replace('원에 마우스를 올리면 자기 유지 확률과 학습 지원 수를 볼 수 있습니다.', '화살표와 원을 클릭하거나 Tab으로 선택할 수 있습니다. 아래 두 확률은 방금 관측한 같은 전이를 두 조건에서 함께 계산합니다.')
s=s.replace('모든 조건부 확률은 설명용이며 경제적 인과관계를 뜻하지 않습니다.', '모든 조건부 확률은 설명용이며 경제적 인과관계를 뜻하지 않습니다. 표시한 오차 범위는 학습 series를 통째로 500회 재표집한 95% 구간입니다. 7개 상태 이름만으로 다음 상태가 결정되는 것은 아닙니다. 실제 계산은 최근 네 구간, 구간 내 시점, 충격 대기와 회복 횟수를 함께 기억합니다. 화살표는 이런 상황을 합친 평균 빈도입니다.')
needle="function graph(state){"
extra="""function percent(p){return (p*100).toFixed(1)+'%';}
function edgeDetail(e){const ci=e.a===e.b?e.probCI:e.jumpCI,valid=ci&&ci.every(v=>v!==null);$('edgeDetail').textContent=stateNames[e.a]+' → '+stateNames[e.b]+' · '+(e.phase?'변화 이후':'정상')+' 조건 | '+(e.a===e.b?'자기 유지 ':'떠나는 전이 중 ')+percent(e.a===e.b?e.prob:e.jump)+(valid?' · 95% 범위 '+percent(ci[0])+'–'+percent(ci[1]):' · 오차 범위를 계산할 자료 부족')+' · 이 전이를 보인 '+e.support+' series / 출발 상태를 보인 '+e.sourceSupport+' series. 전체 관측 시점의 전이확률은 '+percent(e.prob)+'입니다.';}
function transitionReadout(state){const box=$('transitionReadout');box.replaceChildren();if(t===0){box.textContent='첫 관측이므로 이전 상태와의 전이가 아직 없습니다.';return;}const previous=snapshot(pathNow(),t-1)[1];for(const phase of [0,1]){const e=DATA.macro.find(e=>e.phase===phase&&e.a===previous&&e.b===state),d=document.createElement('div'),label=document.createElement('span'),value=document.createElement('strong'),support=document.createElement('span');label.textContent=(phase?'변화 이후':'정상')+' 조건에서 방금 전이';value.textContent=percent(e.prob);support.textContent=e.support>=8?'지원 '+e.support+' series':'지원 부족: '+e.support+' series';d.append(label,value,support);box.append(d);}}
"""
s=s.replace(needle,extra+needle)
old="svgEl('path',{d:`M ${start} Q ${mid} ${end}`,fill:'none',stroke:colors[e.b],'stroke-width':1+e.jump*5,opacity:.5,'marker-end':'url(#arrow'+e.b+')'},svg);"
new="const path=svgEl('path',{d:`M ${start} Q ${mid} ${end}`,fill:'none',stroke:colors[e.b],'stroke-width':1+e.jump*5,opacity:.5,'marker-end':'url(#arrow'+e.b+')',class:'edge-target',tabindex:0,role:'button','aria-label':stateNames[e.a]+'에서 '+stateNames[e.b]+' 전이 상세'},svg);path.addEventListener('click',()=>edgeDetail(e));path.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();edgeDetail(e);}});svgEl('title',{},path,'선택하면 95% 오차 범위와 지원 자료를 표시합니다.');"
assert old in s;s=s.replace(old,new)
old="const self=table.find(e=>e.a===i&&e.b===i);svgEl('title',{},g,"
new="const self=table.find(e=>e.a===i&&e.b===i);g.setAttribute('tabindex','0');g.setAttribute('role','button');g.setAttribute('aria-label',stateNames[i]+' 자기 유지 상세');g.style.cursor='pointer';g.addEventListener('click',()=>edgeDetail(self));g.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();edgeDetail(self);}});svgEl('title',{},g,"
assert old in s;s=s.replace(old,new)
s=s.replace("$('caseSupport').textContent=c.support;", "transitionReadout(state);$('caseSupport').textContent=c.support;")
p.write_text(s,encoding='utf-8')
print('Added supported uncertainty and both-condition transition readouts.')
