from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'src/next6_extension/viewer/template.html';s=p.read_text(encoding='utf-8')
s=s.replace('충격, 추가 확인, 지속 변화, 회복을 시간에 따라 살펴봅니다','충격, 추가 확인, 지속 증거, 회복을 시간에 따라 살펴봅니다')
s=s.replace("function edgeDetail(e){const ci=", "function edgeDetail(e){if(e.sourceSupport===0){$('edgeDetail').textContent=stateNames[e.a]+' 상태의 학습 관측 근거가 없어 이 확률은 사용하지 않습니다.';return;}const ci=")
s=s.replace("stateNames[i]+(self?` / 자기 유지 ${(self.prob*100).toFixed(1)}% / 지원 ${self.sourceSupport} series`:'')", "stateNames[i]+(self&&self.sourceSupport>0?` / 자기 유지 ${(self.prob*100).toFixed(1)}% / 지원 ${self.sourceSupport} series`:' / 학습 관측 근거 없음')")
p.write_text(s,encoding='utf-8')
