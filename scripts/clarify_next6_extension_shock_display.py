from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'src/next6_extension/viewer/template.html';s=p.read_text(encoding='utf-8')
s=s.replace('큰 현재 관측을 별도로 표시합니다. 충격만으로 지속 상태를 확정하지 않습니다.','최근 큰 관측의 영향을 잠시 표시합니다. 충격만으로 지속 상태를 확정하지 않습니다.')
s=s.replace("$('stateExplanation').textContent=explanations[state];", "$('stateExplanation').textContent=explanations[state]+(p[4]?' 현재 시점에는 큰 관측도 함께 감지했습니다.':'');")
p.write_text(s,encoding='utf-8')
