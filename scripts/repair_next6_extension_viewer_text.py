"""Repair UTF-8 UI text passed incorrectly through Windows legacy pipe encoding."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'src/next6_extension/viewer/template.html';s=p.read_text(encoding='utf-8')
start=s.index("$('targetNote').textContent=")+len("$('targetNote').textContent=")
end=s.index("$('showTruth').checked?",start)
s=s[:start]+"$('pathSelect').value==='control'&&c.control?'대응 정상 경로의 변화 발생 정답은 0입니다. 세로선은 변화·충격 경로와의 비교 기준 위치입니다.':"+s[end:]
s=s.replace('공식·합성 변화 위치를 사후 표시 (모델 입력 아님)','정답·충격 위치를 사후 표시 (모델 입력 아님)')
s=s.replace("ctx.fillText('사후 변화 위치',","ctx.fillText(c.tau>=0?'사후 변화 위치':'충격 위치',")
p.write_text(s,encoding='utf-8')
