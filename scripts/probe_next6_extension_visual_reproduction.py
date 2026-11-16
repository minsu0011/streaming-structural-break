"""Rebuild figures and viewer from raw-free derived inputs in an isolated copy."""
from pathlib import Path
import sys,os,shutil,subprocess,argparse,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.next6_extension.common import *
from package_next6_extension import selected_files

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preliminary','final'],required=True);a=parser.parse_args()
    dest=BASE/f'runtime/visual_reproduction_{a.stage}';assert not dest.exists();dest.mkdir()
    selected=[]
    for p in selected_files():
        rel=p.relative_to(BASE) if p.is_relative_to(BASE) else None
        if p.parent==BASE and p.suffix in ('.json','.csv'):selected.append(p)
        elif rel and rel.parts[0] in ('calibration','head_memory','extended_synthetic') and p.suffix in ('.json','.csv'):selected.append(p)
    selected+=list((ROOT/'scripts').glob('render_next6_extension*.py'))+[ROOT/'scripts/build_next6_extension_viewer.py',ROOT/'src/next6_extension/viewer/template.html',ROOT/'src/next6/nodes.py']
    for p in selected:
        target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    if a.stage=='final':
        old=read_json(BASE/'audit/PRELIMINARY_VISUAL_REPRODUCTION.json')
        assert digest(BASE/'audit/sources/PRELIMINARY_VISUAL_REPRODUCTION_SOURCE.py')==old['source_sha256']
        for path,sha in old['input_source_sha256'].items():
            # This file was copied but not executed in the preliminary section;
            # it renders only the four results that were still incomplete.
            if path=='scripts/render_next6_extension_final_figures.py':continue
            assert digest(ROOT/path)==sha,'Previously verified visual input changed: '+path
        for path,sha in old['original_PNG_sha256'].items():assert digest(ROOT/path)==sha
    code="""import sys,runpy,json,os
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'scripts'))
blocked=[]
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0]))
  if p.suffix.lower() in ('.npy','.npz','.parquet'):
   blocked.append(str(p));raise RuntimeError('Raw/array dataset file forbidden during visual reproduction')
sys.addaudithook(guard)
commands=COMMANDS
for name,args in commands:
 sys.argv=[name]+args
 runpy.run_path(str(root/'scripts'/name),run_name='__main__')
assert not blocked
print('VISUAL_REPRODUCTION_NO_RAW_OPEN_PASS')
"""
    commands=[('render_next6_extension_figures.py',[]),('render_next6_extension_memory_figures.py',[]),('render_next6_extension_audit_figures.py',[]),('render_next6_extension_support_figure.py',[]),('render_next6_extension_scale_figure.py',[])] if a.stage=='preliminary' else [('render_next6_extension_final_figures.py',['--section',section]) for section in ('synthetic','bootstrap','latency')]
    runner=dest/'reproduce_visuals.py';runner.write_text(code.replace('COMMANDS',repr(commands)),encoding='utf-8')
    env=os.environ.copy();env['PYTHONPATH']=str(dest/'scripts');env['PYTHONDONTWRITEBYTECODE']='1'
    with activity(a.stage+'_raw_free_visual_reproduction','REPRODUCTION_AUDIT'):
        python=ROOT/'.figure-venv/Scripts/python.exe';r=subprocess.run([str(python),'-B',str(runner)],cwd=dest,env=env,capture_output=True,text=True,encoding='utf-8');(BASE/f'audit/{a.stage.upper()}_VISUAL_REPRODUCTION.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');assert r.returncode==0,r.stderr
        if a.stage=='preliminary':
            demo=BASE/'runtime/demo_venv/Scripts/python.exe';r=subprocess.run([str(demo),'-B',str(dest/'scripts/build_next6_extension_viewer.py')],cwd=dest,env=env,capture_output=True,text=True,encoding='utf-8');assert r.returncode==0,r.stderr
            assert digest(dest/'artifacts/next6_extension/DYNAMIC_STATE_VIEWER.html')==digest(BASE/'DYNAMIC_STATE_VIEWER.html'),'Viewer rebuild differs'
        numbers=(set(range(1,16))|{20}) if a.stage=='preliminary' else {16,17,18,19};comparison=[]
        for p in sorted((BASE/'FIGURES').glob('*.png')):
            if int(p.name[:2]) not in numbers:continue
            q=dest/p.relative_to(ROOT);assert q.is_file()
            comparison.append({'figure':p.name,'original_sha256':digest(p),'reproduced_sha256':digest(q),'PNG_byte_identical':digest(p)==digest(q)})
        assert len(comparison)==len(numbers) and all(r['PNG_byte_identical'] for r in comparison),comparison
    # Only actual plot/viewer dependencies are frozen for the next section.
    volatile={'00_READ_FIRST_NEXT6_EXTENSION_KO.md','RESEARCH_REPORT.md','EXPERIMENT_LEDGER.csv'}
    dependencies=[p for p in selected if p.name not in volatile and not p.name.endswith('_INTEGRITY.json')]
    save(f'audit/{a.stage.upper()}_VISUAL_REPRODUCTION.json',{'status':'PASS','created_utc':utc_now(),'figure_numbers':sorted(numbers),'PNG_comparison':comparison,'HTML_byte_identical':True if a.stage=='preliminary' else 'Covered by unchanged preliminary receipt and final extracted-ZIP check','raw_or_array_dataset_open_attempts':0,'input_source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in dependencies},'original_PNG_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in (BASE/'FIGURES').glob('*.png') if int(p.name[:2]) in numbers},'environment':'Same Windows host, installed Malgun Gothic font and pinned figure venv; not a cross-OS byte identity claim.','source_sha256':digest(Path(__file__))})
    print('Raw-free visual reproduction PASS',a.stage,len(comparison))

if __name__=='__main__':main()
