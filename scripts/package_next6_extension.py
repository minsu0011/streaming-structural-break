"""Build a raw-free handoff, test the extracted ZIP, then copy exactly20 files."""
from pathlib import Path
import sys,os,shutil,zipfile,subprocess,json,re,ctypes,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6_extension.common import *

EXCLUDE_JSON={'FRESH_RP_PHASE_SERIES.json','HANDOFF_AUDIT_NEXT6_EXTENSION.json','DESKTOP_KEY_RESULTS20_MANIFEST.json'}
EXCLUDE_CSV={'H_BLOCK_CALIBRATION.csv','FRESH_RP_SERIES_DIAGNOSTIC.csv','FROZEN_EVIDENCE_NODE_ATTRIBUTION.csv','FINITE_INFLUENCE_STRESS_CASES.csv','CALIBRATION_TRANSPORT_SERIES.csv'}

def selected_files():
    files=set(ROOT.glob('src/**/*.py'))|set(ROOT.glob('configs/*.json'))|set(ROOT.glob('tests/test_next6_extension_*.py'))|set(ROOT.glob('scripts/*next6_extension*.py'))
    files.update([ROOT/'scripts/audit_next6_extension_viewer.mjs',ROOT/'src/next6_extension/viewer/template.html',ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'])
    for p in BASE.iterdir():
        if not p.is_file():continue
        if p.suffix=='.json' and p.name not in EXCLUDE_JSON:files.add(p)
        if p.suffix=='.csv' and p.name not in EXCLUDE_CSV:files.add(p)
        if p.suffix in ('.md','.txt','.html'):files.add(p)
    for name in ('FIGURES','audit','calibration','observable_models','coherent_graphs','fresh_graphs','boundary_sensitivity','bootstrap_signs','graph_bootstrap','coherent_power','coherent_raw_null'):
        directory=BASE/name
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix in ('.png','.svg','.html','.json','.csv','.joblib','.xml','.log','.py'):
                files.add(p)
    # Derived synthetic curves and exactly four preselected RP cases reproduce
    # the displayed figures/viewer. They contain no raw observations or full tau table.
    for case in ('O1','OB16','D2','V2','V4','RETURN_NORMAL'):
        files.add(BASE/f'extended_synthetic/fresh/{case}_EXAMPLES.json')
    files.update((BASE/'head_memory').glob('*/EXAMPLES.json'))
    # The previous raw-free release is preserved intact, separately inspectable.
    files.add(ROOT/'artifacts/next6/NEXT6_RESEARCH_HANDOFF.zip')
    for name in ('MODEL_COMPARISON_NEXT6.csv','STRICT_INPUT_MODEL_COMPARISON_NEXT6.csv','PAIRED_UNCERTAINTY_NEXT6.csv','STRICT_INPUT_PAIRED_NEXT6.csv','PROMOTION_DECISION_NEXT6.json','FINAL_AUDIT_NEXT6.json','HANDOFF_AUDIT_NEXT6.json'):
        p=ROOT/'artifacts/next6'/name
        if p.exists():files.add(p)
    for p in files:
        assert p.is_file() and p.resolve().is_relative_to(ROOT)
        assert p.relative_to(ROOT).parts[0]!='data'
        assert not any(part in {'runtime','numba_cache','__pycache__','.venv','.git'} for part in p.relative_to(ROOT).parts)
        assert p.suffix not in ('.parquet','.npy','.npz','.pyc')
    return sorted(files)

def desktop20(archive):
    buffer=ctypes.create_unicode_buffer(32768);code=ctypes.windll.shell32.SHGetFolderPathW(None,0x10,None,0,buffer);assert code==0
    desktop=Path(buffer.value).resolve();assert desktop.is_dir()
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');dest=desktop/f'StructuralBreak_NEXT6_핵심결과20_{stamp}';assert not dest.exists();dest.mkdir()
    mapping=[
      ('01_READ_FIRST_KO.md',None),('02_DYNAMIC_STATE_VIEWER.html',BASE/'DYNAMIC_STATE_VIEWER.html'),('03_RESEARCH_REPORT.md',BASE/'RESEARCH_REPORT.md'),('04_METHODS_AND_INTERPRETATION.md',BASE/'METHODS_AND_INTERPRETATION.md'),('05_NEXT6_COMPLETE_HANDOFF.zip',archive),
      ('06_DYNAMIC_STATE_NETWORK.png',BASE/'FIGURES/02_OBSERVED_DYNAMIC_STATE_TRANSITION_NETWORK.png'),('07_FRESH_CONFIRMATION_RULES.png',BASE/'FIGURES/01_FRESH_OBSERVATION_CONFIRMATION_RULES.png'),('08_SHOCK_AND_RECOVERY.png',BASE/'FIGURES/04_TRANSIENT_SHOCK_AND_RECOVERY.png'),('09_PERSISTENT_CHANGE_CASES.png',BASE/'FIGURES/05_PERSISTENT_DEPENDENCE_SCALE_UP_SCALE_DOWN.png'),('10_H_REFERENCE_LIMITS.png',BASE/'FIGURES/06_H_REFERENCE_AND_HORIZON_TRANSPORT.png'),
      ('11_LIFETIME_MEMORY_BOTTLENECK.png',BASE/'FIGURES/11_LEGACY_LIFETIME_MAX_MEMORY_BOTTLENECK.png'),('12_GRAPH_BOOTSTRAP_STABILITY.png',BASE/'FIGURES/18_ORIGINAL_GRAPH_GROUP_BOOTSTRAP_STABILITY.png'),('13_SCALE_CONTRACTION_BOTTLENECK.png',BASE/'FIGURES/20_SCALE_CONTRACTION_CONFIRMATION_BOTTLENECK.png'),('14_FINAL_AUDIT.json',BASE/'FINAL_AUDIT_NEXT6_EXTENSION.json'),('15_PAIRED_COUNTEREXAMPLE_COMPARISON.csv',BASE/'FRESH_VS_FROZEN_PAIRED_SYNTHETIC.csv'),
      ('16_STATE_TRANSITION_UNCERTAINTY.csv',BASE/'MACRO_TRANSITION_UNCERTAINTY.csv'),('17_GRAPH_SELECTION_FREQUENCY.csv',BASE/'BOOTSTRAP_EDGE_SELECTION_FREQUENCY.csv'),('18_NEXT_RESEARCH_PLAN.md',BASE/'NEXT_SESSION_PLAN.md'),('19_SCALAR_LATENCY.csv',BASE/'LATENCY_NEXT6_EXTENSION.csv'),('20_ORIGINAL_MODEL_COMPARISON.csv',ROOT/'artifacts/next6/MODEL_COMPARISON_NEXT6.csv')]
    names={str(p.resolve()):name for name,p in mapping if p is not None};entries=[]
    for name,p in mapping:
        if p is None:continue
        target=dest/name;assert p.is_file(),str(p)
        if p.suffix=='.md':
            text=p.read_text(encoding='utf-8')
            def link(match):
                label,url=match.group(1),match.group(2)
                if url.startswith(('https://','http://','#')):return match.group(0)
                path=(p.parent/url).resolve();replacement=names.get(str(path))
                return f'[{label}]({replacement})' if replacement else label+' (전체 ZIP에서 확인)'
            target.write_text(re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text),encoding='utf-8')
        else:shutil.copy2(p,target)
        entries.append({'name':name,'source':str(p),'source_sha256':digest(p),'copied_sha256':digest(target),'markdown_links_adapted_to_flat_folder':p.suffix=='.md'})
    text='# NEXT6 핵심 결과 20개\n\n기존 BL80/D2를 유지합니다. 추가 연구에서 충격·지속 증거·회복을 구분하는 실행 상태망과 시각화를 만들고 병목을 검증했습니다. 분산 감소, 짧은 기준 자료, 긴 관측의 오경보 문제는 남아 있습니다.\n\n먼저 [시간 재생 상태망](02_DYNAMIC_STATE_VIEWER.html)을 열고 [상태망 그림](06_DYNAMIC_STATE_NETWORK.png), [연구 결과](03_RESEARCH_REPORT.md)를 확인하세요.\n\n이 폴더는 정확히 20개 파일의 요약 사본입니다. 원본 프로젝트와 전체 결과를 이동하거나 삭제하지 않았습니다. 전체 소스·그림·모델·재현 설명은 [전체 ZIP](05_NEXT6_COMPLETE_HANDOFF.zip)에 있습니다. ZIP을 풀고 START_HERE.md를 읽으면 됩니다.\n\n## 파일 목록\n\n'+'\n'.join(f'- [{name}]({name})' for name,_ in mapping)+'\n'
    (dest/mapping[0][0]).write_text(text,encoding='utf-8');entries.insert(0,{'name':mapping[0][0],'source':'generated flat-folder guide','copied_sha256':digest(dest/mapping[0][0])})
    assert len(list(dest.iterdir()))==20 and all(p.is_file() for p in dest.iterdir())
    from audit_next6_extension_document_links import check
    desktoplinks=check(sorted(dest.glob('*.md'))+sorted(dest.glob('*.html')),dest)
    save('DESKTOP_KEY_RESULTS20_MANIFEST.json',{'status':'PASS','created_utc':utc_now(),'directory':str(dest),'file_count':20,'copy_only_originals_preserved':True,'local_links_checked':len(desktoplinks),'files':entries})
    return dest

def main():
    final=read_json(BASE/'FINAL_AUDIT_NEXT6_EXTENSION.json');assert final['status']=='PASS'
    assert datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(read_json(BASE/'SESSION.json')['minimum_continue_until_utc'])
    archive=BASE/'NEXT6_EXTENSION_RESEARCH_HANDOFF.zip';assert not archive.exists()
    stage=BASE/'handoff_staging';verify=BASE/'runtime/final_zip_extracted';assert not stage.exists() and not verify.exists();stage.mkdir();verify.mkdir(parents=True)
    files=selected_files();checks={}
    with activity('raw_free_extension_package_and_extracted_demo','HANDOFF_AUDIT'):
        for p in files:
            rel=p.relative_to(ROOT);target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);checks[rel.as_posix()]={'sha256':digest(target),'bytes':target.stat().st_size}
        start=stage/'START_HERE.md';start.write_text('# NEXT6 동적 상태망 인수인계\n\n[먼저 읽기](artifacts/next6_extension/00_READ_FIRST_NEXT6_EXTENSION_KO.md) · [시간 재생 상태망](artifacts/next6_extension/DYNAMIC_STATE_VIEWER.html) · [그림 전체](artifacts/next6_extension/FIGURES/index.html) · [실행 방법](artifacts/next6_extension/REPRODUCTION_GUIDE.md)\n\n기존 BL80/D2를 유지합니다. 이 ZIP은 원자료 없이 저장 상태망과 합성 사례를 실행하는 데 필요한 소스·모델·최소 의존성 명세를 포함합니다. 전체 학습 재현에는 원래 잠긴 ResearchPool과 참조 모델 자료가 별도로 필요합니다. 이전 NEXT6 ZIP은 artifacts/next6 안에 보존했습니다.\n',encoding='utf-8');checks['START_HERE.md']={'sha256':digest(start),'bytes':start.stat().st_size}
        manifest={'created_utc':utc_now(),'files':checks,'excludes':['competition raw observations','whole per-series tau table','large RP feature/prediction arrays','credentials','virtual environments','Numba caches'],'self_hash_policy':'Manifest excludes itself; ZIP hash and extracted-run result are in external HANDOFF_AUDIT_NEXT6_EXTENSION.json.','ledger':'Snapshot taken before the packaging activity closes; not claimed identical to the subsequently appended live ledger.'}
        from audit_next6_extension_document_links import check
        stagedbase=stage/'artifacts/next6_extension';check(sorted(stagedbase.glob('*.md'))+[start,stagedbase/'DYNAMIC_STATE_VIEWER.html',stagedbase/'FIGURES/index.html'],stage)
        mp=stage/'SOURCE_MANIFEST_NEXT6_EXTENSION.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(stage.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(stage).as_posix())
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for member in z.infolist():assert (verify/member.filename).resolve().is_relative_to(verify.resolve()) and not Path(member.filename).is_absolute()
            z.extractall(verify)
        for rel,record in checks.items():assert digest(verify/rel)==record['sha256']
        assert digest(verify/mp.name)==digest(mp)
        python=BASE/'runtime/demo_venv/Scripts/python.exe';assert python.is_file();env=os.environ.copy();env['PYTHONPATH']=str(verify);env['PYTHONDONTWRITEBYTECODE']='1';env['NUMBA_CACHE_DIR']=str(verify/'artifacts/next6_extension/numba_cache')
        rebuilt=subprocess.run([str(python),'-B',str(verify/'scripts/build_next6_extension_viewer.py')],cwd=verify,env=env,capture_output=True,text=True,encoding='utf-8');assert rebuilt.returncode==0,rebuilt.stderr
        assert digest(verify/'artifacts/next6_extension/DYNAMIC_STATE_VIEWER.html')==checks['artifacts/next6_extension/DYNAMIC_STATE_VIEWER.html']['sha256'],'Extracted viewer rebuild changed output'
        result=subprocess.run([str(python),'-B',str(verify/'scripts/verify_next6_extension_portable.py')],cwd=verify,env=env,capture_output=True,text=True,encoding='utf-8');(BASE/'audit/FINAL_EXTRACTED_ZIP_DEMO.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError('Extracted ZIP demo failed: '+result.stderr[-3000:])
        demo=json.loads(result.stdout);assert demo['status']=='PASS' and not demo['raw_or_array_dataset_open_attempts']
        for rel,record in checks.items():assert digest(verify/rel)==record['sha256'],'Demo changed shipped bytes: '+rel
        desktop=desktop20(archive)
    end=utc_now();started=read_json(BASE/'SESSION.json')['started_utc'];elapsed=(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(started)).total_seconds()/3600
    save('HANDOFF_AUDIT_NEXT6_EXTENSION.json',{'status':'PASS','created_utc':end,'archive':str(archive),'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,'shipped_files':len(checks)+1,'manifest_sha256':digest(mp),'ZIP_CRC_and_all_extracted_file_hashes':'PASS','extracted_HTML_rebuild_byte_identical':True,'isolated_extracted_demo':demo,'Desktop_important_files':str(desktop),'Desktop_file_count':20,'started_utc':started,'completed_utc':end,'elapsed_hours_including_packaging':elapsed,'original_files_moved_or_deleted':False})
    print('PASS',archive,desktop,elapsed,flush=True)

if __name__=='__main__':main()
