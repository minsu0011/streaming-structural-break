"""Create a raw-free, hash-listed handoff with an isolated synthetic smoke test."""
from pathlib import Path
import sys,os,shutil,zipfile,subprocess,importlib.metadata,csv,json,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from src.next6.common import BASE,read_json,save,digest,activity,utc_now

def text_files():
    handoff='''# NEXT6 인수인계

`00_READ_FIRST_NEXT6_KO.md`에 결과와 한계를 정리했다. 기존 BL80/D2 canonical은 수정하지 않았다. NEXT6는 ResearchPool 6,400 series의 원래 grouped 5-fold 연구이며 Shadow와 Seal을 사용하지 않았다.

## 파일과 역할

- `src/next6/nodes.py`: H 앞 60%로 기존 엔진을 fit하고 나머지 H를 pseudo-online으로 재생해 16개 노드의 경계와 정규화를 계산한다.
- `src/next6/graph.py`: 자기 과거/공통 제어 대비 sparse cross-lag 예측, factorized 상태 확률, bounded graph/state 요약. 경제적 인과 모델이 아니다.
- `src/next6/model.py`: 고정 LightGBM head. graph 피처는 training head에서도 cross-fit한다.
- `src/next6/runtime.py`: 같은 수학을 실행하는 실제 scalar API. 입력은 H와 현재 scalar뿐이다.
- `artifacts/next6/fold_f/outer/M*/model.joblib`: 기존 champion을 대체하지 않는 fold별 DEV 연구 모델.
- `FIGURES/index.html`: 실제 계산한 PNG/SVG 그림 갤러리. 그림 재생성에는 별도의 matplotlib 환경을 사용했다.

## 실행 순서

아래 명령은 프로젝트 루트에서 실행한다. Windows의 실제 실험 환경은 `.venv\\Scripts\\python.exe -B`였다. 완료된 학습 단계는 재사용한다. 결과 집계·audit·package 명령은 기본적으로 기존 완료 산출물을 덮어쓰지 않으므로 재실행 전 새 output 경로 또는 보존된 완료 receipt를 확인한다.

```powershell
& '.\\.venv\\Scripts\\python.exe' -B scripts/demo_next6_streaming.py --method M3 --fold 0
& '.\\.venv\\Scripts\\python.exe' -B scripts/prepare_next6_nodes.py
& '.\\.venv\\Scripts\\python.exe' -B scripts/run_next6_nested.py --stage inner
& '.\\.venv\\Scripts\\python.exe' -B scripts/run_next6_nested.py --stage outer
& '.\\.venv\\Scripts\\python.exe' -B scripts/score_next6.py
& '.\\.venv\\Scripts\\python.exe' -B scripts/prepare_next6_controls.py
& '.\\.venv\\Scripts\\python.exe' -B scripts/prepare_next6_controls.py --null-graphs
& '.\\.venv\\Scripts\\python.exe' -B scripts/evaluate_next6_controls.py
& '.\\.venv\\Scripts\\python.exe' -B scripts/analyze_next6_stability.py
& '.\\.venv\\Scripts\\python.exe' -B scripts/audit_next6_streaming.py
& '.\\.venv\\Scripts\\python.exe' -B scripts/audit_next6_streaming.py --latency
```

숫자 재현은 기존 split과 hash가 일치하는 원래 ResearchPool cache·reference 피처·모델을 요구한다. 이 자료는 ZIP에 넣지 않았다. `SOURCE_MANIFEST_NEXT6.json`의 외부 의존성 hash를 확인한다. 새 raw scanner나 새 split을 만들며 우회하지 않는다. ZIP의 synthetic scalar demo는 경쟁 데이터 없이 실행할 수 있다. 깨끗한 새 Python 설치까지 검증한 것은 아니며, 복사한 소스와 모델을 기존 고정 가상환경으로 검증했다.

## 설계와 해석상 제한

모든 outer 결과를 열기 전에 5개 fold의 inner 결정을 동결했다. outer 점수를 본 뒤 threshold·weight·node·lag·조합을 바꾸지 않았다. M4가 일부 fold에서만 허용되면 그 결과는 전체 5-fold 후보가 아니다. OWN 대조도 고정 common controls를 유지한다. RANDOM은 같은 edge 수와 target별 parent 수를 쓰지만 관계+전이 24개 요약을 모두 쓰므로 M2/M3 각각의 12개 요약과 head 피처 수가 완전히 같지는 않다. 같은 원 입력·lag·state·memory의 주요 대조는 M1_STRICT이다. 원래 M1에 명시적 상태와 현재 시점이 빠졌음을 감사에서 발견하고, outer 점수 공개 전에 390열 M1_STRICT와 더 엄격한 추가 gate를 고정했다. 학습 recipe는 대조 포함 8개, graph 후보와 기존 inner 결정은 그대로다. M1_INPUT_PARITY_CORRECTION_LOCK.json과 STRICT_INPUT_PAIRED_NEXT6.csv를 확인한다.

동일 raw에서 파생한 두 노드의 확인은 두 독립 원인의 확인이 아니다. 특히 여러 energy/variance 필터의 긴 반응이 outlier를 지속 변화처럼 보이게 할 수 있어 실제 paired 반례를 함께 읽어야 한다. H pseudo-online 비교, 기계적 raw null, 고의로 잘못 연결한 series 경계 trap을 별도 보존했다. state3은 결측/퇴화이며 낮음 state0과 구분한다. 정상/post 확률은 매 시점 함께 계산하고 tau로 어느 모형을 사용할지 고르지 않는다.

paired bootstrap은 series가 단위이며 고정 예측에 조건부다. 전체 CV 재학습 bootstrap이나 과거 DEV 적응성을 보정한 CI가 아니다. 기존 mechanism signature는 사후 관측 지표이고 잠재 DGP 정답이 아니다. 기존 historical regime에는 과거 H transform의 transductive 한계가 있어, 새 H quartile은 별도로 각 outer training에서만 fit했다.

## 오류와 복구

초기 pytest 원본 접근 guard가 ID-only metadata도 차단한 오류는 기존 검토된 compatibility를 재사용해 해결했다. 초기 Numba bool-to-float 문제는 첫 실제 graph fit 전에 수정했다. 이후 float32 범위 밖 scalar/batch 입력 규칙 불일치는 실행 adapter에서 수정했고 과학적 source·학습 파라미터는 바꾸지 않았다. 실패 로그와 수정·재검사 receipt를 보존했다.

## 보존 및 ZIP 범위

ZIP에는 소스·설정·테스트·작은 결과·그림·외부 fold 연구 모델과 비교용 frozen reference weights를 넣었다. raw 관측, credentials, Seal 값, 대용량 feature cache, 전체 OOF/예측 배열, 가상환경, Numba cache는 제외했다. 공식 tau의 전체 per-series 표는 작업 폴더에만 있으며 ZIP에서 제외했다. 그림을 재현하는 소수의 held-out 사례 피처/점수 JSON은 작은 결과 자료다. ZIP 파일별 hash는 `SOURCE_MANIFEST_NEXT6.json`, 별도 smoke·ZIP 무결성 검사는 `HANDOFF_AUDIT_NEXT6.json`이다.
'''
    (BASE/'01_RESEARCH_HANDOFF_NEXT6.md').write_text(handoff,encoding='utf-8')
    nextplan='''# 다음 세션 계획

먼저 NEXT6 read-first, 승격 판정, 같은 입력 M1/OWN/RANDOM 대조, edge 재현 표, paired 이상치 결과를 읽는다. BL80/D2 역할과 원래 split은 유지한다.

이번 outer 성적을 보고 새 weight·threshold·lag를 미세 조정하지 않는다. 재현된 edge라도 계산식 유래인지 먼저 판단하고, 동일 raw 파생 필터가 주는 지속 확인을 독립된 구조 변화 증거로 부르지 않는다. 입력 완비 M1_STRICT 비교와 그 사전 보완 lock을 반드시 확인한다. 원래 M1은 명시적 상태·시점 입력이 없는 ablation이다. 미재현 edge와 M1_STRICT 대비 이득 부족의 원인을 구분한다. 상태 조건부 확률 이득이 탐지 TS-AUC로 이어지지 않으면 확률 보정과 정보 중복을 사전 가설로 따로 설계한다.

추가 연구가 승인되면 가설·비교군·고정 실패 조건·예산을 먼저 기록한다. 기존 ResearchPool의 높은 적응적 재사용 한계를 그대로 명시한다. Shadow/Seal 또는 외부 test를 다음 탐색 데이터로 열지 않는다. 별도의 정당한 검증 계획과 명시적 승인이 생기기 전에는 잠긴 경계를 유지한다.

재현 방법은 handoff 문서와 manifest를 따른다. 저장 모델 replay와 fresh refit을 구분하고, 새로운 프로세스에서 hash·prefix·scalar 경계를 먼저 검사한다. 부분 완료 작업을 완료로 표시하지 않는다.
'''
    (BASE/'NEXT_SESSION_PLAN_NEXT6.md').write_text(nextplan,encoding='utf-8')
    names=list((BASE/'FIGURES').glob('*.png'));names.sort(key=lambda p:(0 if 'fold0_outer_full' in p.name else 1,p.name))
    cards=''.join(f'<article><h2>{html.escape(p.stem)}</h2><a href="{p.name}"><img loading="lazy" src="{p.name}" alt="{html.escape(p.stem)}"></a><p><a href="{p.with_suffix(".svg").name}">확대 가능한 SVG</a></p></article>' for p in names)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NEXT6 실제 결과 그림</title><style>body{max-width:1500px;margin:auto;padding:28px;background:#f3f5f6;color:#22303a;font:17px/1.6 "Malgun Gothic",sans-serif}h1{font-size:34px}article{background:white;padding:22px;margin:24px 0;border-radius:12px}h2{font-size:18px;overflow-wrap:anywhere}img{width:100%;height:auto}a{color:#286781}</style><h1>NEXT6 동적 상태전이망과 검증 그림</h1><p>모두 실제 계산 결과입니다. 정상/post 그래프는 학습 자료의 피처 상태 확률이며 경제적 인과관계를 뜻하지 않습니다. held-out 사례는 고정 기준으로 선택했습니다.</p><p><a href="../00_READ_FIRST_NEXT6_KO.md">결과 보고서</a> · <a href="../MODEL_COMPARISON_NEXT6.csv">모델 비교</a></p>'''+cards+'</html>'
    (BASE/'FIGURES/index.html').write_text(page,encoding='utf-8')

def files_to_ship():
    files=set(ROOT.glob('src/**/*.py'))|set(ROOT.glob('tests/test*.py'))|set(ROOT.glob('tests/conftest.py'))|set(ROOT.glob('scripts/*next6*.py'))
    files.update(ROOT/p for p in ('pyproject.toml','requirements.txt') if (ROOT/p).exists())
    files.update((ROOT/'configs').glob('*.json'))
    files.update(ROOT/p for p in ('00_READ_FIRST_NEXT6_KO.md','01_RESEARCH_HANDOFF_NEXT6.md','NEXT_SESSION_PLAN_NEXT6.md') if (ROOT/p).exists())
    for p in BASE.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(BASE)
        if rel.parts[0] in ('data','synthetic_cache','synthetic_predictions','evaluation_cache','numba_cache','runtime','handoff_staging'):continue
        if p.suffix in ('.npz','.npy','.pyc','.zip'):continue
        if p.name in ('OFFICIAL_TAU_INDEX_NEXT6.csv','MATCHED_CONTROL_IDS.csv','MATCHED_CONTROL_SUPPORT.csv','PSEUDO_AND_NOT_YET_SUPPORT.csv','H_CALIBRATION_DIAGNOSTIC.csv'):continue
        if 'pytest_cache' in rel.parts:continue
        if rel.parts[0].startswith('fold_') and ('outer' not in rel.parts):continue
        if p.suffix.lower() in ('.json','.csv','.md','.txt','.png','.svg','.html','.joblib','.xml','.log'):files.add(p)
    for p in (ROOT/'artifacts/next5/references/fits/PRIMARY/full').glob('*/fold_*/*'):
        if p.name in ('model.joblib','RESULTS.json'):files.add(p)
    return sorted(files)

def main():
    with activity('raw_free_handoff_package','FINAL_AUDIT'):
        text_files()
        for name in ('00_READ_FIRST_NEXT6_KO.md','01_RESEARCH_HANDOFF_NEXT6.md','NEXT_SESSION_PLAN_NEXT6.md'):
            dest=ROOT/name;assert not dest.exists(),'Existing root handoff must be preserved'
            text=(BASE/name).read_text(encoding='utf-8').replace('](FIGURES/','](artifacts/next6/FIGURES/')
            dest.write_text(text,encoding='utf-8')
        required=['00_READ_FIRST_NEXT6_KO.md','01_RESEARCH_HANDOFF_NEXT6.md','NEXT6_PROTOCOL_LOCK.json','DATA_ACCESS_AUDIT_NEXT6.json','FEATURE_LINEAGE.csv','FEATURE_STATE_CATALOG.csv',
            'EVENT_LABEL_CONTRACT.md','EVENT_SUPPORT_TABLE.csv','CORRELATION_ATLAS','LAGGED_RELATION_RESULTS.csv','MECHANICAL_EDGE_AUDIT.csv','EDGE_STABILITY_RESULTS.csv',
            'GRAPH_SELECTION_PROVENANCE.json','STATE_TRANSITION_TABLES','STREAMING_GRAPH_FEATURE_CATALOG.csv','MODEL_COMPARISON_NEXT6.csv','NESTED_CV_NEXT6.csv',
            'PAIRED_UNCERTAINTY_NEXT6.csv','CONTROL_RESULTS_NEXT6.csv','MECHANISM_RESULTS_NEXT6.csv','TIME_AND_BREAK_AGE_NEXT6.csv','COUNTEREXAMPLE_RESULTS_NEXT6.csv',
            'LEAKAGE_AUDIT_NEXT6.json','PREFIX_PARITY_NEXT6.json','LATENCY_NEXT6.json','REPRODUCTION_NEXT6.json','FINAL_AUDIT_NEXT6.json','EXPERIMENT_LEDGER.csv','NEXT_SESSION_PLAN_NEXT6.md']
        assert all((BASE/p).exists() for p in required),'Required handoff output missing'
        stage=BASE/'handoff_staging';assert not stage.exists(),'Preserve previous staging; choose a new explicit path';stage.mkdir()
        selected=files_to_ship();versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','lightgbm','numba','joblib','pandas','pyarrow','pytest','threadpoolctl')}
        manifest={'created_utc':utc_now(),'files_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in selected},'python':sys.version,'packages':versions,
            'external_data_dependencies':{'RP_manifest_sha256':read_json(BASE/'PREFLIGHT.json')['cache_manifest_sha256'],'RP_directory':'data/next5/research_5d54e316ff5bf25d',
                'raw_byte_hashes':read_json(BASE/'PREFLIGHT.json')['raw_sha256'],'split_sha256':read_json(BASE/'PREFLIGHT.json')['split_sha256'],
                'reference_features':'existing hash-validated data/next5/REF_J102 and REF_C205 caches; excluded'},
            'omitted':['raw observations','credentials','Seal contents','large features and OOF','full per-series official tau table','virtual environments','Numba caches'],
            'manifest_self_hash_excluded':True,'reproducibility_scope':'portable fitted scalar demo; full fitting needs unchanged external RP data'}
        save('SOURCE_MANIFEST_NEXT6.json',manifest);selected.append(BASE/'SOURCE_MANIFEST_NEXT6.json')
        for p in selected:
            dest=stage/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        env=os.environ.copy();env['PYTHONPATH']=str(stage);env['NUMBA_CACHE_DIR']=str(stage/'artifacts/next6/numba_cache')
        command=[sys.executable,'-B',str(stage/'scripts/demo_next6_streaming.py'),'--method','M3','--fold','0']
        r=subprocess.run(command,cwd=stage,env=env,capture_output=True,text=True,encoding='utf-8')
        (BASE/'final_audit/HANDOFF_SMOKE.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');assert r.returncode==0,r.stderr
        # Zip only the explicit pre-smoke file list, excluding generated caches.
        archive=BASE/'NEXT6_RESEARCH_HANDOFF.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in selected:z.write(stage/p.relative_to(ROOT),p.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name in z.namelist():
                assert not name.startswith('/') and '..' not in Path(name).parts
                assert Path(name).suffix not in ('.npz','.npy','.pyc') and not name.startswith('data/')
                assert not any(part in ('credentials','secrets','.venv','numba_cache') for part in Path(name).parts)
        save('HANDOFF_AUDIT_NEXT6.json',{'status':'PASS','created_utc':utc_now(),'archive':archive.name,'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size,
            'file_count':len(selected),'raw_and_large_arrays_excluded':True,'isolated_source_copy_synthetic_scalar_demo':'PASS','smoke_command':command,
            'same_fixed_Python_environment':True,'fresh_Python_install_not_claimed':True,'manifest_sha256':digest(BASE/'SOURCE_MANIFEST_NEXT6.json')})
        print('Handoff created:',archive,archive.stat().st_size,flush=True)

if __name__=='__main__':main()
