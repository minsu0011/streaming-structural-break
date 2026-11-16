"""Two permitted reduced diagnostics; scores stay unread until final selection.

Runs the actual authenticated crunch_tools.test with pinned unchanged runner
code and the canonical raw directory. No train/seal targets are loaded by this
inference-only local test. Reduced targets are read once per reserved use.
"""
import argparse,contextlib,io,json,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.streaming import feature_version
from src.utils.artifacts import write_json,utc_now,sha256
from scripts.verify_scorer import load_official,official_score

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['baseline','primary'],required=True);parser.add_argument('--model-directory')
    args=parser.parse_args()
    ledger_path=ROOT/'REDUCED_TEST_USAGE_LEDGER.json';usage=json.loads(ledger_path.read_text(encoding='utf-8'))
    if usage['scored_uses']>=2 or any(e['kind']==args.kind for e in usage['entries']):raise RuntimeError('Reduced diagnostic budget already consumed for this kind')
    if args.kind=='primary':
        from src.validation.adaptive import ResearchBudget
        if ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json').remaining_experiment_seconds()>0:
            raise RuntimeError('The single final reduced diagnostic remains reserved for the original final buffer, even after early DEV artifact preparation')
        frozen=json.loads((ROOT/'DEV_CHAMPION_CANDIDATES.json').read_text(encoding='utf-8'))
        if frozen.get('status')!='FROZEN_FOR_FINAL_DIAGNOSTICS':raise RuntimeError('Primary must be frozen before the final reduced diagnostic')
        if not args.model_directory:raise ValueError('Final model directory required')
        model_directory=Path(args.model_directory).resolve()
        selected=frozen['candidates']['PRIMARY_DEV_CANDIDATE']
        if model_directory!=(ROOT/selected['model_directory']).resolve():raise RuntimeError('Reduced smoke must use the frozen primary artifact directory')
        descriptor=json.loads((model_directory/'model.json').read_text())
        if descriptor.get('status')!='FROZEN_DEV_PRIMARY_REFIT' or descriptor.get('candidate')!=selected['candidate']:raise RuntimeError('Reduced smoke requires the completed frozen primary refit')
        if descriptor.get('seal_rows_fitted')!=0 or descriptor.get('split_sha256')!=frozen['split_sha256']:raise RuntimeError('Primary refit scope does not match the unopened seal contract')
    else:
        model_directory=ROOT/'artifacts/deployment/reduced_baseline'
        model_directory.mkdir(parents=True,exist_ok=True)
        write_json(model_directory/'model.json',{'kind':'baseline','name':'B1_official_ewma','feature_hash':feature_version()})
    prediction_directory=ROOT/'data/processed/reduced_diagnostics'/args.kind
    report={'status':'RUNNING','kind':args.kind,'timestamp':utc_now(),'official_crunch_tools_test_called':False,
        'force_first_train':False,'seal_target_performance_evaluated':False,'submitted':False,
        'canonical_data_directory_override':'data/raw','runner_source':'vendor/official/runner.py',
        'runner_sha256':sha256(ROOT/'vendor/official/runner.py'),'runner_algorithm_modified':False,
        'model_descriptor_sha256':sha256(model_directory/'model.json')}
    started=time.perf_counter();captured=io.StringIO()
    try:
        with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
            import crunch
            import crunch.constants
            import crunch.inline
            from crunch.unstructured import LocalCodeLoader,RunnerModule
            import main as user_module
            crunch.constants.DOT_DATA_DIRECTORY=str(ROOT/'data/raw')
            crunch.inline.DOT_PREDICTION_DIRECTORY=str(prediction_directory)
            crunch_tools=crunch.load_notebook(user_module,model_directory_path=str(model_directory))
            crunch_tools._runner_module=RunnerModule.load(LocalCodeLoader(path=str(ROOT/'vendor/official/runner.py')))
            np.random.seed(20260908)
            report['official_crunch_tools_test_called']=True
            crunch_tools.test(force_first_train=False,raise_abort=True,no_determinism_check=False)
        prediction_path=prediction_directory/'prediction.parquet'
        prediction=pd.read_parquet(prediction_path).sort_index()
        if not prediction.index.is_unique or not np.isfinite(prediction.prediction).all() or not prediction.prediction.between(0,1).all():raise RuntimeError('Prediction contract failed')
        report['determinism_passed']='determinism check: passed' in captured.getvalue()
        if not report['determinism_passed']:raise RuntimeError('Official tester did not report determinism PASS')
        report['rows']=len(prediction);report['series']=int(prediction.index.get_level_values('id').nunique())
        report['prediction_sha256']=sha256(prediction_path);report['elapsed_runner_seconds']=time.perf_counter()-started
        # Reserve before opening labels, so a subsequent failure cannot cause an
        # unrecorded extra target inspection or accidental retry.
        entry={'kind':args.kind,'timestamp':utc_now(),'status':'TARGET_READ_RESERVED','model_descriptor_sha256':report['model_descriptor_sha256'],
            'prediction_sha256':report['prediction_sha256'],'scored_once':False,'score_visible_during_research':False}
        usage['entries'].append(entry);usage['scored_uses']+=1;write_json(ledger_path,usage)
        target=pd.read_parquet(ROOT/'data/raw/y_test.reduced.parquet').sort_index()
        if not prediction.index.equals(target.index):raise RuntimeError('Reduced prediction target indices differ')
        score=official_score(load_official(),prediction,target)
        hidden=ROOT/'artifacts/reduced_private_until_final'/f'{args.kind.upper()}_SCORE.json'
        write_json(hidden,{'kind':args.kind,'ts_auc':score,'rows':len(target),'timestamp':utc_now(),
            'prediction_sha256':report['prediction_sha256'],'target_sha256':sha256(ROOT/'data/raw/y_test.reduced.parquet'),
            'usage_number':usage['scored_uses'],'read_only_after_primary_freeze':True})
        entry.update(status='SCORED_ONCE',scored_once=True,score_artifact=str(hidden.relative_to(ROOT)),score_artifact_sha256=sha256(hidden))
        write_json(ledger_path,usage);report['status']='PASS';report['reduced_scored_uses_total']=usage['scored_uses']
    except Exception as exc:
        report['status']='FAIL';report['exception_type']=type(exc).__name__
        report['error_stack']=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]
        # Never write arbitrary exception messages or authenticated downloader logs.
    write_json(ROOT/'artifacts/real'/f'CRUNCH_TEST_{args.kind.upper()}_REAL.json',report)
    print(json.dumps(report),flush=True)
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
