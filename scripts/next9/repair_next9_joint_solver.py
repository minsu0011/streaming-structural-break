"""Resume the same capped joint objectives, preserving every prior artifact."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now,activity

def main():
    from src.next9.queue import identity_alive
    assert not (BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json').exists()
    queue=read_json(BASE/'runtime/TASK_QUEUE.json');assert all(r['status']=='PAUSED_NUMERICAL_REPAIR' for r in queue)
    for path in (BASE/'runtime/queue').glob('*/attempt_*/START.json'):
        r=read_json(path);assert not identity_alive(r['pid'],r['process_created']),str(path)
    with activity('Preserve capped optimizer outputs and register same-objective bounded numerical continuation','SOLVER_REPAIR','B9_NORMALIZATION_STATE_AVAILABILITY'):
        oldroot=(BASE/'joint_heads').resolve();archive=(BASE/'repairs/solver_cap250_preserved/joint_heads').resolve()
        assert oldroot.is_relative_to(BASE.resolve()) and archive.is_relative_to((BASE/'repairs').resolve())
        assert oldroot.is_dir() and not archive.exists()
        hashes={p.relative_to(oldroot).as_posix():digest(p) for p in oldroot.rglob('*') if p.is_file()}
        assert hashes
        archive.parent.mkdir(parents=True,exist_ok=False)
        oldroot.rename(archive)  # Both resolved absolute targets checked above.
        for name,sha in hashes.items():assert digest(archive/name)==sha
        source=(ROOT/'scripts/next9/fit_next9_joint_heads.py').read_text(encoding='utf-8')
        source=source.replace('JOINT_HEAD_IMPLEMENTATION_LOCK_NEXT9.json','JOINT_HEAD_IMPLEMENTATION_LOCK_V2_NEXT9.json')
        source=source.replace('from src.next9.learning import fit_classifier','from src.next9.learning_converged_v2 import fit_classifier,continue_classifier')
        source=source.replace('                model=fit_classifier(full[:,indices],y,ids,groups,parent)',"""                previous=BASE/'repairs/solver_cap250_preserved/joint_heads'/parent.name/mode
                checkpoint=previous/(method+'.joblib')
                if checkpoint.exists():
                    oldspec=read_json(previous/'SPEC_FROZEN.json')
                    assert digest(checkpoint)==oldspec['files_sha256'][checkpoint.name]
                    model=joblib.load(checkpoint)
                    assert model.training['input_names']==list(names) and model.training['feature_candidates']==candidates
                    assert oldspec['rule_specs']=={str(k):v for k,v in rule_specs.items()}
                    model=continue_classifier(model,full[:,indices],y,ids,groups,parent)
                    model.training['preserved_cap_checkpoint_sha256']=digest(checkpoint)
                else:
                    model=fit_classifier(full[:,indices],y,ids,groups,parent)""")
        new='scripts/next9/fit_next9_joint_heads_v2.py';compile(source,new,'exec');atomic(ROOT/new,source)
        oldlock=BASE/'JOINT_HEAD_IMPLEMENTATION_LOCK_NEXT9.json';lock=read_json(oldlock)
        files=[new,'src/next9/learning_converged_v2.py','scripts/next9/tests/test_solver_continuation.py']
        lock.update(timestamp=utc_now(),supersedes_sha256=digest(oldlock),
            source_sha256={**lock['source_sha256'],**{p:digest(ROOT/p) for p in files}},
            optimizer_repair='Same standardized TRAIN binary logistic objective,C=.1,groupweights andtol1e-6. Preservefirst250iteration cap checkpoint; ifnonconverged continueoncefromitscoefficients foratmost5000additionaliterations. Alreadyconvergedmodelsreusedwithoutrefit. No hyperparameterorqualityselection.',
            additional_fit_calls_recorded=True,old_cap_outputs_preserved=True,
            TRAIN_only_diagnosis_sha256=digest(BASE/'repairs/JOINT_SOLVER_TRAIN_DIAGNOSIS_NEXT9.json'),
            regression_sha256=digest(BASE/'tests/SOLVER_CONTINUATION_TESTS_V2.xml'))
        atomic(BASE/'JOINT_HEAD_IMPLEMENTATION_LOCK_V2_NEXT9.json',lock)
        before=read_json(BASE/'repairs/QUEUE_BEFORE_SOLVER_MAINTENANCE_NEXT9.json')['tasks'];prior={r['id']:r for r in before}
        output=[];changes=[]
        for item in queue:
            original=prior[item['id']]
            if original['script']!='scripts/next9/fit_next9_joint_heads.py':output.append(original);continue
            replacement={**original,'id':original['id']+'_SOLVER_V2','script':new,'source_sha256':digest(ROOT/new),
                'gates':[g.replace('JOINT_HEAD_IMPLEMENTATION_LOCK_NEXT9.json','JOINT_HEAD_IMPLEMENTATION_LOCK_V2_NEXT9.json') for g in original['gates']]}
            output.append(replacement);changes.append(dict(original=original,replacement=replacement))
        assert len(changes)==20
        atomic(BASE/'repairs/JOINT_SOLVER_CONTINUATION_REPAIR_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
            classification='Numerical convergence repair before outer evaluation; unchanged statisticalobjective',
            preserved_directory=archive.relative_to(BASE).as_posix(),preserved_files_sha256=hashes,
            old_current_output_directory_moved_only_after_all_owned_jobs_stopped=True,
            no_old_NEXT8_or_canonical_file_changed=True,existing_converged_models_reused=True,
            no_new_scientific_candidate=True,optimizer_cap_single5000continuation=True,
            training_gradient_diagnosis_sha256=digest(BASE/'repairs/JOINT_SOLVER_TRAIN_DIAGNOSIS_NEXT9.json'),
            replacements=changes))
        atomic(BASE/'registrations/JOINT_HEAD_SOLVER_V2_TASKS.json',dict(timestamp=utc_now(),tasks=[r['replacement'] for r in changes]))
        atomic(BASE/'runtime/TASK_QUEUE.json',output,immutable=False)

if __name__=='__main__':main()
