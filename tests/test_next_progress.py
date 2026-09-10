from concurrent.futures import ThreadPoolExecutor
import json
from src.next.progress import append_exclusive_checkpoint,best_completed_primary


def test_concurrent_checkpoints_never_replace_existing_records(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda number:append_exclusive_checkpoint(tmp_path,{'number':number}),range(40)))
    assert len(set(paths))==40
    assert {json.loads(path.read_text(encoding='utf-8'))['number'] for path in paths}==set(range(40))


def test_global_best_excludes_bug_and_incomplete_screen_score(tmp_path):
    for eid,record in [('good',{'status':'FULL_FAIL','full':{'status':'COMPLETE','mean':.62}}),
                       ('bug',{'status':'BUG','full':{'status':'COMPLETE','mean':.99}}),
                       ('screen',{'status':'SCREEN_FAIL','screen':{'mean':.999}})]:
        directory = tmp_path/'artifacts/next/experiments'/eid
        directory.mkdir(parents=True)
        (directory/'RESULTS.json').write_text(json.dumps(record),encoding='utf-8')
    assert best_completed_primary(tmp_path,.9999)==(.62,'good')
