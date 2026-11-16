"""Hash-bound fitted model loading, usable in a raw-free portable archive."""
import numpy as np
import joblib
from src.next9.common import BASE,ROOT,read_json,digest
from src.next9.streaming import FrozenHeads
from src.next9.runtime_laws import RuntimeSelfLaw,RuntimeCrossLaw


def load_heads(fold,mode):
    scope=f'outer_{fold}';head=BASE/'joint_heads'/scope/mode;cross=BASE/'cross_and_tabular'/scope/mode;meta=BASE/'meta_correction'/scope/mode
    h,c,m=[read_json(p/'DONE.json') for p in [head,cross,meta]]
    for directory,done in [(head,h),(cross,c)]:
        # Prediction arrays are deliberately absent from a portable payload.
        for name,sha in done['files_sha256'].items():
            if not name.endswith('.npz'):assert digest(directory/name)==sha,name
    spec=read_json(head/'SPEC_FROZEN.json');cs=read_json(cross/'SPEC_FROZEN.json')
    assert h['scope']==c['scope']==m['scope']==spec['scope']
    models={name:joblib.load(head/(name+'.joblib')) for name in sorted(set(spec['aliases'].values()))}
    laws={}
    for j in spec['law_features']:
        d=BASE/'self_transitions'/scope/mode/f'F{j:03d}';done=read_json(d/'DONE.json')
        assert digest(d/'LAW.joblib')==done['law_sha256'];laws[j]=RuntimeSelfLaw(joblib.load(d/'LAW.joblib'))
    cross_laws={}
    for j in sorted(set(e['destination'] for e in cs['provisional_edges'])):
        d=cross/f'FINAL_DESTINATION_{j}'
        assert digest(d/'CROSS_LAW.joblib')==cs['final_forecast_files_sha256'][str(j)]['CROSS_LAW.joblib']
        cross_laws[j]=RuntimeCrossLaw(joblib.load(d/'CROSS_LAW.joblib'))
    correction=None
    if m['eligible']:
        assert digest(meta/'CORRECTION.joblib')==m['correction_sha256'];correction=joblib.load(meta/'CORRECTION.joblib')
    return FrozenHeads(spec,models,laws,cross_laws,None if c['M6_alias']=='M5' else joblib.load(cross/'M6.joblib'),
        joblib.load(cross/'TABULAR_SAME_INPUT.joblib'),correction)


def reference_paths(fold,mode):
    if mode=='NORMAL':return [ROOT/f'artifacts/next5/references/fits/PRIMARY/full/{kind}/fold_{fold}/model.joblib' for kind in ['J102','C205']]
    old=ROOT/read_json(BASE/'SESSION_NEXT9.json')['previous_NEXT8_base']
    return [old/'early_baseline'/f'outer_{fold}_EARLY_T199'/kind/'model.joblib' for kind in ['J102','C205']]


def load_baseline(fold,mode):
    from src.next5.model import FeatureModel
    from src.next5.runtime import SharedResearchBaseline
    # Existing NEXT8 portable manifest hash-binds these exact excluded fits.
    old=ROOT/read_json(BASE/'SESSION_NEXT9.json')['previous_NEXT8_base']
    manifest=read_json(old/'PORTABLE_RUNTIME_MANIFEST_NEXT8.json')
    paths=reference_paths(fold,mode)
    for p in paths:assert digest(p)==manifest['files_sha256'][p.relative_to(ROOT).as_posix()]
    models=[FeatureModel.load(p) for p in paths]
    scope=read_json(BASE/'joint_heads'/f'outer_{fold}'/mode/'SPEC_FROZEN.json')['scope']
    for model in models:assert model.training['training_ids_sha256']==scope['training_ids_sha256']
    return SharedResearchBaseline(*models)
