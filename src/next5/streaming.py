"""Read-only current-observation API for hash-bound fitted research artifacts."""
import sys
sys.dont_write_bytecode=True
import hashlib
from pathlib import Path
import re
import numpy as np
from src.next5.artifacts import ROOT,BASE,read_json,digest
from src.next5.guard import ResearchGuard
from src.next5.synthetic import load_reference,load_candidate
from src.next5.runtime import PreparedComposite,SharedResearchBaseline,source_hash as runtime_hash
from src.next.readonly_jit import disable_loaded_disk_caches,source_hash as readonly_hash


def source_hash():return hashlib.sha256(Path(__file__).read_bytes()+runtime_hash().encode()+readonly_hash().encode()).hexdigest()


def load_prepared(descriptor_path):
    descriptor_path=Path(descriptor_path);descriptor=read_json(descriptor_path)
    if descriptor.get('kind')!='NEXT5_READONLY_FITTED_RESEARCH_COMPOSITE':raise RuntimeError('Explicit research-only descriptor required')
    if descriptor['streaming_source_sha256']!=source_hash():raise RuntimeError('Scalar execution source changed')
    cid=descriptor['candidate'];scope=descriptor['scope'];fold=descriptor['fold']
    if scope not in ('full','screen') or type(fold) is not int or fold not in (range(5) if scope=='full' else (0,4)):
        raise ValueError('Only previously fitted registered research folds allowed')
    if not isinstance(cid,str) or not re.fullmatch(r'[A-Za-z0-9_]+',cid):raise ValueError('Fixed candidate identifier required')
    guard=ResearchGuard();models,reference_proof=load_reference(scope,fold)
    expert,registration,candidate_proof=load_candidate(cid,scope,fold)
    if candidate_proof!=descriptor['candidate_proof'] or reference_proof!=descriptor['reference_proof']:
        raise RuntimeError('Fitted research artifact identity changed')
    if registration['combination']!=descriptor['combination']:raise RuntimeError('Frozen static combination changed')
    guard.check_state();disable_loaded_disk_caches()
    return PreparedComposite(models['J102'],models['C205'],expert,registration['combination'])


def infer(datasets,descriptor_path):
    prepared=load_prepared(descriptor_path);index=np.arange(256,dtype=np.float64)
    historical=(np.sin(index*.37)+.2*np.cos(index*.11)).astype(np.float32)
    prepared.new_state(historical).predict_one(0.)
    yield None
    for historical,online in datasets:
        state=prepared.new_state(historical)
        for point in online:yield state.predict_one(point)
