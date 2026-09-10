"""Deterministic inference-only model encoding, independent of pickle aliasing.

Encode frozen predictive arrays and feature contracts as canonical finite JSON
inside a length/hash checked zlib envelope. Omit the unused training estimator.
No original model file is altered. Loaded bundles retain the audited classes
and compact-tree prediction code with estimator=None.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import zlib
import numpy as np
from src.models.compact_trees import CompactTrees
from src.next.engine import EngineConfig
from src.next.extensions import ExtendedBundle
from src.next.registered_extensions import RegisteredExtendedBundle
from src.next.late_blend import validate_next_component
from src.next2.io import Next2Bundle,register,registration_hash
from src.next2.blend import FixedBlendBundle

MAGIC=b'SB_NEXT2_MODEL_V1\n'
MAX_PAYLOAD=8_000_000
TREE_FIELDS=('roots','features','thresholds','left','right','values','missing_left')


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def record(model):
    if isinstance(model,FixedBlendBundle):
        model.validate()
        return {'class':'FixedBlendBundle','weight':model.weight,'implementation_sha256':model.implementation_sha256,
            'primary':record(model.primary),'complement':record(model.complement)}
    if type(model) not in (ExtendedBundle,RegisteredExtendedBundle,Next2Bundle):
        raise ValueError('Only the reviewed frozen binary bundle classes can be encoded')
    if isinstance(model,Next2Bundle):
        register(model.extension)
        if model.next2_registration_sha256!=registration_hash():
            raise RuntimeError('NEXT2 registration source changed')
    validate_next_component(model)
    result={'class':type(model).__name__,'configuration':asdict(model.config),'names':list(model.names),
        'columns':np.asarray(model.columns,dtype=np.int32).tolist(),'learner':model.learner,
        'engine_hash':model.engine_hash,'implementation_hash':model.implementation_hash,'old_arch_hash':model.old_arch_hash,
        'extension':model.extension,'extension_implementation_hash':model.extension_implementation_hash,
        'tree':{key:getattr(model.compact,key).tolist() for key in TREE_FIELDS}}
    result['tree']['bias']=float(model.compact.bias)
    if isinstance(model,RegisteredExtendedBundle):
        result.update(registration=model.registration,registration_source_sha256=model.registration_source_sha256)
    elif isinstance(model,Next2Bundle):
        result['next2_registration_sha256']=model.next2_registration_sha256
    return result


def validate_tree(tree,n_columns):
    size=len(tree.features)
    if not 0<size<=100_000 or not 0<len(tree.roots)<=1000:
        raise ValueError('Unsupported bounded forest size')
    if any(getattr(tree,k).shape!=(size,) for k in TREE_FIELDS[1:]):
        raise ValueError('Tree array dimensions differ')
    if tree.roots.ndim!=1 or np.any((tree.roots<0)|(tree.roots>=size)):
        raise ValueError('Tree roots outside node arrays')
    if np.any((tree.features < -1)|(tree.features>=n_columns)):
        raise ValueError('Tree feature index outside selected columns')
    if not np.isfinite(tree.thresholds).all() or not np.isfinite(tree.values).all() or not np.isfinite(tree.bias):
        raise ValueError('Finite numeric tree parameters required')
    interior=tree.features>=0
    index=np.arange(size)
    # Audited DFS exports always put both children after their parent. This
    # also excludes cyclic traversal before any compiled inference is called.
    for child in (tree.left,tree.right):
        if np.any((child[interior]<=index[interior])|(child[interior]>=size)):
            raise ValueError('Tree child order or bounds invalid')


def restore(item):
    kind=item['class']
    if kind=='FixedBlendBundle':
        model=FixedBlendBundle(restore(item['primary']),restore(item['complement']),float(item['weight']),item['implementation_sha256'])
        model.validate()
        return model
    classes={'ExtendedBundle':ExtendedBundle,'RegisteredExtendedBundle':RegisteredExtendedBundle,'Next2Bundle':Next2Bundle}
    if kind not in classes:
        raise ValueError('Unknown canonical model class')
    names=tuple(item['names'])
    if not names or len(names)>512 or not all(isinstance(n,str) for n in names) or len(set(names))!=len(names):
        raise ValueError('Bounded unique named features required')
    columns=np.asarray(item['columns'],dtype=np.int32)
    if columns.ndim!=1 or not len(columns) or np.any((columns<0)|(columns>=len(names))) or len(np.unique(columns))!=len(columns):
        raise ValueError('Selected column indices invalid')
    values=item['tree']
    tree=CompactTrees(np.asarray(values['roots'],dtype=np.int32),np.asarray(values['features'],dtype=np.int32),
        np.asarray(values['thresholds'],dtype=np.float64),np.asarray(values['left'],dtype=np.int32),
        np.asarray(values['right'],dtype=np.int32),np.asarray(values['values'],dtype=np.float64),
        np.asarray(values['missing_left'],dtype=np.bool_),float(values['bias']))
    validate_tree(tree,len(columns))
    kwargs={'config':EngineConfig(**item['configuration']),'names':names,'columns':columns,'compact':tree,
        'estimator':None,'learner':item['learner'],'engine_hash':item['engine_hash'],
        'implementation_hash':item['implementation_hash'],'old_arch_hash':item['old_arch_hash'],
        'extension':item['extension'],'extension_implementation_hash':item['extension_implementation_hash']}
    if kind=='RegisteredExtendedBundle':
        kwargs.update(registration=item['registration'],registration_source_sha256=item['registration_source_sha256'])
    elif kind=='Next2Bundle':
        register(item['extension'])
        if item['next2_registration_sha256']!=registration_hash():
            raise RuntimeError('Canonical NEXT2 source contract changed')
        kwargs['next2_registration_sha256']=item['next2_registration_sha256']
    model=classes[kind](**kwargs)
    validate_next_component(model)
    return model


def dumps(model):
    payload=json.dumps({'format':'NEXT2_CANONICAL_V1','model':record(model)},sort_keys=True,
        separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    if len(payload)>MAX_PAYLOAD:
        raise ValueError('Canonical model exceeds bounded payload')
    return MAGIC+len(payload).to_bytes(8,'little')+hashlib.sha256(payload).digest()+zlib.compress(payload,9)


def loads(content):
    if not isinstance(content,bytes) or len(content)>MAX_PAYLOAD or not content.startswith(MAGIC):
        raise ValueError('Canonical model envelope invalid')
    offset=len(MAGIC)
    length=int.from_bytes(content[offset:offset+8],'little')
    digest=content[offset+8:offset+40]
    if not 0<length<=MAX_PAYLOAD or len(digest)!=32:
        raise ValueError('Canonical payload length invalid')
    decoder=zlib.decompressobj()
    payload=decoder.decompress(content[offset+40:],length+1)
    if len(payload)!=length or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or hashlib.sha256(payload).digest()!=digest:
        raise ValueError('Canonical model checksum or decompression boundary invalid')
    item=json.loads(payload)
    if item.get('format')!='NEXT2_CANONICAL_V1':
        raise ValueError('Canonical model version invalid')
    model=restore(item['model'])
    if dumps(model)!=content:
        raise ValueError('Model envelope is not in the canonical encoding')
    return model


def save(model,path):
    path=Path(path)
    content=dumps(model)
    if path.exists():
        if path.read_bytes()!=content:
            raise RuntimeError('Refusing to overwrite different canonical model bytes')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content)


def load(path):
    return loads(Path(path).read_bytes())
