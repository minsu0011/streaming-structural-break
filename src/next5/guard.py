"""Research-only default admission before prospective Shadow confirmation."""
import numbers
import numpy as np
from src.next.guard import SealGuard, SealAccessError
from src.next5.artifacts import ROOT,BASE,read_json,digest


class ShadowAccessError(SealAccessError):
    pass


class ResearchSplit:
    def __init__(self,root):
        self.root=root;self.lock=read_json(root/'NEXT5_SPLIT_LOCK.json')
        if self.lock['status']!='FROZEN_NEXT5_RESEARCH_SPLITS':raise ShadowAccessError('Frozen ResearchPool split required')
        self.digest=digest(root/'NEXT5_SPLIT_LOCK.json')
        self.groups={int(i):int(g) for i,g in self.lock['groups'].items()}
        self.mapping={int(i):('NEXT5_RESEARCH_POOL',int(f)) for i,f in self.lock['mappings']['PRIMARY'].items()}

    def assignment(self,sid):
        if isinstance(sid,bool) or not isinstance(sid,numbers.Integral) or int(sid) not in self.mapping:
            raise ShadowAccessError('Only locked ResearchPool integer IDs have research folds')
        return self.mapping[int(sid)]


class ResearchGuard:
    """Application-level guard; no claim of an operating-system sandbox."""
    def __init__(self,root=ROOT):
        from pathlib import Path
        self.root=Path(root);self.base=SealGuard(self.root)
        self.lock=read_json(self.root/'NEXT5_SHADOW_LOCK.json');self.shadow_lock_sha=digest(self.root/'NEXT5_SHADOW_LOCK.json')
        if self.lock['status']!='FROZEN_NEXT5_PROSPECTIVE_SHADOW':raise ShadowAccessError('Prospective Shadow lock required')
        if self.lock['original_assignments_sha256']!=self.base.split.digest or self.lock['seal_lock_sha256']!=self.base.lock_sha:
            raise ShadowAccessError('Original split/seal boundary changed')
        members={}
        for key in ('research','shadow'):
            path=self.root/self.lock[key+'_ids_file']
            if digest(path)!=self.lock[key+'_ids_sha256']:raise ShadowAccessError('Prospective membership bytes changed')
            members[key]=frozenset(int(s) for s in path.read_text(encoding='ascii').splitlines())
        self.research_ids=members['research'];self.shadow_ids=members['shadow'];self.dev_ids=self.research_ids
        self.seal_ids=self.base.seal_ids;self.lock_sha=self.base.lock_sha;self.ledger_sha=self.base.ledger_sha
        if self.research_ids&self.shadow_ids or self.research_ids|self.shadow_ids!=self.base.dev_ids:
            raise ShadowAccessError('Research/Shadow partition integrity failure')
        if {self.base.split.groups[i] for i in self.research_ids}&{self.base.split.groups[i] for i in self.shadow_ids}:
            raise ShadowAccessError('Historical duplicate group crosses prospective boundary')
        self.split=ResearchSplit(self.root)
        if self.split.lock['shadow_lock_sha256']!=self.shadow_lock_sha or set(self.split.mapping)!=self.research_ids:
            raise ShadowAccessError('Research split and Shadow lock disagree')

    def check_state(self):
        self.base.check_state()
        if digest(self.root/'NEXT5_SHADOW_LOCK.json')!=self.shadow_lock_sha or digest(self.root/'NEXT5_SPLIT_LOCK.json')!=self.split.digest:
            raise ShadowAccessError('Prospective lock changed during operation')
        for key in ('research','shadow'):
            if digest(self.root/self.lock[key+'_ids_file'])!=self.lock[key+'_ids_sha256']:
                raise ShadowAccessError('Prospective membership changed during operation')

    def admit(self,ids=None,*,purpose='NEXT5 research access'):
        values=self.research_ids if ids is None else list(ids)
        if any(isinstance(i,bool) or not isinstance(i,numbers.Integral) for i in values):
            raise ShadowAccessError('Integral ResearchPool IDs required')
        requested=frozenset(int(i) for i in values)
        if not requested or not requested.issubset(self.research_ids):
            raise ShadowAccessError(purpose+': Shadow, LOCAL SEAL and unknown IDs are forbidden')
        self.check_state();return sorted(requested)

    def partition(self,train_ids,validation_ids):
        train=set(self.admit(train_ids,purpose='training'));valid=set(self.admit(validation_ids,purpose='validation'))
        if train&valid or {self.split.groups[i] for i in train}&{self.split.groups[i] for i in valid}:
            raise ShadowAccessError('Training/validation historical groups overlap')
        return train,valid

    def scanner(self,filename,ids=None,*,columns=None,batch_size=262144):
        admitted=self.admit(ids,purpose='read '+filename)
        return self.base.scanner(filename,admitted,columns=columns,batch_size=batch_size)

    def labels(self,ids=None):
        admitted=self.admit(ids,purpose='target access')
        table=self.scanner('y_train_index.parquet',admitted).to_table().to_pandas()
        if 'id' in table.columns:table=table.set_index('id')
        if not table.index.is_unique or set(map(int,table.index))!=set(admitted):
            raise ShadowAccessError('ResearchPool label coverage mismatch')
        return table

    def record_rows(self,ids,*,purpose):
        ids=np.asarray(ids);self.admit(np.unique(ids),purpose=purpose)
        return {'purpose':purpose,'rows':len(ids),'series':len(np.unique(ids)),'shadow_rows':0,'seal_rows':0,
                'research_split_sha256':self.split.digest,'shadow_lock_sha256':self.shadow_lock_sha}
