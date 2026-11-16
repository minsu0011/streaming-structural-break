"""Small predictive state summaries with train-only support and abstention.

These factorized conditional tables are not a causal graph or a full Markov
description of the raw process. States summarize H-calibrated feature levels.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import joblib
from src.next8.common import atomic,read_json,digest,utc_now,activity
from src.next8.features import DETAILED,HORIZONS,UNKNOWN
from src.next8.learning import Design,one_hot
from src.next8.cache import immutable_npz
from src.next8.screening import sampled_rows
from src.next8.movement import movement_target,cross_eligible

MIN_GROUPS=64
PSEUDOCOUNT=.5
RATIO_BOUND=4.


def transition_inputs(cache,rows,j,parents):
    own=cache.states(rows,[j])[:,0]
    other=cache.states(rows,parents,previous=True) if parents else np.empty((len(rows),0),dtype=np.int8)
    return own,other


def transition_tabular_design(cache,rows,j,parents,continuous):
    own,other=transition_inputs(cache,rows,j,parents)
    values=[continuous.values,one_hot(own)]
    names=list(continuous.names)+[f'F{j:03d}_current_H_state_{s}' for s in range(4)]
    for k,p in enumerate(parents):
        values.append(one_hot(other[:,k]))
        names.extend(f'F{p:03d}_previous_H_state_{s}' for s in range(4))
    return Design(np.column_stack(values).astype(np.float32),tuple(names),continuous.ids)


def _group_counts(ids,own,other=None):
    support=np.zeros((3,) if other is None else (3,3),dtype=np.int32)
    for a in range(3):
        if other is None:support[a]=len(np.unique(ids[own==a]))
        else:
            for b in range(3):support[a,b]=len(np.unique(ids[(own==a)&(other==b)]))
    return support


@dataclass
class FactorizedTransition:
    prior: np.ndarray
    own_table: np.ndarray
    own_support: np.ndarray
    pair_tables: list
    pair_support: list
    parents: tuple
    training: dict

    @classmethod
    def fit(cls,own,other,target,ids,parents,scope):
        own=np.asarray(own);other=np.asarray(other);target=np.asarray(target);ids=np.asarray(ids)
        scope.admit_training(np.unique(ids))
        if other.shape!=(len(own),len(parents)) or len(target)!=len(own) or len(ids)!=len(own):
            raise ValueError('Transition input shape mismatch')
        if any(np.any((a<0)|(a>UNKNOWN)) for a in (own,other,target)):
            raise ValueError('Invalid state coding')
        keep=(own<UNKNOWN)&(target<UNKNOWN)
        own=own[keep];other=other[keep];target=target[keep];ids=ids[keep]
        # Exactly one unit of total weight per admitted training series.
        if len(ids):
            _,code,count=np.unique(ids,return_inverse=True,return_counts=True)
            weights=1./count[code]
        else:weights=np.empty(0)
        prior=np.bincount(target,weights=weights,minlength=3)+PSEUDOCOUNT
        prior/=prior.sum()
        own_table=np.zeros((3,3))
        for a in range(3):
            mask=own==a
            counts=np.bincount(target[mask],weights=weights[mask],minlength=3)+PSEUDOCOUNT
            own_table[a]=counts/counts.sum()
        support=_group_counts(ids,own)
        tables=[];supports=[]
        for k in range(len(parents)):
            table=np.zeros((3,3,3))
            for a in range(3):
                for b in range(3):
                    mask=(own==a)&(other[:,k]==b)
                    counts=np.bincount(target[mask],weights=weights[mask],minlength=3)+PSEUDOCOUNT
                    table[a,b]=counts/counts.sum()
            tables.append(table);supports.append(_group_counts(ids,own,other[:,k]))
        return cls(prior,own_table,support,tables,supports,tuple(parents),{
            **scope.provenance(),'method':'factorized_predictive_state_summary',
            'minimum_distinct_training_series_per_context':MIN_GROUPS,'pseudocount':PSEUDOCOUNT,
            'per_parent_likelihood_ratio_bound':RATIO_BOUND,'training_rows':len(ids),
            'breakOccurred_used':False,'causal_interpretation':False,
            'is_full_markov_state':False,'unknown_or_sparse_parent':'Ignore that factor and preserve own-history distribution'})

    def predict(self,own,other,*,use_parents=True):
        own=np.asarray(own);other=np.asarray(other)
        if other.shape!=(len(own),len(self.parents)):raise ValueError('Changed transition parent input schema')
        out=np.zeros((len(own),4));active=np.zeros(len(own),dtype=np.int8)
        supported=np.zeros(len(own),dtype=bool)
        out[own==UNKNOWN,UNKNOWN]=1.
        for a in range(3):
            at=np.flatnonzero(own==a)
            if not len(at):continue
            base=self.own_table[a] if self.own_support[a]>=MIN_GROUPS else self.prior
            logits=np.tile(np.log(base),(len(at),1))
            supported[at]=self.own_support[a]>=MIN_GROUPS
            if use_parents and self.own_support[a]>=MIN_GROUPS:
                for k in range(len(self.parents)):
                    for b in range(3):
                        mask=other[at,k]==b
                        if self.pair_support[k][a,b]<MIN_GROUPS:continue
                        ratio=np.clip(self.pair_tables[k][a,b]/base,1/RATIO_BOUND,RATIO_BOUND)
                        logits[mask]+=np.log(ratio);active[at[mask]]+=1
            probability=np.exp(logits-logits.max(axis=1,keepdims=True))
            out[at,:3]=probability/probability.sum(axis=1,keepdims=True)
        return out.astype(np.float32),{'active_parent_factors':active,'own_context_supported':supported}


def random_parent_map(graph,horizon):
    """Fixed edge swaps preserve in/out degree separately at each horizon."""
    from src.next8.forecasting import parents_at
    original=[(p,j) for j in DETAILED for p in parents_at(graph,j,horizon)]
    edges=list(original);rng=np.random.default_rng(8101604+int(horizon));accepted=0
    if len(edges)>=2:
        for _ in range(1000):
            i,k=map(int,rng.choice(len(edges),2,replace=False))
            a,j=edges[i];b,l=edges[k]
            if a==b or j==l:continue
            new_a=(b,j);new_b=(a,l)
            if not cross_eligible(b,j) or not cross_eligible(a,l):continue
            other=set(edges)-{edges[i],edges[k]}
            if new_a in other or new_b in other:continue
            edges[i]=new_a;edges[k]=new_b;accepted+=1
    nodes=sorted(set(v for e in original for v in e))
    degree=lambda es:{str(n):[sum(a==n for a,b in es),sum(b==n for a,b in es)] for n in nodes}
    assert len(edges)==len(original) and degree(edges)==degree(original)
    return {j:sorted(p for p,k in edges if k==j) for j in DETAILED},{
        'horizon':horizon,'original_edges':[list(e) for e in sorted(original)],
        'random_edges':[list(e) for e in sorted(edges)],'degrees_out_in':degree(edges),
        'accepted_swaps':accepted,'fixed_seed':8101604+horizon,'same_edge_count_and_degrees':True,
        'identical_graph_no_informative_random_contrast':set(edges)==set(original),
        'empty_or_unswappable_graph_not_filled_with_extra_edges':True}


def fit_transition_comparators(cache,scope,graph,directory,*,early=False):
    from src.next8.forecasting import parents_at,validate_graph_scope
    validate_graph_scope(graph,scope)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    cutoff=199 if early else None
    tr_all=sampled_rows(cache,scope.training_ids,cutoff)
    va=sampled_rows(cache,scope.validation_ids)
    if early:va=va[cache.meta['time_online'][va]>=400]
    scope.admit_training(np.unique(cache.meta['dataset_id'][tr_all]))
    scope.admit_validation(np.unique(cache.meta['dataset_id'][va]))
    random_maps={};random_receipts=[]
    for h in HORIZONS:
        mapping,receipt=random_parent_map(graph,h);random_maps[h]=mapping;random_receipts.append(receipt)
    if (directory/'RANDOM_GRAPH.json').exists():assert read_json(directory/'RANDOM_GRAPH.json')==random_receipts
    else:atomic(directory/'RANDOM_GRAPH.json',random_receipts)
    files={}
    for j in DETAILED:
        for h in HORIZONS:
            p=directory/f'F{j:03d}_h{h}';p.mkdir(parents=True,exist_ok=True)
            if (p/'DONE.json').exists():
                done=read_json(p/'DONE.json');assert done['scope']==scope.provenance()
                for name,sha in done['files_sha256'].items():assert digest(p/name)==sha
                files[p.name+'/DONE.json']=digest(p/'DONE.json');continue
            if any(p.iterdir()):raise RuntimeError('Partial transition output preserved for explicit repair')
            with activity(f'Fit factorized transitions {scope.name} F{j:03d} h{h}','TRANSITION','B2_STATE_OBSERVABILITY'):
                tr=cache.forecast_rows(tr_all,h,cutoff)
                target=movement_target(cache,tr,j,h)['state'];ids=cache.meta['dataset_id'][tr]
                parents=parents_at(graph,j,h);random_parents=random_maps[h][j]
                own,other=transition_inputs(cache,tr,j,parents)
                model=FactorizedTransition.fit(own,other,target,ids,parents,scope)
                vown,vother=transition_inputs(cache,va,j,parents)
                t1,s1=model.predict(vown,vother,use_parents=False)
                t3,s3=model.predict(vown,vother)
                joblib.dump(model,p/'T3.joblib')
                if sorted(random_parents)==sorted(parents):
                    t4=t3;s4=s3;random_alias=True
                else:
                    rown,rother=transition_inputs(cache,tr,j,random_parents)
                    random_model=FactorizedTransition.fit(rown,rother,target,ids,random_parents,scope)
                    rva,rvo=transition_inputs(cache,va,j,random_parents)
                    t4,s4=random_model.predict(rva,rvo)
                    joblib.dump(random_model,p/'T4.joblib');random_alias=False
                immutable_npz(p/'PREDICTIONS.npz',global_row=va,dataset_id=cache.meta['dataset_id'][va],
                    T1_state=t1,T3_state=t3,T4_state=t4,T3_active_factors=s3['active_parent_factors'],
                    T4_active_factors=s4['active_parent_factors'],own_supported=s1['own_context_supported'])
                saved={f.name:digest(f) for f in p.iterdir() if f.is_file()}
                atomic(p/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
                    'feature':j,'horizon':h,'parents':parents,'random_parents':random_parents,
                    'T4_alias_T3_identical_parents':random_alias,'training':model.training,
                    'files_sha256':saved,'outer_score_released':False,
                    'T0_reference':'Matched D0_state_persistence and D0_state_majority from full comparator',
                    'T2_reference':'Matched ordinary depth3 classifier; exact own/parent state categories also supplied',
                    'validation_future_targets_used':False})
                files[p.name+'/DONE.json']=digest(p/'DONE.json')
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'files_sha256':files,'random_graph_sha256':digest(directory/'RANDOM_GRAPH.json'),
        'outer_score_released':False,'transition_claim':'Predictive low-dimensional summary, not a full Markov or causal graph'})
