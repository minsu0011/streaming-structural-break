import inspect
import numpy as np
from src.next6.graph import Graph,GraphState,discover,design
from src.next6.nodes import NodeState,temporal_replay

def fixture_graph():
    coef=np.zeros((2,16,82));coef[:,:,81]=.1;coef[1,:,81]=.5
    tables=np.full((2,16,27,3),1/3);tables[1,:,:,2]=.5;tables[1,:,:,0:2]=.25
    parents=[[] for _ in range(16)];parents[1]=[(0,1)];parents[2]=[(0,0)]
    return Graph(parents,[],coef,coef.copy(),np.ones((2,16)),tables,np.full((2,16,3,3),1/3),np.full((2,16,27),50),{})

def test_graph_batch_live_future_mutation_and_reset():
    r=np.random.default_rng(41);h=r.normal(size=320).astype(np.float32);o=r.normal(size=80).astype(np.float32)
    n=NodeState(h);initial=n.pseudo_s[-16:].copy();x,s,t=n.replay(o);g=fixture_graph()
    batch=GraphState(g,initial).replay(t,s);live=GraphState(g,initial)
    expected=np.stack([live.update(a,b) for a,b in zip(t,s)])
    np.testing.assert_array_equal(batch,expected)
    np.testing.assert_array_equal(batch[:35],GraphState(g,initial).replay(t[:35],s[:35]))
    mutant=t.copy();mutant[35:]=100
    np.testing.assert_array_equal(batch[:35],GraphState(g,initial).replay(mutant,s)[:35])
    assert np.isfinite(batch).all() and batch.shape==(80,24)
    assert tuple(inspect.signature(GraphState.update).parameters)==('self','temporal','state')

def test_no_confirmation_only_ewma_cannot_build_persistent_state():
    g=fixture_graph();g.parents=[[] for _ in range(16)];s=np.full((200,16),2,dtype=np.int8)
    t=np.full((200,144),8,dtype=np.float32)
    values=GraphState(g,np.ones((16,16),dtype=np.int8)).replay(t,s)
    assert np.all(values[:,11]==0) and np.all(values[:,8]==0)
    assert np.all(values[:,10]==1)

def test_unknown_states_are_missing_not_low():
    s=np.full((12,16),3,dtype=np.int8);t=np.zeros((12,144),dtype=np.float32)
    result=GraphState(fixture_graph(),np.full((16,16),3,dtype=np.int8)).replay(t,s)
    assert np.all(result[:,19]==1) and np.all(result[:,18]==0)
    assert np.isfinite(result).all()

def test_discovery_recovers_known_lag_on_heldout_series():
    rng=np.random.default_rng(823);xs=[];ys=[];ids=[];labels=[]
    for sid in range(96):
        x=rng.normal(size=(192,16)).astype(np.float32)
        for t in range(2,len(x)):x[t,3]=.85*x[t-2,1]+.3*x[t-1,3]+rng.normal(scale=.15)
        temporal=temporal_replay(x,np.zeros((16,16),dtype=np.float32),np.zeros((2,16)),np.zeros(1,dtype=np.int64))
        take=np.arange(16,len(x),4);xs.append(design(temporal[take],take));ys.append(x[take]);ids.extend([sid]*len(take));labels.extend(take>=96)
    sample={'X':np.concatenate(xs),'Y':np.concatenate(ys),'phase':np.asarray(labels,dtype=np.uint8),'ids':np.asarray(ids)}
    edges,rows=discover(sample,{i:i for i in range(96)})
    assert any(e['source']==1 and e['target']==3 and e['lag']==2 for e in edges)
    assert len(edges)<=24 and all(sum(e['target']==j for e in edges)<=2 for j in range(16))
