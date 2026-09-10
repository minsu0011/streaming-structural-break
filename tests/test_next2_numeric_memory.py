import numpy as np
from src.next2.numeric_memory import inventory


def test_retained_views_count_full_owner_once():
    owner=np.zeros((100,4),dtype=np.float64)
    views=(owner[:2],owner[::3,1],owner)
    found=inventory(views)
    assert len(found)==1
    assert sum(v['bytes'] for v in found.values())==3200


def test_distinct_allocations_and_cycle():
    a=np.zeros(32,dtype=np.float32);b=a.copy()
    values=[a,b];values.append(values)
    assert len(inventory(values))==2
    assert sum(v['bytes'] for v in inventory(values).values())==256
