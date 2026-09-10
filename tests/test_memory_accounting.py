import sys
import numpy as np
from scripts.stress_streaming import deep_size


def test_nested_shared_array_is_counted_once():
    array=np.arange(100,dtype=np.float64)
    inner=(array,array)
    value=[inner,array]
    assert deep_size(value)==sys.getsizeof(value)+sys.getsizeof(inner)+sys.getsizeof(array)


def test_view_retains_owner_without_double_counting():
    array=np.arange(100,dtype=np.float64)
    view=array[::2]
    value=(view,array)
    assert deep_size(value)==sys.getsizeof(value)+sys.getsizeof(view)+sys.getsizeof(array)


def test_cycle_and_nested_growth_are_visible():
    value=[];value.append(value)
    assert deep_size(value)==sys.getsizeof(value)
    before=deep_size(value)
    value.append((np.zeros(1000),))
    assert deep_size(value)>=before+8000
