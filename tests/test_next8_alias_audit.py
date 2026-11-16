"""UNKNOWN masks can differ despite identical fitted coordinate names."""
import src.next8
import numpy as np
import pytest
from src.next8.learning import Design
from src.next8.joint_models import joint_design
from src.next8.alias_audit import assert_alias_coordinates


class Cache:
    meta={'dataset_id':np.array([31,32])}
    def current(self,rows,cols):return np.array([[4.],[-4.]])[rows]
    def lag(self,rows,cols,lag):return np.zeros((len(rows),len(cols)))
    def context(self,rows):return np.zeros((len(rows),7))
    def states(self,rows,cols,previous=False):return np.full((len(rows),len(cols)),3,dtype=np.int8)


def test_identical_names_do_not_hide_unknown_fallback_change():
    selection={'stable_features':[],'variable_features':[0],'classifications':[],
               'orientation_state_specs':{'0':{'kind':'OWN_F'}}}
    a=joint_design(Cache(),np.arange(2),selection,'M4')
    b=joint_design(Cache(),np.arange(2),selection,'M2')
    with pytest.raises(ValueError,match='UNKNOWN'):assert_alias_coordinates(a,b,('F000_variable_linear',))


def test_unretained_extra_columns_cannot_change_alias_prediction():
    a=Design(np.array([[1.,2.],[1.,3.]],dtype=np.float32),('constant_context','unused_graph'),np.array([31,32]))
    b=Design(np.ones((2,1),dtype=np.float32),('constant_context',),np.array([31,32]))
    assert_alias_coordinates(a,b,('constant_context',))
