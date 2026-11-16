"""Portable JSON descriptors for the two tiny gate families."""
import numpy as np
from scipy.special import expit
from src.next3.learned_router import LogisticGate,TreeGate
from src.next3.router import blend_with_weight


def describe_gate(model):
    result={'format':'NEXT3_GATE_JSON_V1','training_detail':model.training_detail}
    if isinstance(model,LogisticGate):
        return {**result,'kind':'R2_LOGISTIC_GATE','center':model.center.tolist(),
            'scale':model.scale.tolist(),'coefficients':model.coefficients.tolist()}
    if isinstance(model,TreeGate):
        tree=model.estimator.tree_
        return {**result,'kind':'R3_DEPTH2_GATE','classes':model.estimator.classes_.tolist(),
            'children_left':tree.children_left.tolist(),'children_right':tree.children_right.tolist(),
            'feature':tree.feature.tolist(),'threshold':tree.threshold.tolist(),'value':tree.value[:,0,:].tolist()}
    raise ValueError('Unregistered gate type')


def descriptor_weights(descriptor,features):
    if descriptor['format']!='NEXT3_GATE_JSON_V1':raise ValueError('Unknown gate format')
    if descriptor['kind']=='R2_LOGISTIC_GATE':
        x=np.asarray(features,dtype=np.float64)
        z=np.clip((x-np.asarray(descriptor['center']))/np.asarray(descriptor['scale']),-8.,8.)
        return .3*expit(np.column_stack([np.ones(len(z)),z])@np.asarray(descriptor['coefficients']))
    if descriptor['kind']=='R3_DEPTH2_GATE':
        x=np.asarray(features,dtype=np.float32)
        nodes=np.zeros(len(x),dtype=np.int32)
        left=np.asarray(descriptor['children_left']);right=np.asarray(descriptor['children_right'])
        feature=np.asarray(descriptor['feature']);threshold=np.asarray(descriptor['threshold'])
        for _ in range(2):
            active=np.flatnonzero(left[nodes]>=0)
            current=nodes[active]
            nodes[active]=np.where(x[active,feature[current]]<=threshold[current],left[current],right[current])
        assert (left[nodes]<0).all()
        values=np.asarray(descriptor['value'])[nodes]
        classes=descriptor['classes']
        probability=values[:,classes.index(1)]/values.sum(axis=1) if 1 in classes else np.zeros(len(x))
        return np.floor(.3*probability*10.+.5)/10.
    raise ValueError('Unknown gate kind')


def descriptor_predict(descriptor,features):
    return blend_with_weight(features[:,0],features[:,1],descriptor_weights(descriptor,features))
