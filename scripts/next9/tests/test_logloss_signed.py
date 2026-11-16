import src.next9
import numpy as np
from src.next9.logloss_signed import binary_loss


def test_unsigned_labels_cannot_wrap_negative():
    y=np.array([0,1],dtype=np.uint8);p=np.array([.5,.5],dtype=np.float32)
    old=-y*np.log(p)-(1-y)*np.log1p(-p)
    assert old[1]<0
    np.testing.assert_allclose(binary_loss(y,p),np.log(2),rtol=0,atol=1e-15)


def test_calibrated_predictions_improve_loss_and_endpoints_finite():
    y=np.array([0,1],np.uint8)
    assert binary_loss(y,np.array([.1,.9],np.float32)).mean()<binary_loss(y,np.array([.9,.1],np.float32)).mean()
    assert np.isfinite(binary_loss(y,np.array([1.,0.],np.float32))).all()
    np.testing.assert_array_equal(binary_loss(y,np.array([.2,.8],np.float32)),binary_loss(y.astype(float),np.array([.2,.8],np.float32)))


def test_compare_paired_loss_direction_with_unsigned_labels():
    from src.next9.inference_signed_v4 import compare
    y=np.tile(np.array([0,1],np.uint8),16)
    ids=np.repeat(np.arange(16),2);times=np.tile([0,0],16)
    predictions={'O1':np.full(32,.5,np.float32),'O3_MOVE':np.where(y==1,.8,.2).astype(np.float32)}
    result=compare(y,predictions,times,ids,'SIGNED_LOSS_REGRESSION',draws=2000)
    np.testing.assert_allclose(result['logloss']['O1'],np.log(2),rtol=0,atol=1e-15)
    assert result['contrasts']['O3_MOVE__vs__O1']['logloss_gain']['ci_low']>.4
    assert all(value>=0 for value in result['logloss'].values())
