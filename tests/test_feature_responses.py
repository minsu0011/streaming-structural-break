import numpy as np
from src.features.streaming import replay
from src.validation.synthetic import fixture

def features(kind):
    h,o,_=fixture(kind)
    return np.array(list(replay(h,o)))

def test_mean_direction_and_persistence():
    none,up,down=map(features,['no_break','mean_up','mean_down'])
    assert up[-80:,21].mean()>none[-80:,21].mean()+3
    assert down[-80:,21].mean()<none[-80:,21].mean()-3
    outlier=features('single_outlier')
    assert abs(outlier[-50:,21]).mean()<abs(up[-50:,21]).mean()

def test_variance_both_directions():
    none,up,down=map(features,['no_break','variance_up','variance_down'])
    assert up[-80:,24].mean()>none[-80:,24].mean()+1
    assert down[-80:,24].mean()<none[-80:,24].mean()-1

def test_dependence_change():
    none,up,reverse=map(features,['no_break','ar_up','ar_reverse'])
    assert up[-120:,30].mean()>none[-120:,30].mean()+1
    assert reverse[-120:,30].mean()<-2

def test_distribution_change_and_no_break_finiteness():
    # Multi-replicate fixed-seed response: tail rarity needs more observations.
    deltas=[]
    for seed in range(12):
        h,o,_=fixture('heavy_tail',seed=seed,online_length=1000)
        f=np.array(list(replay(h,o)))
        deltas.append(f[-500:,42].mean())  # tail3 deviation, scale80
    assert np.mean(deltas)>0
    assert np.isfinite(features('no_break')).all()
