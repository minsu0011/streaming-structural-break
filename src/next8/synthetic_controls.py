"""Fixed matched counterexamples through the identical NEXT8 feature engine."""
from pathlib import Path
import numpy as np
from src.next8.common import atomic,read_json,digest,utc_now,activity
from src.next8.features import FeatureState,N_FEATURES,NORMALIZATION,source_hash
from src.next8.cache import FeatureCache,immutable_npz

BACKGROUNDS=('Gaussian','AR08','ARCH1','Student_t5')
H_LENGTHS=(512,2048)
INTERVENTIONS=('none','single_shock_8sd_at256','repeated_shocks_8sd_at256_264_272',
               'scale_up_variance2_after256','scale_down_variance05_after256')
SEED=8101604
SERIES_PER_BACKGROUND=128
ONLINE_LENGTH=1024
BURN=2048


def matched_path(background,index,H_length):
    if background not in BACKGROUNDS or not 0<=index<SERIES_PER_BACKGROUND or H_length not in H_LENGTHS:
        raise ValueError('Outside frozen synthetic menu')
    seed=np.random.SeedSequence([SEED,BACKGROUNDS.index(background),int(index)])
    rng=np.random.default_rng(seed)
    n=BURN+max(H_LENGTHS)+ONLINE_LENGTH
    if background=='Student_t5':raw=rng.standard_t(5,n)/np.sqrt(5/3)
    else:
        noise=rng.normal(size=n)
        if background=='Gaussian':raw=noise
        else:
            raw=np.empty(n);previous=0.
            for k,e in enumerate(noise):
                value=.8*previous+.6*e if background=='AR08' else np.sqrt(.2+.8*previous*previous)*e
                raw[k]=value;previous=value
    raw=raw[BURN:].astype(np.float32)
    historical=raw[max(H_LENGTHS)-H_length:max(H_LENGTHS)].copy()
    online=raw[max(H_LENGTHS):].copy()
    return historical,online


def interventions(online,index):
    if np.asarray(online).shape!=(ONLINE_LENGTH,):raise ValueError('Frozen synthetic online horizon')
    sign=1. if index%2==0 else -1.
    output=[]
    for name in INTERVENTIONS:
        values=np.asarray(online,dtype=np.float32).copy()
        if name.startswith('single'):values[256]+=sign*8.
        elif name.startswith('repeated'):values[[256,264,272]]+=sign*8.
        elif name.startswith('scale_up'):values[256:]*=np.float32(np.sqrt(2.))
        elif name.startswith('scale_down'):values[256:]*=np.float32(np.sqrt(.5))
        output.append(values)
    return output


def block_coordinates(block):
    if not 0<=block<len(BACKGROUNDS)*SERIES_PER_BACKGROUND*len(H_LENGTHS):raise ValueError('Synthetic block outside menu')
    unit,h=divmod(int(block),len(H_LENGTHS));background,index=divmod(unit,SERIES_PER_BACKGROUND)
    return BACKGROUNDS[background],index,H_LENGTHS[h],unit


def generate_block(block,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    background,index,H_length,group=block_coordinates(block)
    if (directory/'DONE.json').exists():
        done=read_json(directory/'DONE.json')
        assert done['block']==block and done['feature_source_sha256']==source_hash()
        assert digest(directory/'DERIVED_FEATURES.npz')==done['derived_features_sha256']
        return done
    if any(directory.iterdir()):raise RuntimeError('Partial synthetic block preserved for explicit repair')
    historical,online=matched_path(background,index,H_length)
    initial=FeatureState(historical);variants=interventions(online,index)
    values=[];summary=[]
    for name,points in zip(INTERVENTIONS,variants):
        state=FeatureState.restore(initial);feature,_=state.replay(points)
        values.append(feature)
        summary.append({'intervention':name,'boundary_fraction_per_feature':np.mean(np.abs(feature)>=12,axis=0).tolist(),
                        'H_invalid_feature_count':int(np.sum(~initial.valid))})
    values=np.concatenate(values)
    ids=1000000000+block*len(INTERVENTIONS)+np.arange(len(INTERVENTIONS),dtype=np.int64)
    repeated=lambda a:np.repeat(np.asarray(a)[None,...],len(INTERVENTIONS),axis=0)
    immutable_npz(directory/'DERIVED_FEATURES.npz',values=values,dataset_id=ids,
        seed_values=repeated(initial.history),low=repeated(initial.low),high=repeated(initial.high),
        valid=repeated(initial.valid),epsilon=repeated(initial.epsilon),context=repeated(initial.context),
        center=initial.center,spread=initial.spread,H_length=np.full(len(INTERVENTIONS),H_length,dtype=np.int32))
    done={'status':'COMPLETE','timestamp':utc_now(),'block':block,'background':background,'series_index':index,
        'H_length':H_length,'dependency_group':group,'interventions':list(INTERVENTIONS),
        'feature_source_sha256':source_hash(),'normalization':NORMALIZATION,
        'derived_features_sha256':digest(directory/'DERIVED_FEATURES.npz'),'shape':list(values.shape),
        'summary':summary,'synthetic_only':True,'real_dataset_access':False,
        'H_fit_reused_by_deepcopy_for_matched_interventions':True,
        'same_innovation_path_across_H_lengths':True,'H_lengths_and_interventions_not_independent_samples':True,
        'event_time_never_given_to_feature_engine':True,'shock_sign_balanced_by_index_parity':True}
    atomic(directory/'DONE.json',done);return done


class SyntheticCache:
    """Derived synthetic observations, explicitly separate from RP admission.

    Reuses tested causal array lookup functions, never calls a real-data loader
    or represents synthetic IDs as admitted ResearchPool IDs. Inference only.
    """
    current=FeatureCache.current
    lag=FeatureCache.lag
    states=FeatureCache.states
    epsilon=FeatureCache.epsilon
    context=FeatureCache.context
    forecast_rows=FeatureCache.forecast_rows

    def __init__(self,directory):
        directory=Path(directory);done=read_json(directory/'DONE.json')
        if not done['synthetic_only'] or done['real_dataset_access']:raise ValueError('Synthetic-only cache required')
        assert done['feature_source_sha256']==source_hash()
        assert digest(directory/'DERIVED_FEATURES.npz')==done['derived_features_sha256']
        with np.load(directory/'DERIVED_FEATURES.npz',allow_pickle=False) as a:
            self.values=a['values'].copy()
            self.series={k:a[k].copy() for k in ('dataset_id','seed_values','low','high','valid','epsilon','context','H_length')}
        self.series_index=np.repeat(np.arange(len(INTERVENTIONS)),ONLINE_LENGTH)
        t=np.tile(np.arange(ONLINE_LENGTH,dtype=np.int32),len(INTERVENTIONS))
        condition=np.repeat(np.arange(len(INTERVENTIONS)),ONLINE_LENGTH)
        target=((condition>=3)&(t>=256)).astype(np.uint8)
        self.meta={'dataset_id':self.series['dataset_id'][self.series_index],'time_online':t,
            'target':target,'intervention':condition,
            'dependency_group':np.full(len(t),done['dependency_group'],dtype=np.int32)}
        self.receipt=done
        assert self.values.shape==(ONLINE_LENGTH*len(INTERVENTIONS),N_FEATURES)
