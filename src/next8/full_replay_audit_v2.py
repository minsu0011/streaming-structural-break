"""Full replay v2: use the actual float64 M6 blend on float32 head arrays.

Supersedes only the unexecuted v1 audit helper, not any fitted predictor.
"""
import hashlib
import numpy as np
from numba import njit
from src.next8.common import BASE,read_json,digest
from src.next8.features import FeatureState,N_FEATURES,DETAILED,UNKNOWN
from src.next8.runtime_features import raw_arguments,normalized_step
from src.next8.cache import FeatureCache
from src.next8.data import outer_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.runtime_audit import load_scope_runtime,observations
from src.next8.graph_states import predict_graph_state
from src.next8.joint_models import joint_design,model_design
from src.next6.data import reference_matrix
from src.next5.model import feature_module
from src.next8.integration import fixed_blend


def audit_m6_probability(baseline,head,eligible):
    return fixed_blend(baseline,head) if eligible else np.asarray(baseline,dtype=np.float64)



@njit(cache=True)
def full_scalar_bank_replay(online,engine,normalization):
    result=np.empty((len(online),153),dtype=np.float32)
    for k in range(len(online)):result[k]=normalized_step(float(online[k]),engine,normalization)
    return result


class ForbiddenLabels:
    def __getitem__(self,key):raise AssertionError('Fresh inference used breakOccurred labels')
    def __array__(self,*args,**kwargs):raise AssertionError('Fresh inference consumed breakOccurred labels')


class FreshSeriesCache:
    """Only fresh H state and scalar-derived observed feature values."""
    current=FeatureCache.current
    lag=FeatureCache.lag
    states=FeatureCache.states
    epsilon=FeatureCache.epsilon
    context=FeatureCache.context
    forecast_rows=FeatureCache.forecast_rows

    def __init__(self,sid,historical,online):
        seed=FeatureState(historical);self.seed=seed
        self.series={k:np.asarray(v)[None,...] for k,v in {
            'dataset_id':sid,'seed_values':seed.history.copy(),'low':seed.low,'high':seed.high,'valid':seed.valid,
            'epsilon':seed.epsilon,'context':seed.context,'center':seed.center,'spread':seed.spread,'fit_length':seed.fit_length,
            'H_length':len(historical),'calibration_clip_count':seed.calibration_clip_count,
            'reference_sha256':seed.normalizer_fit_sha256}.items()}
        if seed.engines:
            engine,normalization=raw_arguments(seed,range(N_FEATURES))
            self.values=full_scalar_bank_replay(np.asarray(online,dtype=np.float32),engine,normalization)
        else:self.values=np.zeros((len(online),N_FEATURES),dtype=np.float32)
        self.series['online_bound_count']=(np.abs(self.values)>=12).sum(axis=0)[None,:]
        self.series_index=np.zeros(len(online),dtype=np.int32)
        self.meta={'dataset_id':np.full(len(online),sid,dtype=np.int64),'time_online':np.arange(len(online),dtype=np.int32),
            'target':ForbiddenLabels()}
        assert seed.calibration_reference_sha256==seed.online_reference_sha256


def load_audit_models(cache):
    bundles={}
    for early in (False,True):
        for fold in range(5):
            scope=early_scope(outer_scope(fold,cache.data),early)
            heads,baseline,reference,saved,states,integration=load_scope_runtime(cache,scope,fold,early=early)
            bundles[fold,early]={'scope':scope,'heads':heads,'baseline':baseline,'reference':reference,
                'saved':saved,'states':states,'integration':integration}
    matrices={kind:reference_matrix(kind,cache.data) for kind in ('J102','C205')}
    return bundles,matrices


def compare_fresh_head(fresh,global_rows,bundle):
    rows=np.arange(len(fresh.values));cross=np.full((len(rows),len(DETAILED)),UNKNOWN,dtype=np.int8)
    for j,model in bundle['heads']['M5'].graph_models.items():cross[:,DETAILED.index(j)]=predict_graph_state(fresh,rows,j,model)[0]
    states=bundle['states'];at=np.searchsorted(states['global_row'],global_rows)
    np.testing.assert_array_equal(states['global_row'][at],global_rows);np.testing.assert_array_equal(cross,states['state'][at])
    saved=bundle['saved'];keep=np.isin(global_rows,saved['global_row']);loc=np.searchsorted(saved['global_row'],global_rows[keep])
    seen={};result={};predictions={}
    for method,prepared in bundle['heads'].items():
        actual=prepared.method
        if actual not in seen:
            design=model_design(prepared.model,joint_design(fresh,rows,prepared.selection,actual,cross))
            seen[actual]=prepared.model.predict_proba(design)[:,1].astype(np.float32)
        p=seen[actual];predictions[method]=p
        error=float(np.max(np.abs(p[keep].astype(float)-saved[method][loc]))) if keep.any() else None
        assert error is None or error<=np.finfo(np.float32).eps
        result[method]={'saved_points':int(keep.sum()),'maximum_float32_error':error,
            'bitwise_equal':bool(np.array_equal(p[keep].view(np.uint32),saved[method][loc].view(np.uint32))) if keep.any() else None,
            'fresh_prediction_sha256':hashlib.sha256(p.tobytes()).hexdigest(),'actual_alias_method':actual}
    return result,predictions


def audit_series(cache,sid,bundles,matrices):
    h,o,global_rows=observations(cache,sid);fold=cache.data.guard.split.assignment(sid)[1]
    fresh=FreshSeriesCache(sid,h,o);index=cache.data.id_to_index[sid]
    existing=np.asarray(cache.values[global_rows],dtype=np.float32)
    # Equality includes signed-zero bits, beyond ordinary numerical equality.
    np.testing.assert_array_equal(fresh.values.view(np.uint32),existing.view(np.uint32))
    for name,value in fresh.series.items():np.testing.assert_array_equal(value[0],cache.series[name][index],err_msg=name)
    base=bundles[fold,False]['baseline'];reference_models=(base.j102,base.c205)
    full_reference_features=[];reference_parity={}
    for kind,model in zip(('J102','C205'),reference_models):
        # Models/sources are verified on worker admission and by the final
        # integrity audit; do not rehash identical source files for every H.
        x=feature_module(model.module_name).make_state(h,model.settings).replay(o)
        expected=np.asarray(matrices[kind][global_rows],dtype=np.float32)
        np.testing.assert_array_equal(x.view(np.uint32),expected.view(np.uint32));full_reference_features.append(x)
        reference_parity[kind]={'bitwise_equal':True,'columns':x.shape[1],'fresh_features_sha256':hashlib.sha256(x.tobytes()).hexdigest()}
    results={}
    for early in (False,True):
        bundle=bundles[fold,early];head_result,new_prediction=compare_fresh_head(fresh,global_rows,bundle)
        baseline=bundle['baseline'];b=.8*baseline.j102.predict(full_reference_features[0])+.2*baseline.c205.predict(full_reference_features[1])
        reference=bundle['reference'];keep=np.isin(global_rows,reference['global_row']);at=np.searchsorted(reference['global_row'],global_rows[keep])
        error=float(np.max(np.abs(b[keep]-reference['prediction'][at]))) if keep.any() else None
        assert error is None or error<1e-12
        saved=bundle['saved'];head_at=np.searchsorted(saved['global_row'],global_rows[keep])
        np.testing.assert_array_equal(saved['global_row'][head_at],global_rows[keep])
        eligible=bundle['integration']['eligible']
        integrated=audit_m6_probability(b,new_prediction['M5'],eligible)
        expected_integrated=audit_m6_probability(reference['prediction'][at],saved['M5'][head_at],eligible)
        integrated_error=float(np.max(np.abs(integrated[keep]-expected_integrated))) if keep.any() else None
        assert integrated_error is None or integrated_error<=.05*np.finfo(np.float32).eps+1e-12
        results['EARLY' if early else 'NORMAL']={'scope':bundle['scope'].provenance(),'head_results':head_result,
            'baseline_points':int(keep.sum()),'baseline_float64_max_error':error,
            'baseline_prediction_sha256':hashlib.sha256(b.tobytes()).hexdigest(),
            'M6_eligible':eligible,'M6_fixed_composition_source_verified':True,'M6_replay_composition_max_abs':integrated_error,
            'M6_fresh_prediction_sha256':hashlib.sha256(integrated.tobytes()).hexdigest()}
    return {'status':'PASS','series_id':sid,'outer_fold':fold,'H_length':len(h),'online_points':len(o),
        'normalized_features':N_FEATURES,'fresh_scalar_kernel_feature_cache_bitwise_equal':True,
        'all_H_metadata_equal':sorted(fresh.series),'fresh_features_sha256':hashlib.sha256(fresh.values.tobytes()).hexdigest(),
        'full_H_original_reference_feature_parity':reference_parity,'predictions':results,
        'head_features_and_graph_states_built_from_fresh_H_and_scalar_online_kernel':True,
        'saved_feature_cache_used_only_as_comparison_target':True,'break_labels_or_tau_as_inference_input':False,
        'new_model_fits':0,'short_H_zero_UNKNOWN_fallback':len(h)<160}
