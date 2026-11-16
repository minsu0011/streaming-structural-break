"""Formula inventory, historical search restoration, and label-free H audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
import csv
import re
import numpy as np
from src.next8.common import BASE,save,read_json,csv_write,digest,utc_now,activity,note
from src.next8.features import NAMES,DETAILED,NORMALIZATION,family,mechanical_group,relation_type

KO={'innovation_location':'조건부 혁신의 위치','innovation_scale':'조건부 혁신의 크기',
    'innovation_distribution':'조건부 혁신의 꼬리·부호 분포','innovation_dependence':'조건부 혁신의 시차 의존',
    'raw_energy':'혁신 제곱 에너지','conditional_energy':'조건부 분산 대비 혁신 에너지',
    'variance_forecast':'H에서 맞춘 ARCH 분산 예측','conditional_scale_likelihood':'짧은 창의 조건부 분산 대안 우도'}


def formula(j,name):
    if j>=117:
        return ('z=clip((conditional_AR8_innovation-H_z_mean)/H_z_sd,-6,6); '
                'LL(v,w)=-w/2*log(v)+(1-1/v)/2*sum(z^2) over last min(observed_count,w); '
                'v in {.5,.7,1.5,2}; log_mixture=logmeanexp LL; lower uses {.5,.7}, '
                'upper uses {1.5,2}; upper_minus_lower is difference. Ages4,8,16,32,64,128,256,512. '
                'Final4 coordinates logmeanexp(all),max(all),logmeanexp(short4),logmeanexp(long4).')
    if j>=93:
        return ('u_t=H-fitted AR8 innovation / H-fitted ARCH conditional sd; '
                'r_t=u_t*u_(t-lag)-H_mean_product, lag=1,2,4,8 by channel; '
                'standardize using H product sd, clip +/-12; selected EWMA8/32 finite-count variance correction, '
                'log1p positive/negative CUSUM with drift.25, or Gaussian log-Bayes mixture on ages1..128. '
                'Exact evidence(): src/next2/dependence_runtime.py.')
    if j>=77:
        return ('E=(AR8 residual-H_mean)^2 / H_stationary_variance (raw) or H_ARCH_forecast (conditional); '
                'a=log(E+1e-4); d=a-H_median(a); lower=clip(max(-d,0)/H_lower_median,0,12); '
                'upper=clip(max(d,0)/H_upper_median,0,12); medians floor1e-4. '
                'current, EWMA8/32 of H-centered/scaled branch, or log1p CUSUM drift.25. '
                'Exact branch_reference/branch_values/appended_step: src/next4/scale_asymmetry.py.')
    if j>=51:
        channel='(r-H_mean)^2/H_stationary_variance-1' if j<64 else 'log(H_ARCH_forecast/H_stationary_variance)'
        return (channel+'; H-center/sd then clip +/-12. Selected current/EWMA8/32/128, '
                'CUSUM +/- drift.25, GLR max/logmean/logage, Bayes .25/1, cumulative or decayed GLR max. '
                'GLR/Bayes use H-estimated serial variance on ages1..512. '
                'Exact serial_variance_step and serial_channel_step in src/next/.')
    base=name.removeprefix('arch8__')
    if base in ('current_z','current_abs_z','current_robust_z'):
        return ('H-fitted AR8 innovation divided by H-fitted ARCH1 sd; '
                'historical-median/MAD feature calibration. Current z, abs(z), or robust z respectively; '
                'underlying standardized input clip +/-20. Exact _arch_feature_step and streaming._update.')
    if base.startswith('cusum'):
        return 'log1p(max(0,previous_CUSUM +/- z - frozen drift)); H-only calibrated ARCH8 input.'
    if base=='running_mean':return 'running arithmetic mean of observed H-calibration+Online standardized ARCH8 innovations; no final-length input.'
    return ('Exponential moment update m=(1-alpha)m+alpha*stat(z), alpha fixed by name scale5/20/160; '
            'mean=(m_z-H_mean)*sqrt(effective_count); abs_mean=abs(mean); '
            'log_second=log(m_z2/H_second); log_variance=log((m_z2-m_z^2)/H_variance) with fixed floors; '
            'absolute_moment=(m_absz-H_absz)*sqrt(count); tail2/tail3/sign standardized rate deviation; '
            'quantile_l1=sum|current-H bin rates|*sqrt(count); '
            'lagK=(m_z_t_z_t-K-H_lagproduct)*sqrt(count); '
            'ar_mean/ar_log_second use H-fitted AR1 error. Select the statistic named by this row; '
            'exact src/features/streaming.py::_update + src/features/historical_calibration.py.')


def main():
    with activity('Complete deployed bank lineage and historical search inventory','FEATURE_LINEAGE','B7_NORMALIZATION_AND_LINEAGE'):
        rows=[]
        for j,name in enumerate(NAMES):
            fam=family(j)
            cumulative=any(s in name for s in ('cusum','memory_max','running_mean'))
            lags=[int(x) for x in re.findall(r'lag(\d+)',name)]
            if 93<=j<117:lags=[(1,2,4,8)[(j-93)//6]]
            sources=(['src/next4/sequential.py','src/next2/dependence_runtime.py','src/next2/complement_bank.py'] if j>=117 else
                     ['src/next2/dependence_runtime.py','src/next2/complement_bank.py'] if j>=93 else
                     ['src/next4/scale_asymmetry.py','src/next/serial_variance_evidence.py','src/features/streaming.py','src/features/historical_calibration.py'])
            rows.append({'feature_id':f'NX8_F{j:03d}','coordinate':j,'original_name':name,'family':fam,
                'meaning_KO':KO.get(fam,'조건부 혁신의 '+str((1,2,4,8)[(j-93)//6])+'시차 의존' if 93<=j<117 else fam),
                'formula':formula(j,name),'NEXT8_transform':'clip((original_feature-H40_median)/max(H40_IQR/1.349,.05),-12,12)',
                'raw_inputs':'One scalar series, H and current/past Online observations; every family shares original raw source',
                'H_fit':'First60% H; same engine continues through last40% calibration and Online; no full-H refit',
                'current_observation_included':True,'explicit_raw_lags':str(lags or [0]),
                'window_or_memory':'Fixed age windows <=512 plus inherited AR/filter state; EWMA has decaying infinite memory; CUSUM/max potentially whole observed prefix',
                'cumulative':cumulative,'mathematical_range_NEXT8':'[-12,12]',
                'primitive_range':'Formula/source-defined; signed even when transformed source is nonnegative, because H-centered',
                'clipping':'Underlying source-specific clips/floors plus final +/-12; per-series boundary counts exported',
                'warmup':'H40 continuation, first32 calibration rows excluded for reference estimates; last16 normalized H rows seed lags',
                'missing_policy':'Nonfinite H/Online rejected; H<160 deterministic zero/UNKNOWN; degenerate H20..80 span <=1e-6 UNKNOWN',
                'direct_transform_of':('same primitive '+mechanical_group(j)),
                'shared_raw_group':'single_raw_series_all153','mechanical_group':mechanical_group(j),
                'support_delay':'H-calibrated feature available immediately; only an OBSERVATION, not independent event confirmation',
                'in_frozen_BL80':bool(j<77 or 93<=j<117),'in_frozen_D2':bool(j>=117),
                'detailed_eligible_before_scores':j in DETAILED,'normalization_id':NORMALIZATION,
                'source_files':'|'.join(sources),'source_sha256':'|'.join(digest(ROOT/p) for p in sources),
                'classification_scope':'Recomputed separately within each training scope; no global RP classification used as CV input'})
        csv_write('FEATURE_LINEAGE_NEXT8.csv',rows)
        registry=[]
        for path in (ROOT/'11_FEATURE_CATALOG.csv',ROOT/'20_EXPERIMENT_LEDGER.csv',
                     *[ROOT/f'MODEL_RESULTS_NEXT{k}.csv' for k in (2,3,4,5)],
                     ROOT/'artifacts/next4/ledger/final_v1/EXPERIMENT_LEDGER.csv',
                     ROOT/'artifacts/next5/ledger/final_v1/EXPERIMENT_LEDGER.csv'):
            with path.open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
            fields=sorted(set(k for r in old for k in r))
            registry.append({'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'rows':len(old),
                'columns':fields,'purpose':'Historical hypotheses/configurations/status restored; historical scores not used as new nested selection evidence'})
            if 'MODEL_RESULTS_NEXT' in path.name:
                selected=[]
                for r in old:
                    selected.append({k:r[k] for k in ('experiment_id','candidate','family','parent','hypothesis','component_change',
                        'change','scientific_question','status','decision','screen_pass','full_completed','feature_count') if k in r})
                save('restoration/'+path.stem+'_HYPOTHESES.json',selected)
        save('HISTORICAL_FEATURE_SEARCH_INVENTORY_NEXT8.json',{'status':'PASS','timestamp':utc_now(),'files':registry,
             'original_catalog_rows':57,'current_distinct_reference_coordinates':153,
             'prior_invariant_variable_certification_found':False,
             'scope_limit':'All153 current J/C/B2/D2 reference-bank coordinates audited and screened. Historical alternative configurations are inventoried, preserved and not silently counted as freshly retested features.',
             'canonical_models_not_assumed_invariant_only':True})
        relations=[]
        for destination in DETAILED:
            for origin in range(len(NAMES)):
                if origin==destination or mechanical_group(origin)==mechanical_group(destination):
                    relations.append({'source':f'NX8_F{origin:03d}','destination':f'NX8_F{destination:03d}',
                        'edge_type':relation_type(origin,destination),'common_raw':True,
                        'independent_confirmation':False,'empirical_incremental_support':'NOT_TESTED_YET',
                        'mechanical_reason':mechanical_group(destination),'H_reference':NORMALIZATION})
        csv_write('MECHANICAL_RELATION_AUDIT.csv',relations)
    note('Lineage inventory saved','153 formula-linked coordinates, historical57-feature catalog and NEXT2-NEXT5 search ledgers mapped','B7_NORMALIZATION_AND_LINEAGE')


if __name__=='__main__':main()
