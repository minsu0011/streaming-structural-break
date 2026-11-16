"""All pairs of frozen references, admitted dependence signal and fixed blend."""
from pathlib import Path
import sys
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.evaluation import load_full_oof
from scripts.forensic_next_candidate import rank_within_time
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

J='J102_SERIAL_VARIANCE_512';Y='Y001_LONG_VARIANCE_ARCH1';R='B104_LONG_VARIANCE_NO_GLR_MEMORY'
C='C205_DEPENDENCE';B='BL2_C205_DEPENDENCE_80'
MODELS=(J,Y,R,C,B)
SIGNATURES=('MEAN_DOMINANT','SCALE_DOMINANT','DEPENDENCE_DOMINANT','TAIL_DOMINANT','MIXED','WEAK_OR_AMBIGUOUS')


def correlation(left,right):
    if len(left)<2 or np.std(left)==0 or np.std(right)==0:return None
    return float(np.corrcoef(left,right)[0,1])


def main():
    out=ROOT/'artifacts/next2/finalist_disagreement';out.mkdir(parents=True,exist_ok=True)
    data=prepare_dev(ROOT);predictions={};sources={};metadata=None
    for eid in MODELS:
        _,_,meta,p,source=load_full_oof(data,eid)
        if metadata is None:metadata=meta
        else:
            for name in metadata:np.testing.assert_array_equal(meta[name],metadata[name])
        predictions[eid]=p;sources[eid]=source
    meta_path=ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    meta_lock=json.loads((meta_path.parent/'REGIME_ASSIGNMENT_LOCK.json').read_text())
    assert sha256(meta_path)==meta_lock['files_sha256'][meta_path.name]
    series=pd.read_parquet(meta_path).set_index('dataset_id');data.guard.admit(series.index.to_numpy(),purpose='DEV-only final candidate disagreement slices')
    identity={'models':list(MODELS),'sources':sources,'signature_sha256':sha256(meta_path),
        'script_sha256':sha256(Path(__file__)),'rank_source_sha256':sha256(ROOT/'scripts/forensic_next_candidate.py'),
        'new_fits':0,'promotion_authority':False,
        'distilled_identity':'J102 exact tree-used runtime has identical OOF probabilities; reuse J102 rather than duplicate it.'}
    lock=out/'POLICY.json'
    if lock.exists():assert json.loads(lock.read_text())==identity
    else:write_json(lock,identity)
    rows=[];cases=[]
    for fold in range(5):
        selected=np.flatnonzero(metadata['fold']==fold)
        ids,t,y=(metadata[k][selected] for k in ('dataset_id','time_online','target'))
        sm=series.reindex(ids)
        p={eid:predictions[eid][selected] for eid in MODELS}
        ranks={eid:rank_within_time(p[eid],t) for eid in MODELS}
        masks={'all':np.ones(len(ids),bool),'negative_points':y==0,'no_break_controls':sm.tau.to_numpy()<0}
        masks.update({'historical_regime_'+str(k):sm.historical_regime.to_numpy()==k for k in range(5)})
        for name in SIGNATURES:
            masks['mechanism_only_'+name]=sm.signature.to_numpy()==name
            masks['mechanism_plus_controls_'+name]=np.isin(sm.signature.to_numpy(),[name,'NO_BREAK_CONTROL'])
        for scope,mask in masks.items():
            if not mask.any():continue
            support_ids=np.unique(ids[mask]);support=series.loc[support_ids]
            ordinary_ranks={eid:rankdata(p[eid][mask],method='average') for eid in MODELS}
            for left,right in itertools.combinations(MODELS,2):
                a,b=p[left][mask],p[right][mask];ra,rb=ranks[left][mask],ranks[right][mask]
                row={'left':left,'right':right,'fold':fold,'scope':scope,'points':int(mask.sum()),'series':len(support),
                    'break_series':int((support.tau>=0).sum()),'null_series':int((support.tau<0).sum()),
                    'pearson':correlation(a,b),'spearman_pooled_within_scope':correlation(ordinary_ranks[left],ordinary_ranks[right]),
                    'correlation_of_full_fold_within_time_percentile_ranks':correlation(ra,rb),
                    'mean_absolute_rank_difference':float(np.abs(ra-rb).mean()),
                    'mean_absolute_probability_difference':float(np.abs(a-b).mean()),
                    'probability_bitwise_equal':bool(np.array_equal(a,b))}
                negative=y[mask]==0
                for fraction in (.01,.05):
                    la,lb=ra>=1-fraction,rb>=1-fraction;prefix='top_'+str(fraction)
                    row.update({prefix+'_jaccard':float((la&lb).sum()/max(1,(la|lb).sum())),
                        prefix+'_disagreement_fraction':float((la!=lb).mean()),
                        prefix+'_left_only_fraction':float((la&~lb).mean()),prefix+'_right_only_fraction':float((~la&lb).mean()),
                        prefix+'_negative_point_disagreement':float((la!=lb)[negative].mean()) if negative.any() else None})
                rows.append(row)
        frame=pd.DataFrame({'dataset_id':ids,'absolute_rank_difference':np.abs(ranks[J]-ranks[B]),
            'signed_probability_difference':p[B]-p[J]})
        grouped=frame.groupby('dataset_id').agg(mean_absolute_rank_difference=('absolute_rank_difference','mean'),
            maximum_absolute_rank_difference=('absolute_rank_difference','max'),mean_blend_minus_j_probability=('signed_probability_difference','mean'))
        chosen=grouped.sort_values(['mean_absolute_rank_difference','dataset_id'],ascending=[False,True]).head(10).reset_index()
        chosen['fold']=fold
        chosen=chosen.merge(series[['historical_regime','signature']].reset_index(),on='dataset_id',validate='one_to_one')
        cases.extend(chosen.to_dict('records'))
        print('NEXT2 FINALIST DISAGREEMENT',fold,len(rows),flush=True)
    table=pd.DataFrame(rows);table.to_csv(out/'PAIRWISE_BY_SLICE.csv',index=False)
    pd.DataFrame(cases).to_csv(out/'LARGEST_BLEND_DISAGREEMENT_SERIES.csv',index=False)
    aggregate=table.groupby(['left','right','scope'],as_index=False).agg(folds=('fold','count'),
        points=('points','sum'),break_series=('break_series','sum'),null_series=('null_series','sum'),
        mean_pearson=('pearson','mean'),mean_spearman=('spearman_pooled_within_scope','mean'),
        mean_within_time_rank_correlation=('correlation_of_full_fold_within_time_percentile_ranks','mean'),
        mean_absolute_rank_difference=('mean_absolute_rank_difference','mean'),
        mean_top1_jaccard=('top_0.01_jaccard','mean'),mean_top5_jaccard=('top_0.05_jaccard','mean'),
        mean_top1_negative_disagreement=('top_0.01_negative_point_disagreement','mean'),
        mean_top5_negative_disagreement=('top_0.05_negative_point_disagreement','mean'))
    aggregate['break_support_at_least_100']=aggregate.break_series>=100
    aggregate.to_csv(out/'PAIRWISE_SUMMARY.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),
        'pairs':10,'fold_slice_rows':len(rows),'new_fits':0,'seal_rows':0,
        'files_sha256':{name:sha256(out/name) for name in ('PAIRWISE_BY_SLICE.csv','PAIRWISE_SUMMARY.csv','LARGEST_BLEND_DISAGREEMENT_SERIES.csv')},
        'definitions':'Ordinary Spearman uses ranks recomputed in each stated scope. Within-time percentile ranks and top tails are computed in each full held-out fold before slicing. Negative points include pre-change observations; no-break controls are reported separately. Means across folds are descriptive, not Fisher-z estimates or uncertainty intervals. Mechanism-only and mechanism-plus-controls are distinct.'})
    checkpoint('Full pairwise finalist and mechanism disagreement',['Keep correlation definitions explicit','Complete engineering and synthetic tradeoff synthesis'])


if __name__=='__main__':main()
