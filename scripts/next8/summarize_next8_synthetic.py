"""Summarize all fixed synthetic cases without treating variants as samples."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,digest,atomic,utc_now,activity,note,csv_write
from src.next8.synthetic_summary import load_synthetic_group_arrays,synthetic_summary_tables


def main():
    lock=read_json(BASE/'SYNTHETIC_SUMMARY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    directory=BASE/'synthetic_summary';directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise RuntimeError('Synthetic summary outputs already exist; no implicit reruns')
    with activity('Summarize all fixed matched synthetic groups','SYNTHETIC_SUMMARY','B7_NORMALIZATION_AND_LINEAGE'):
        arrays=load_synthetic_group_arrays();files={};operations=[];counts={}
        for name,rows,work in synthetic_summary_tables(arrays):
            csv_write('synthetic_summary/'+name+'.csv',rows);files[name+'.csv']=digest(directory/(name+'.csv'))
            counts[name]=len(rows);operations.extend(work)
        atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'files_sha256':files,'row_counts':counts,
            'independent_innovation_groups':512,'groups_per_background':128,'matched_H_lengths':[512,2048],
            'fold_procedures_averaged_before_uncertainty':5,'draws':2000,'seed':8101604,
            'resampling_unit':'Original synthetic innovation path, stratified by fixed background; H/intervention/model variants paired within path',
            'all_fixed_windows_and_horizons_retained':True,'no_synthetic_refits_or_threshold_optimization':True,
            'intervals':'Pointwise descriptive 95% bootstrap; extensive fixed table is not a multiplicity-adjusted discovery family',
            'positive_fraction_is_posterior':False,'synthetic_score_response_is_not_real_TS_AUC_gain':True,
            'constant_and_bitwise_equal_columns_use_exact_bootstrap_equivalence':True,'operations':operations})
    note('Complete matched synthetic counterexample uncertainty','512 independent paths, all fixed cases retained','B7_NORMALIZATION_AND_LINEAGE')


if __name__=='__main__':main()
