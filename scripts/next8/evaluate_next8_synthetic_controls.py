"""Finite matched null/intervention evaluation using frozen models only."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
import numpy as np
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity,note
from src.next8.cache import immutable_npz
from src.next8.synthetic_controls import SyntheticCache,block_coordinates
from src.next8.synthetic_evaluation import (load_frozen_models,baseline_matrices,predict_block_heads,summarize_scores,
    auxiliary_errors,WINDOWS,AUX_METHODS,SCORE_METHODS)
from src.next8.features import DETAILED,HORIZONS


def main(part,parts):
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json');assert release['status']=='FROZEN'
    lock=read_json(BASE/'SYNTHETIC_EVALUATION_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    with activity(f'Load synthetic frozen model bank part{part}','SYNTHETIC_MODEL_LOAD','B7_NORMALIZATION_AND_LINEAGE'):
        folds=load_frozen_models(release)
        for fold in folds[1:]:
            for m,ref in zip(fold['baseline'],folds[0]['baseline']):assert m.settings==ref.settings and m.feature_sha256==ref.feature_sha256
    records=[]
    for block in range(part,1024,parts):
        directory=BASE/'synthetic_evaluation'/f'block_{block:04d}';directory.mkdir(parents=True,exist_ok=True)
        if (directory/'DONE.json').exists():
            done=read_json(directory/'DONE.json')
            for name,sha in done['files_sha256'].items():assert digest(directory/name)==sha
            records.append({'block':block,'done_sha256':digest(directory/'DONE.json')});continue
        if any(directory.iterdir()):raise RuntimeError('Partial synthetic model application preserved')
        with activity(f'Matched synthetic model/dynamics block{block}','SYNTHETIC_EVALUATION','B7_NORMALIZATION_AND_LINEAGE'):
            cache=SyntheticCache(BASE/'synthetic_controls'/f'block_{block:04d}')
            reference_features=baseline_matrices(block,folds[0]['baseline'])
            scores=[];summaries=[];movement=[];counts=[];aliases=[]
            for fold in folds:
                prediction,cross,alias=predict_block_heads(cache,fold['heads'],fold['baseline'],reference_features,fold['eligible'])
                error,count=auxiliary_errors(cache,fold['auxiliary'])
                scores.append(prediction);summaries.append(summarize_scores(prediction));movement.append(error);counts.append(count);aliases.append(alias)
            immutable_npz(directory/'SCORES.npz',score=np.stack(scores))
            immutable_npz(directory/'SUMMARY.npz',score_summary=np.stack(summaries),movement_error=np.stack(movement),movement_count=np.stack(counts))
            background,index,H_length,group=block_coordinates(block)
            atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'block':block,'background':background,
                'innovation_index':index,'H_length':H_length,'dependency_group':group,'independent_fold_models_are_new_samples':False,
                'folds_as_repeated_frozen_procedures_not_independent_groups':True,'model_refits_on_synthetic':0,
                'score_methods':list(SCORE_METHODS),'auxiliary_methods':list(AUX_METHODS),'auxiliary_features':list(DETAILED),
                'horizons':list(HORIZONS),'windows':[list(w) for w in WINDOWS],
                'score_summary_metrics':['mean_probability','max_probability','fraction_score_gt_.5','fraction_score_gt_.8'],
                'threshold_status':'Fixed descriptive probes, not optimized alarm thresholds or deployment recommendations',
                'movement_error_metrics':['MSE','MAE'],'movement_domain':'Known H feature state, stride8, availablefuturetargets only',
                'score_axis_order':['fold','row','method'],'summary_axis_order':['fold','intervention','window','method','metric'],
                'movement_axis_order':['fold','feature','horizon','method','intervention','window','metric'],
                'aliases':aliases,'synthetic_features_sha256':cache.receipt['derived_features_sha256'],
                'files_sha256':{name:digest(directory/name) for name in ('SCORES.npz','SUMMARY.npz')},
                'real_observations_or_labels_read':False,'frozen_real_RP_model_artifacts_read':True,
                'full_H_baseline_normalization_separate_from_new_H60_H40':True,
                'H_lengths_and_interventions_share_innovations_not_independent':True})
        records.append({'block':block,'done_sha256':digest(directory/'DONE.json')})
    atomic(BASE/'synthetic_evaluation'/f'PART_{part}_DONE.json',{'status':'COMPLETE','timestamp':utc_now(),
        'part':part,'parts':parts,'records':records,'prediction_freeze_sha256':digest(BASE/'PREDICTION_FREEZE_NEXT8.json')})
    note('Complete fixed synthetic model application','part'+str(part)+'/'+str(parts)+'; no synthetic model fitting','B7_NORMALIZATION_AND_LINEAGE')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,choices=range(16),required=True);p.add_argument('--parts',type=int,choices=(16,),default=16)
    a=p.parse_args();main(a.part,a.parts)
