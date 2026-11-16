"""Explain the closed objective/context/state menus without another fit."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.diagnostics import diversity
from src.next4.frozen import fold_predictions
from src.next4.model import FeatureModel
from src.next4.historical_context import CONTEXT_NAMES
from src.next.data import prepare_dev


def tree_splits(model,fold):
    rows=[]
    def walk(node,tree,depth):
        if 'split_feature' not in node:return
        column=int(model.columns[int(node['split_feature'])]);name=model.names[column]
        rows.append({'fold':fold,'tree':tree,'depth':depth,'feature':name,'is_H_context':column>=77,
            'split_gain':float(node['split_gain']),'training_internal_count':int(node['internal_count'])})
        walk(node['left_child'],tree,depth+1);walk(node['right_child'],tree,depth+1)
    for i,tree in enumerate(model.estimator.booster_.dump_model()['tree_info']):walk(tree['tree_structure'],i,0)
    return rows


def main():
    candidates=['E1_NESTED_PAIRWISE_LINEAR','E2_NESTED_PAIRWISE_INTERACTIONS',
        'G1_H_CONTEXT_BASE','G2_H_CONTEXT_NESTED_PAIRWISE','H1_FIXED_LINEAR_STATE_READOUT']
    out=BASE/'closed_menu_diagnostics';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'candidates':candidates,'source_sha256':digest(Path(__file__)),
        'questions':['How much G1 split gain and root use came from static H context?',
            'Do pairwise optimizers converge and which fixed expert weights do they learn?',
            'What is standalone diversity for failed-screen or negative-full candidates omitted from full forensics?'],
        'no_counterfactual_ablation_scores':True,'new_fits':0,'no_candidate_selection_or_parameter_changes':True},immutable=True)
    data=prepare_dev(ROOT);metadata=data.rows();all_diversity=[];objectives=[];splits=[];evidence={}
    for candidate in candidates:
        directory=BASE/'experiments'/candidate/'full'
        if not (directory/'RESULTS.json').exists():directory=directory.parent/'screen'
        full=read_json(directory/'RESULTS.json');scope=full['scope'];evidence[str((directory/'RESULTS.json').relative_to(ROOT))]=digest(directory/'RESULTS.json')
        for fr in full['fold_records']:
            fold=fr['fold'];rows=np.load(directory/f'fold_{fold}_global_rows.npy',allow_pickle=False)
            for name in (f'fold_{fold}_expert.npy',f'fold_{fold}_global_rows.npy',f'fold_{fold}_expert.joblib'):
                assert digest(directory/name)==fr['files_sha256'][name]
            expert=np.load(directory/f'fold_{fold}_expert.npy',allow_pickle=False)
            frozen,_=fold_predictions(data,fold,scope,rows)
            role='learned_combination' if candidate.startswith(('E','G2')) else 'augmented_J102' if candidate.startswith('G1') else 'independent_state_readout'
            for key,ref in frozen.items():
                all_diversity.append({'candidate':candidate,'scope':scope,'fold':fold,'expert_role':role,'reference':key,
                    **diversity(expert,ref,metadata['time_online'][rows],metadata['target'][rows])})
            if candidate.startswith(('E','G2')):
                detail=fr['training'];weights=fr['weights']
                objectives.append({'candidate':candidate,'scope':scope,'fold':fold,'initial_pair_loss':detail['initial_pair_loss'],
                    'final_pair_loss':detail['final_pair_loss'],'optimizer_iterations':detail['optimizer_iterations'],
                    'optimizer_success':detail['optimizer_success'],'J_weight':weights[0],'C_weight':weights[1],
                    'static_C_share_if_no_context':weights[1]/(weights[0]+weights[1]),'parameter_count':len(weights),
                    'outer_validation_delta':fr['delta']})
            if candidate=='G1_H_CONTEXT_BASE':splits.extend(tree_splits(FeatureModel.load(directory/f'fold_{fold}_expert.joblib'),fold))
            progress('Closed-menu causal diagnosis',candidate+' '+scope+' fold '+str(fold)+' diversity and fixed model structure')
            print('NEXT4 CLOSED MENU',candidate,scope,fold,flush=True)
    split_frame=pd.DataFrame(splits)
    grouped=split_frame.groupby(['fold','feature','is_H_context'],as_index=False).agg(split_gain=('split_gain','sum'),split_count=('split_gain','size'))
    structure=[]
    for fold,frame in split_frame.groupby('fold'):
        total=frame.split_gain.sum();context=frame[frame.is_H_context];root=frame[frame.depth==0]
        structure.append({'fold':int(fold),'H_context_gain_fraction':float(context.split_gain.sum()/total),
            'H_context_split_fraction':float(len(context)/len(frame)),
            'H_context_root_tree_fraction':float(root.is_H_context.mean()),'trees':int(root.tree.nunique())})
    pd.DataFrame(all_diversity).to_csv(out/'DIVERSITY.csv',index=False)
    pd.DataFrame(objectives).to_csv(out/'PAIRWISE_FIT_STRUCTURE.csv',index=False)
    grouped.to_csv(out/'G1_FEATURE_SPLIT_GAINS.csv',index=False)
    pd.DataFrame(structure).to_csv(out/'G1_CONTEXT_USE.csv',index=False)
    data.guard.check_state()
    files=['DIVERSITY.csv','PAIRWISE_FIT_STRUCTURE.csv','G1_FEATURE_SPLIT_GAINS.csv','G1_CONTEXT_USE.csv']
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'new_fits':0,'seal_rows':0,
        'G1_context_use':structure,'paired_fit_structure':objectives,'evidence_sha256':evidence,
        'scope':'Descriptive post-result diagnosis, no new predictors, feature ablations, scores under altered inputs or tuning. Split gains describe the fitted learner and do not establish causal feature value.',
        'files_sha256':{name:digest(out/name) for name in files}},immutable=True)
    print('NEXT4 CLOSED MENU DIAGNOSTICS COMPLETE',flush=True)


if __name__=='__main__':main()
