"""Exact same-time error complementarity and fixed-catalog pairwise gaps."""
from pathlib import Path
import argparse
import csv
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from numba import set_num_threads
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.data import prepare_research,admitted_npz
from src.next5.references import component
from src.next5.bootstrap import paired_summary
from src.next5.scheduler import register_job
from scripts.decompose_next5_scale_references import joint_ordering,time_groups
from src.scoring.ts_auc import ts_auc


def write_table(path,rows):
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    root=BASE/'information';lock_path=root/'POLICY_LOCK.json'
    if args.register:
        candidates=sorted(p.stem for p in (BASE/'registrations').glob('*.json'))
        names=('scripts/analyze_next5_information.py','scripts/decompose_next5_scale_references.py','src/next5/data.py','src/next5/guard.py','src/next5/bootstrap.py')
        atomic_json(lock_path,{'created_utc':utc_now(),'candidates':candidates,'sources_sha256':{name:digest(ROOT/name) for name in names},
            'scope':'All completed full candidates plus failed-screen M2 at its actual screen scope. No refit or scope substitution.',
            'pair_rule':'Every within-fold same-online-time positive-negative pair. Exact tie credit 0.5.',
            'correlation_rule':'Pearson and average-rank Spearman after within-fold/time centering; report experts separately from fixed composites.',
            'pairwise_gap_rule':'Every pair among completed full candidates using the original same 2000 draws; descriptive and not selection-adjusted.',
            'new_fits':0,'new_random_draws':False,'new_candidate_from_analysis':False},immutable=True)
        register_job({'id':'INFO001_EXACT_COMPLEMENTARITY','kind':'validation','priority':87,
            'depends_on_success':['UNC001_MATCHED_CATALOG']+['BOOT_'+cid for cid in candidates],
            'argv':['scripts/analyze_next5_information.py'],'description':'All same-time pair errors, expert/composite correlations and all catalog pairwise gaps; no fit or adaptive resampling','retry_safe':False})
        return
    lock=read_json(lock_path)
    for name,sha in lock['sources_sha256'].items():assert digest(ROOT/name)==sha
    atomic_json(root/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(lock_path),'new_fits':0},immutable=True)
    set_num_threads(2);data=prepare_research();metadata=data.rows();references={};sources=[]
    for kind in ('J102','C205','B2_augmented_J','D2_expert'):
        references[kind],proof=component(kind,'full','PRIMARY',metadata,data.guard);sources.append(proof)
    base=.8*references['J102']+.2*references['C205'];b2=.8*references['B2_augmented_J']+.2*references['C205'];d2=.9*base+.1*references['D2_expert']
    correlations=[];pairs=[];joints=[];counts=[];matrices={};credits=np.indices((3,3,3),dtype=float)/2.
    with threadpool_limits(limits=2):
        for cid in lock['candidates']:
            scope='full' if (BASE/'experiments'/cid/'full/RESULTS.json').exists() else 'screen'
            directory=BASE/'experiments'/cid/scope;result=read_json(directory/'RESULTS.json');path=directory/'OOF.npz'
            assert digest(path)==result['OOF_sha256'];values=admitted_npz(path,data.guard,'Exact ResearchPool pair geometry')
            rows=values['global_row'];np.testing.assert_array_equal(values['dataset_id'],metadata['dataset_id'][rows])
            names=['BL80','expert','composite'];predictions=[values['baseline'],values['expert'],values['prediction']]
            if scope=='full':names+=['B2','D2'];predictions +=[b2[rows],d2[rows]]
            p=np.column_stack(predictions);cross={kind:np.zeros((len(names),len(names))) for kind in ('Pearson','Spearman')};pair_total=0
            for fold in (range(5) if scope=='full' else (0,4)):
                within=np.flatnonzero(metadata['fold'][rows]==fold);fold_rows=rows[within]
                order,cuts=time_groups(metadata['time_online'][fold_rows]);probability=p[within][order];y=metadata['target'][fold_rows][order]
                progress('Exact pair complementarity',cid+' '+scope+' fold '+str(fold)+'; no pair subsampling')
                joint=joint_ordering(y,probability[:,:3],cuts).sum(axis=0);mass=int(joint.sum());pair_total+=mass
                point=np.array([float(np.sum(joint*credits[k])/mass) for k in range(3)])
                for k in range(3):assert abs(point[k]-ts_auc(y,probability[:,k],metadata['time_online'][fold_rows][order]))<1e-14
                expert_gain=credits[1]-credits[0];blend_gain=credits[2]-credits[0]
                pairs.append({'candidate':cid,'scope':scope,'fold':fold,'pairs':mass,'BL80_TS_AUC':point[0],'expert_TS_AUC':point[1],'composite_TS_AUC':point[2],
                    'composite_delta':float(point[2]-point[0]),'expert_positive_improvement_mass':float(np.sum(joint*np.maximum(expert_gain,0))/mass),
                    'expert_damage_mass':float(np.sum(joint*np.maximum(-expert_gain,0))/mass),
                    'composite_positive_improvement_mass':float(np.sum(joint*np.maximum(blend_gain,0))/mass),
                    'composite_damage_mass':float(np.sum(joint*np.maximum(-blend_gain,0))/mass),
                    'expert_rescue_unrealized_by_fixed_blend_pair_fraction':float(joint[(expert_gain>0)&(blend_gain<=0)].sum()/mass),
                    'composite_rescue_pair_fraction':float(joint[blend_gain>0].sum()/mass)})
                for a in range(3):
                    for b in range(3):
                        for c in range(3):joints.append({'candidate':cid,'scope':scope,'fold':fold,'BL80_credit':a/2,'expert_credit':b/2,'composite_credit':c/2,'pairs':int(joint[a,b,c])})
                for start,stop in zip(cuts[:-1],cuts[1:]):
                    local=probability[start:stop]
                    centered=local-local.mean(axis=0);cross['Pearson']+=centered.T@centered
                    ranks=(rankdata(local,axis=0,method='average')-.5)/(stop-start)-.5;cross['Spearman']+=ranks.T@ranks
                print('NEXT5 INFORMATION',cid,scope,fold,mass,flush=True)
            for i,left in enumerate(names):
                for j,right in enumerate(names):
                    if j<=i:continue
                    row={'candidate':cid,'scope':scope,'left':left,'right':right}
                    for kind,matrix in cross.items():
                        denominator=np.sqrt(matrix[i,i]*matrix[j,j]);row['within_fold_time_'+kind]=float(matrix[i,j]/denominator) if denominator>0 else None
                    correlations.append(row)
            counts.append({'candidate':cid,'scope':scope,'exact_same_time_pairs':pair_total,'new_fits':0})
            if scope=='full':
                path=BASE/'bootstrap'/cid/'AGGREGATE_DRAWS.npz';proof=read_json(path.parent/'RESULTS.json');assert digest(path)==proof['files_sha256'][path.name]
                with np.load(path,allow_pickle=False) as source:matrices[cid]=source['candidate_fold_scores']
    comparisons=[];catalog=sorted(matrices)
    for i,left in enumerate(catalog):
        for right in catalog[i+1:]:comparisons.append({'candidate':left,'reference':right,**paired_summary(matrices[right],matrices[left]),'selection_adjusted':False})
    tables={'EXACT_PAIR_DECOMPOSITION.csv':pairs,'EXACT_JOINT_CREDITS.csv':joints,'SCORE_CORRELATIONS.csv':correlations,
        'PAIR_COVERAGE.csv':counts,'ALL_CANDIDATE_PAIRWISE_GAPS.csv':comparisons}
    for name,rows in tables.items():write_table(root/name,rows)
    data.guard.check_state();atomic_json(root/'RESULTS.json',{'created_utc':utc_now(),'status':'PASS','candidate_count':len(counts),
        'full_candidate_pairwise_comparisons':len(comparisons),'reference_sources':sources,'exact_pair_score_parity':True,
        'new_fits':0,'new_random_draws':False,'shadow_rows':0,'seal_rows':0,'files_sha256':{name:digest(root/name) for name in tables},
        'interpretation':'Expert disagreement is not proof of useful independent information. Rescue and damage are both reported. The unrealized-rescue fraction is not an attainable oracle or a new blending proposal.'},immutable=True)
    progress('Exact information audit complete','Twenty original scopes, expert/composite pair-rescue and damage, all 171 completed-catalog pairwise gaps; no new fitted hypothesis')


if __name__=='__main__':main()
