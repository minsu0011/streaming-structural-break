from pathlib import Path
import sys,ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from src.next6_extension.common import *
from src.next6_extension.categorical_controls import generated
from src.next6_extension.factorized import fit_pair as old_pair
from src.next6_extension.factorized_coherent import fit_pair as coherent_pair
from summarize_next6_extension import rows

def main():
    assert (BASE/'LEGACY_MEMORY_DETAIL_DONE.json').exists();out=[];feature=[]
    for case in ('O1','OB16'):
        rr=rows(BASE/f'legacy_memory/{case}/SERIES_ATTRIBUTION.csv');groups=sorted(set(r['group'] for r in rr));windows=sorted(set(int(r['relative_start']) for r in rr));index={(int(r['seed']),int(r['fold']),int(r['relative_start']),r['group']):r for r in rr};rng=np.random.default_rng([2026091310,case=='OB16']);weight=rng.multinomial(64,np.full(64,1/64),size=2000)/64
        for low in windows:
            for group in groups:
                for metric in ('mean_signed_contribution','mean_absolute_contribution','conditional_J102_effect'):
                    values=np.array([np.mean([float(index[s,f,low,group][metric]) for f in range(5)]) for s in range(64)]);lo,hi=np.quantile(weight@values,[.025,.975]);out.append({'scenario':case,'relative_start':low,'relative_end':index[0,0,low,group]['relative_end'],'group':group,'metric':metric,'mean':float(values.mean()),'ci_lo':lo,'ci_hi':hi,'whole_seed_replicates':2000,'independent_seeds':64,'folds_carried_together':5})
        rr=rows(BASE/f'legacy_memory/{case}/FEATURE_DIFFERENCES.csv')
        for low in windows:
            for column in range(77):
                pp=[r for r in rr if int(r['relative_start'])==low and int(r['column'])==column];assert len(pp)==64;feature.append({'scenario':case,'relative_start':low,'column':column,'name':pp[0]['name'],'mean_absolute_difference':float(np.mean([float(r['mean_absolute_feature_difference']) for r in pp])),'maximum_absolute_difference':max(float(r['max_absolute_feature_difference']) for r in pp)})
    csv_write('LEGACY_MEMORY_PRIMITIVE_SUMMARY.csv',out);csv_write('LEGACY_FEATURE_MEMORY_SUMMARY.csv',feature)
    rejection=[];edges=[]
    for fold in range(5):
        rr=rows(BASE/f'coherent_graphs/fold_{fold}/INNER_DISCOVERY.csv');keys=sorted(set((int(r['source']),int(r['target'])) for r in rr));counts={'candidates':len(keys),'both_halves_positive_gain':0,'both_halves_gain_at_least_001':0,'both_halves_positive_fraction_at_least_55':0,'both_all_gates':0,'both_full_context_support':0}
        for source,target in keys:
            pp=[r for r in rr if int(r['source'])==source and int(r['target'])==target];assert len(pp)==2;gain=[float(r['balanced_full_vs_null_gain']) for r in pp];fraction=[float(r['balanced_positive_series_fraction']) for r in pp];backoff=[len(ast.literal_eval(r['whole_ownstate_backoff'])) for r in pp];counts['both_halves_positive_gain']+=min(gain)>0;counts['both_halves_gain_at_least_001']+=min(gain)>=.001;counts['both_halves_positive_fraction_at_least_55']+=min(fraction)>=.55;counts['both_all_gates']+=all(r['eligible_half']=='True' for r in pp);counts['both_full_context_support']+=max(backoff)==0
            edges.append({'fold':fold,'source':source,'target':target,'minimum_half_gain':min(gain),'maximum_half_gain':max(gain),'minimum_half_positive_fraction':min(fraction),'maximum_backed_off_own_strata':max(backoff),'selected':all(r['eligible_half']=='True' for r in pp)})
        rejection.append({'fold':fold,**counts,'no_outer_selection':True})
    csv_write('COHERENT_GRAPH_REJECTION_SUMMARY.csv',rejection);csv_write('COHERENT_GRAPH_CANDIDATE_DIAGNOSTIC.csv',edges)
    worst=read_json(BASE/'SPARSE_NULL_COHERENCE_COVERAGE.json')['worst'];a,b,p,ids=generated(worst['groups'],32,[20260913,910,worst['seed']],.6,'central80percent','marginal_shift');_,before,support,_=old_pair(a,b,p,ids,0,4);_,after,_,receipt=coherent_pair(a,b,p,ids,0,4)
    def differences(table):
        out=[]
        for own in range(3):
            row=[]
            for i in range(2):
                for j in range(2):
                    odds=table[:,own,i,j]*table[:,own,i+1,j+1]/(table[:,own,i,j+1]*table[:,own,i+1,j]);row.append(float(np.log(odds[1])-np.log(odds[0])))
            out.append(row)
        return out
    save('SPARSE_NULL_REPAIR_VISUAL_DATA.json',{'groups':worst['groups'],'seed':worst['seed'],'before_log_odds_phase_difference':differences(before),'after_log_odds_phase_difference':differences(after),'raw_support':support.tolist(),'whole_ownstate_backoff':receipt['whole_ownstate_backoff'],'not_RP_data':True})
    save('LEGACY_AND_GRAPH_BOTTLENECK_SUMMARIES_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'graph_candidate_pairs_per_fold':52,'legacy_primitive_source_sha256':digest(ROOT/'src/next/channel_evidence.py'),'original_models_unchanged':True})
    print(rejection)
if __name__=='__main__':main()
