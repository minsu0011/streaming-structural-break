"""Apply the prospective complement gates once after all parent diagnostics close."""
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress


def main():
    menu_path=BASE/'families/F_MENU_LOCK.json';menu=read_json(menu_path);gate=menu['third_expert_admission']
    records=[];evidence={};admitted=[];replacements=[]
    for family in menu['third_expert_pool_families']:
        family_menu=read_json(BASE/'families'/(family+'_MENU_LOCK.json'))
        for candidate in family_menu['candidate_ids']:
            root=BASE/'experiments'/candidate;screen=read_json(root/'screen/RESULTS.json')
            row={'candidate':candidate,'family':family,'eligible':False,'reasons':[]}
            if not screen['screen_pass']:
                row['reasons'].append('FAILED_SCREEN');records.append(row);continue
            path=root/'full/RESULTS.json';full=read_json(path);evidence[path.relative_to(ROOT).as_posix()]=digest(path)
            assert full['status']=='COMPLETE'
            path=BASE/'bootstrap'/candidate/'RESULTS.json';boot=read_json(path);evidence[path.relative_to(ROOT).as_posix()]=digest(path)
            assert boot['status']=='PASS';paired=boot['paired_all_time']
            row.update(full_mean_delta=full['mean_delta'],standalone_mean=full['standalone_mean'],
                paired_ci_lower=paired['ci_2_5'],paired_probability_positive=paired['probability_delta_positive'])
            if full['mean_delta']<gate['full_mean_delta_min']:row['reasons'].append('FIXED_BLEND_GAIN_BELOW_0003')
            if full['standalone_mean']<gate['standalone_mean_min']:row['reasons'].append('STANDALONE_BELOW_054')
            if paired['ci_2_5']<=gate['paired_ci_lower_strict_min']:row['reasons'].append('PAIRED_CI_NOT_STRICTLY_POSITIVE')
            if paired['probability_delta_positive']<gate['paired_probability_min']:row['reasons'].append('PAIRED_PROBABILITY_BELOW_09')
            forensic=BASE/'forensic'/candidate/'RESULTS.json'
            required=full['mean_delta']>=-.0005 or family in ('B','D')
            if required:
                proof=read_json(forensic);assert proof['status']=='PASS';evidence[forensic.relative_to(ROOT).as_posix()]=digest(forensic)
                dp=forensic.parent/'STANDALONE_DIVERSITY.csv';assert digest(dp)==proof['files_sha256'][dp.name]
                diversity=pd.read_csv(dp);correlations=diversity.groupby('reference').within_time_spearman.mean().to_dict()
                row['mean_within_time_spearman']=correlations
                assert set(correlations)==set(gate['correlation_references'])
                if any(not pd.notna(correlations[k]) or correlations[k]>=gate['mean_within_time_rank_correlation_max_strict'] for k in gate['correlation_references']):
                    row['reasons'].append('WITHIN_TIME_CORRELATION_AT_LEAST_098_OR_UNDEFINED')
            elif not row['reasons']:
                raise RuntimeError('An otherwise admitted expert lacks the required fixed forensic result')
            row['eligible']=not row['reasons'];records.append(row)
            if row['eligible']:admitted.append(row)
    for candidate in menu['replacement_pool']:
        root=BASE/'experiments'/candidate;screen=read_json(root/'screen/RESULTS.json')
        if not screen['screen_pass']:continue
        path=root/'full/RESULTS.json';full=read_json(path);evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        if full['performance_numeric_admission']:replacements.append({'candidate':candidate,'full_mean_delta':full['mean_delta']})
    third=sorted(admitted,key=lambda r:(-r['full_mean_delta'],r['candidate']))[0]['candidate'] if admitted else None
    replacement=sorted(replacements,key=lambda r:(-r['full_mean_delta'],r['candidate']))[0]['candidate'] if replacements else None
    ready=third is not None and replacement is not None
    result={'created_utc':utc_now(),'family_menu_sha256':digest(menu_path),'third_expert_pool':records,
        'replacement_admitted_pool':replacements,'selected_third_expert':third,'selected_augmented_J_composite':replacement,
        'F1_disposition':'DEDUPLICATE_REUSE_'+third if third else 'NO_ADMITTED_INDEPENDENT_THIRD_EXPERT',
        'F2_disposition':'READY_FOR_EXACT_SOURCE_REGISTRATION_BEFORE_SCORE' if ready else 'CONDITIONAL_SKIP_NO_ADMITTED_COMPONENT_PAIR',
        'F2_recipe':menu['F2_recipe'],'new_F_scores':0,'new_fits':0,'new_full_candidates':0,'promotion':False,
        'evidence_sha256':evidence,'no_fallback_or_weight_search':True,
        'scope':'One application of the pre-frozen admission menu; closed rejected experts are not replaced with a merely positive point estimate.'}
    atomic_json(BASE/'families/F_SELECTION.json',result,immutable=True)
    progress('Conditional expert-combination family closed',result['F2_disposition'])
    print('NEXT4 FAMILY F SELECTION',result['F2_disposition'],'third',third,'replacement',replacement,flush=True)


if __name__=='__main__':main()
