"""Apply the frozen X-parent rule once after the basic finite menu closes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress


def main():
    family_lock=read_json(ROOT/'HYPOTHESIS_FAMILY_LOCK_NEXT5.json');results=[]
    for family in ('S','P','W','E','R','M'):
        for cid in family_lock['families'][family]['slots']:
            full=BASE/'experiments'/cid/'full/RESULTS.json';skipped=BASE/'experiments'/cid/'full/SKIPPED.json'
            assert full.exists() or skipped.exists(),'Basic finite menu must close before X-parent selection'
            if full.exists():results.append(read_json(full))
    selected={};evidence={}
    for family in ('S','P','R'):
        eligible=[r for r in results if r['family']==family and (r['mean_delta']>=.0003 or r['standalone_mean']>=.54)]
        assert eligible,'Do not create an interaction without its frozen prerequisite'
        best=sorted(eligible,key=lambda r:(-r['mean_delta'],r['candidate']))[0]
        regpath=BASE/'registrations'/(best['candidate']+'.json');reg=read_json(regpath)
        selected[family]={k:reg[k] for k in ('experiment_id','feature_module','feature_settings','feature_sha256')}
        evidence[family]={'candidate':best['candidate'],'mean_delta':best['mean_delta'],'standalone_mean':best['standalone_mean'],
            'registration_sha256':digest(regpath),'full_result_sha256':digest(BASE/'experiments'/best['candidate']/'full/RESULTS.json')}
    regpath=BASE/'references/registrations/REF_C205.json';reg=read_json(regpath)
    dep=read_json(BASE/'references/fits/PRIMARY/full/C205/RESULTS.json');assert dep['mean_TS_AUC']>=.54
    selected['D']={k:reg[k] for k in ('experiment_id','feature_module','feature_settings','feature_sha256')}
    evidence['D']={'candidate':'REF_C205','standalone_mean':dep['mean_TS_AUC'],'registration_sha256':digest(regpath),
        'full_result_sha256':digest(BASE/'references/fits/PRIMARY/full/C205/RESULTS.json')}
    # These coordinates are predetermined semantic roles, not selected by their
    # measured feature correlation or a target-dependent interaction search.
    interactions={kind:[] for kind in ('scale_persistence','scale_tail','dependence_scale')}
    for channel in ('raw','conditional'):
        for direction in ('lower','upper'):
            for window,stat in ((32,'slow64'),(128,'positive_confirmation')):
                interactions['scale_persistence'].append({'left':f'next5_student5_{channel}_w{window}_{direction}',
                    'right':f'next5_two_timescale_{channel}_{direction}_{stat}','left_scale':float(window),'right_scale':4.})
            for window in (32,128):
                interactions['scale_tail'].append({'left':f'next5_student5_{channel}_w{window}_{direction}',
                    'right':f'next5_energy_pit_{channel}_{direction}_ewma{window}','left_scale':float(window),'right_scale':4.})
    for channel in range(4):
        for window,half in ((32,32),(128,8)):
            interactions['dependence_scale'].append({'left':f'next5_student5_conditional_w{window}_total',
                'right':f'next2_complement__dependence_{channel}_ewma{half}','left_scale':float(window),'right_scale':4.})
    # The selected parents must actually expose the semantic coordinates. Any
    # missing coordinate closes X rather than silently substituting a search.
    from src.next5.model import feature_module
    settings={}
    for kind,second in (('scale_persistence','P'),('scale_tail','R'),('dependence_scale','D')):
        parents=[selected['S'],selected[second]]
        names=[feature_module(p['feature_module']).feature_names(p['feature_settings']) for p in parents]
        for pair in interactions[kind]:assert pair['left'] in names[0] and pair['right'] in names[1]
        settings[kind]={'version':'next5_cross_family_v1','kind':kind,'parents':parents,'interactions':interactions[kind]}
    atomic_json(BASE/'locks/X_PARENT_SELECTION.json',{'created_utc':utc_now(),'status':'FROZEN_BEFORE_X_SCORE','selected':evidence,
        'rule':'Highest full mean delta among prerequisite-qualified family members; lexicographic ID breaks ties.',
        'basic_full_count':len(results),'failed_screen_count':17-len(results),'old_full_DEV_scores_used':False,
        'semantic_coordinate_rule':'Eight fixed bounded products per slot; no all-pairs, label-driven coordinate or weight search.',
        'conditional_selection_limitation':'Parents selected using ResearchPool outcomes; prospective Shadow remains unopened confirmation only.'},immutable=True)
    atomic_json(BASE/'locks/X_SETTINGS_LOCK.json',{'created_utc':utc_now(),'status':'FROZEN_BEFORE_X_SCORE','settings':settings,
        'parent_selection_sha256':digest(BASE/'locks/X_PARENT_SELECTION.json')},immutable=True)
    progress('Cross-family parents frozen',str({key:value['candidate'] for key,value in evidence.items()})+'; exactly eight semantic bounded interactions per slot')


if __name__=='__main__':main()
