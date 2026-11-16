"""Freeze diagnostics before new synthetic scores; bind each existing fitted model."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job
from src.next4.synthetic import sources


def main():
    suite_path=BASE/'diagnostic_policy/SYNTHETIC_POLICY.json'
    if not suite_path.exists():
        xml=BASE/'maintenance/SYNTHETIC_UNIT_TESTS.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
        assert sum(int(s.get('tests',0)) for s in suites)==6 and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
        old=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
        atomic_json(suite_path,{'status':'FROZEN_BEFORE_NEXT4_FITTED_SYNTHETIC_SCORES','created_utc':utc_now(),
            'original_generator_policy_sha256':digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
            'sources_sha256':sources(),'unit_test_xml_sha256':digest(xml),
            'candidate_rule':'Every Family A/B/C/D/H candidate after full completion or failed screen, using fixed original fold 0. B2 instead uses its single separately nominated all-DEV diagnostic fit. Never fit a failed screen on full DEV.',
            'scope_rule':'Candidate and J102/C205/BL80 use identical all-DEV or original full/screen fold-0 training scope. Fold-0 results are exploratory fitted-model diagnostics, not all-DEV deployment claims.',
            'fold0_sizes':{'calibration':512,'evaluation':256,'long_null':256},
            'all_dev_sizes':{'calibration':1000,'evaluation':512,'long_null':512},
            'counterexamples':[s['id'] for s in old['scenarios']],
            'additional_counterexamples':['CB_N3_070','CB_N3_050','OB16'],
            'additional_definition':'Unchanged crossed-background generator partition 3 for Student-t/ARCH N3 variance factors 0.7/0.5; OB16 uses original O1 clean paired paths plus 16 alternating +/-8 stationary-SD impulses, no lasting DGP change.',
            'historical_length':old['historical_length'],'tau_cycle':old['tau_cycle'],
            'long_null_regimes':['N0','N1','N2','N3'],'long_null_horizons':[1024,4096,16384],
            'threshold_alphas':[.01,.05],'score_quantiles':[.5,.9,.95,.99,.999],
            'calibration':'Model-specific order statistic ceil((n+1)*(1-alpha)) of independent partition-0 1024-point null maxima with strict > alarms. Fixed across horizons; no uniform anytime guarantee.',
            'paired_bootstrap_replicates':2000,'paired_bootstrap_seed':20260908,
            'bootstrap_unit':'Paired generating seed: each clean control and its altered trajectory are resampled together. Same fixed draw matrix across scenarios of equal size; conditional fixed-model uncertainty.',
            'limits':'No adaptive scenario, severity, seed, threshold, sample-size or model changes after scores. All synthetic effects diagnostic; real primary and SCALE CI gates remain authoritative.',
            'stop_rule':'One fixed complete diagnostic suite per nominated model; keep partial failures and never overwrite completed cohorts. No model tuning from synthetic outcomes.',
            'new_fits':0,'seal_rows':0},immutable=True)
    else:assert read_json(suite_path)['sources_sha256']==sources()
    for rp in sorted((BASE/'registrations').glob('*.json')):
        reg=read_json(rp);family=reg['family'];candidate=reg['experiment_id']
        if family not in ('A','B','C','D','H'):continue
        if candidate=='B2_LOG_ENERGY_SEMITAIL':
            directory=BASE/'refits'/candidate/'diagnostic_all_dev_v1';scope='all_dev';model_path=directory/'model.joblib'
            dependency='B501_ALL_DEV_DIAGNOSTIC_REFIT';receipt=directory/'RESULTS.json'
            if not receipt.exists():continue
        else:
            directory=BASE/'experiments'/candidate/'full';scope='full_fold0'
            if not (directory/'RESULTS.json').exists():
                directory=BASE/'experiments'/candidate/'screen';scope='screen_fold0'
                if not (directory/'RESULTS.json').exists() or read_json(directory/'RESULTS.json')['screen_pass']:continue
            receipt=directory/'RESULTS.json';model_path=directory/'fold_0_expert.joblib'
            menu=read_json(BASE/'families'/(family+'_MENU_LOCK.json'));index=menu['candidate_ids'].index(candidate)
            dependency=f'{family}{index+(101 if scope=="full_fold0" else 1):03d}_'+('FULL' if scope=='full_fold0' else 'SCREEN')
        identifier=candidate+'_'+scope;nom_path=BASE/'synthetic_nominations'/(identifier+'.json')
        if nom_path.exists():continue
        if scope=='all_dev':assert read_json(receipt)['status']=='PASS'
        else:
            fr=read_json(directory/'fold_0.json');assert digest(model_path)==fr['files_sha256'][model_path.name]
        atomic_json(nom_path,{'id':identifier,'created_utc':utc_now(),'candidate':candidate,'family':family,
            'training_scope':scope,'model_path':model_path.relative_to(ROOT).as_posix(),'model_sha256':digest(model_path),
            'registration_sha256':digest(rp),'training_result_sha256':digest(receipt),'suite_sha256':digest(suite_path),
            'new_fits':0,'new_candidate_identity':False,'promotion':False},immutable=True)
        register_job({'id':'SYN_'+identifier,'kind':'validation','family':family,'candidate_id':candidate,
            'priority':150 if scope=='all_dev' else 160,'depends_on_success':[dependency],
            'argv':['scripts/stress_next4_candidate.py','--nomination',identifier],
            'description':'Fixed matched-scope fitted-model counterexamples and long-null, no fitting',
            'retry_safe':False,'nomination_sha256':digest(nom_path)})
        print('NEXT4 SYNTHETIC QUEUED',identifier,flush=True)
    progress('Fitted synthetic diagnostic suites registered','Finite unchanged DGPs, matched training scopes, independent alarm calibration; no extra candidates')


if __name__=='__main__':main()
