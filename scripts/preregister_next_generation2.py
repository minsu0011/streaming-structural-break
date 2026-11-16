"""Two forensic hypotheses: H-null calibration and AR coefficient score tests."""
from pathlib import Path
from dataclasses import asdict
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.engine import EngineConfig
from src.next.score_evidence import ScoreConfig,SCORE_NAMES
from src.next.extensions import extension_hash
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    path=ROOT/'configs/next_generation2_candidates.json'
    if path.exists():
        print('Generation 2 already frozen')
        return
    candidates=[]
    core=['I','S1','S2','S3','S4']
    def add(eid,parent,hypothesis,change,extension,groups,base='none',config=None,learner='M4_lightgbm',keep=None):
        item={'experiment_id':eid,'parent_id':parent,'generation':2,'hypothesis':hypothesis,'change':change,
            'expected_effect':'Reduce H-regime nuisance or directly detect conditional AR coefficient changes',
            'risks':'H-fit covariance and calibration can be optimistic; short-context signal loss; fixed DEV screening uncertainty',
            'engine_config':asdict(config or EngineConfig()),'groups':groups,'base':base,'learner':learner,
            'weighting':'W3','params':{},'phase':eid.split('_')[1],'extension':extension,
            'extension_frozen_hash':extension_hash(extension)}
        if base=='arch8':
            item['base_family_selection']='ABCD'
        if keep is not None:
            item['keep_names']=keep
        candidates.append(item)
    def hcal(config=None):
        return {'kind':'historical_null','settings':{'engine_config':asdict(config or EngineConfig())}}
    hypothesis='IID analytic scaling leaves temporal and heavy-tail nuisance across series; historical-null median/MAD scaling may improve cross-series comparability.'
    add('C001_HCAL_CORE','N001_SEQ_AR8_SD_CORE',hypothesis,'Only H-null median/MAD feature scaling is added',hcal(),core)
    add('C002_HCAL_SCALE','N020_STANDALONE_S2',hypothesis,'Only H-null scaling of the instantaneous and scale feature family',hcal(),['I','S2'])
    p12=EngineConfig(order=12)
    add('C003_HCAL_AR12','N005_WHITEN_AR12',hypothesis,'Only H-null scaling on the fixed AR12 core',hcal(p12),core,config=p12)
    gaussian=EngineConfig(normalization='gaussian')
    add('C004_HCAL_GAUSSIAN','N012_NORM_GAUSSIAN',hypothesis,'Only H-null scaling on the Gaussian innovation core',hcal(gaussian),core,config=gaussian)
    for eid,parent,groups in [('C005_HCAL_ARCH_CORE','R036_ARCH_ADD_CORE',core),('C006_HCAL_ARCH_DEPENDENCE','R031_ARCH_ADD_S3',['S3']),
                              ('C007_HCAL_ARCH_GLR','R033_ARCH_ADD_G',['G']),('C008_HCAL_ARCH_BAYES','R034_ARCH_ADD_B',['B'])]:
        add(eid,parent,hypothesis,'Only the appended new statistics receive H-null calibration; ARCH8 ABCD base remains unchanged',hcal(),groups,base='arch8')
    def score(order=8,covariance='full'):
        return {'kind':'ar_score','settings':asdict(ScoreConfig(order=order,covariance=covariance))}
    hypothesis='For an AR coefficient change, innovation times past original regressors is a direct conditional-mean score; innovation-times-innovation autocorrelation may discard this direction.'
    add('S001_ARSCORE_STANDALONE','N021_STANDALONE_S3',hypothesis,'Replace the dependence bank with the fixed H-covariance-normalized AR score vector',score(),['SCORE'])
    add('S002_ARSCORE_DIAGONAL','S001_ARSCORE_STANDALONE',hypothesis,'Only historical score covariance normalization changes from full to diagonal',score(covariance='diagonal'),['SCORE'])
    add('S003_ARSCORE_ARCH','R013_ARCH_REFERENCE',hypothesis,'Append only the 19 AR coefficient score summaries to frozen ARCH8 ABCD',score(),['SCORE'],base='arch8')
    add('S004_ARSCORE_ARCH_DIAGONAL','S003_ARSCORE_ARCH',hypothesis,'Only score covariance normalization changes to diagonal',score(covariance='diagonal'),['SCORE'],base='arch8')
    for order in [4,12]:
        add(f'S00{5 if order==4 else 6}_ARSCORE_ARCH_P{order}','S003_ARSCORE_ARCH',hypothesis,
            f'Only auxiliary AR score order changes from 8 to {order}',score(order),['SCORE'],base='arch8',config=EngineConfig(order=order))
    add('S007_ARSCORE_ARCH_HISTGB','S003_ARSCORE_ARCH',hypothesis,'Only the small tree learner changes to HistGB',score(),['SCORE'],base='arch8',learner='M3_histgb')
    for eid,label,keep in [('S008_ARSCORE_ARCH_EWMA','Keep only instantaneous and EWMA AR score summaries',[n for n in SCORE_NAMES if 'instant' in n or 'ewma' in n]),
                           ('S009_ARSCORE_ARCH_NO_BAYES','Remove only the AR-score Bayes mixture summaries',[n for n in SCORE_NAMES if 'bayes' not in n]),
                           ('S010_ARSCORE_ARCH_NO_GLR','Remove only the AR-score GLR summaries',[n for n in SCORE_NAMES if 'glr' not in n])]:
        add(eid,'S003_ARSCORE_ARCH',hypothesis,label,score(),['SCORE'],base='arch8',keep=['ar_score__'+n for n in keep])
    write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':2,
        'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
        'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'forensic_inputs':['CURRENT_TOP_FORENSIC.md','artifacts/next/historical_holdout/SUMMARY.csv','MODEL_RESULTS_NEXT.csv'],
        'thresholds_splits_seeds_unchanged':True,
        'historical_null_policy':{'replay':'full H through a clone of its H-fitted initial state','burn_in':160,'center':'median',
            'scale':'1.4826 * MAD, floor 0.05','output_clip':12,'excluded':'elapsed time, hypothetical maximizing age, persistent memory'},
        'ar_score_policy':{'regressors':'lagged original robust-standardized input, not lagged innovations','covariance':'empirical H score covariance plus 1% trace ridge',
            'online_refitting':False,'online_complexity':'O(p^2 + number_of_fixed_ages*p), p <=12; AR filtering itself O(p)',
            'null_distribution_claim':'No calibrated chi-square, posterior or e-value claim; empirical ranking evidence only'},
        'candidates':candidates})
    print('FROZEN GENERATION 2',len(candidates),'candidates')


if __name__=='__main__':
    main()
