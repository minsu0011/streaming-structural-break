"""Freeze diagnostic model/seed/group choices before fitted synthetic scores."""
from pathlib import Path
import sys
import subprocess
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job


def main():
    assert not list((BASE/'synthetic/candidates').glob('*/START.json'))
    xml=BASE/'maintenance/OUTLIER_ATTRIBUTION_UNIT_TESTS.xml';assert not xml.exists()
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_next4_outlier_attribution.py','--junitxml='+str(xml)],cwd=ROOT,check=True)
    suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==3 and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    candidates={'B1_SPLIT_ENERGY_BRANCHES':'full_fold0','B2_LOG_ENERGY_SEMITAIL':'all_dev',
        'D2_DISCRETE_SCALE_MIXTURE':'full_fold0','D3_HAZARD_MIXTURE':'full_fold0'}
    nominations={candidate+'_'+scope:digest(BASE/'synthetic_nominations'/(candidate+'_'+scope+'.json')) for candidate,scope in candidates.items()}
    names=['src/next4/outlier_attribution.py','scripts/attribute_next4_outlier.py','src/next4/synthetic.py','src/next4/model.py']
    path=BASE/'diagnostic_policy/OUTLIER_ATTRIBUTION_POLICY.json'
    atomic_json(path,{'status':'FROZEN_BEFORE_NEXT4_FITTED_SYNTHETIC_SCORES','created_utc':utc_now(),
        'nominations':nominations,'sources_sha256':{name:digest(ROOT/name) for name in names},
        'unit_test_sha256':digest(xml),'synthetic_policy_sha256':digest(BASE/'diagnostic_policy/SYNTHETIC_POLICY.json'),
        'scenarios':{'O1':1,'OB16':16},'paired_seed_count':128,'post_intervention_window':256,
        'group_rule':'Seven B groups (old ARCH, two old serial groups, four new branches); five D2 direction/window groups; three D3 hazards plus contrasts. Expected margin, unchanged frozen partner and rounding residual are separate.',
        'no_new_fits_or_prediction_recipes':True,'no_new_generating_seed':True,'no_real_data_perturbations':True,
        'stop_rule':'One descriptive attribution pass per nominated fixed model and both outlier scenarios. No outcome-conditioned path selection, feature repair, weight tuning, group dropping or promotion override.'},immutable=True)
    for identifier in nominations:
        register_job({'id':'AT_'+identifier,'kind':'validation','priority':157,'depends_on_success':['SYN_'+identifier],
            'argv':['scripts/attribute_next4_outlier.py','--nomination',identifier],
            'description':'Fixed paired outlier Tree-SHAP margin decomposition, unchanged model and synthetic seeds',
            'retry_safe':False,'policy_sha256':digest(path)})
    progress('Outlier attribution policy frozen','Three algebra/group oracle tests passed; four models, two fixed cases and first 128 original seeds only')
    print('NEXT4 OUTLIER ATTRIBUTION PREPARED',flush=True)


if __name__=='__main__':main()
