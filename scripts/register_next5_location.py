"""Two predeclared independent-signal questions with a strict weak-expert stop."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.registration import register_family


def main():
    entries=[]
    for cid,kind,question,change in [
      ('M1_ROBUST_LOCATION_LIKELIHOOD','student_location','Can robust signed t5 location evidence contribute a signal distinct from scale/dependence?',
       '16 features: t5 log density ratio for fixed shifts -0.75/+0.75 with unit null variance, H-only centering/scaling, current and EWMA8/32/128 on raw/conditional innovations.'),
      ('M2_SIGN_RUN_EVIDENCE','sign_run','Can median-relative sign imbalance and run persistence contribute without amplitude evidence?',
       '12 features: H-median signs centered/scaled using H, current and EWMA8/32/128, signed log run and signed geometric-run evidence using an H-only smoothed same-sign probability. Reset per series.')]:
        entries.append({'experiment_id':cid,'kind':kind,'parent':['ResearchPool_BL80','H-only NEXT5 innovation channels'],
          'scientific_question':question,'mechanism':'Bounded robust signed evidence and sign persistence as a low-budget location complement.',
          'exact_feature_change':change,'implementation_constants':{'t_df':5.,'shifts':[-.75,.75],'H_scale_floor':.1,'run_probability_smoothing':1.,'standalone_screen_min':.52},
          'expected_risk':['Location is weak in historical real DEV','Signs can also respond to dependence changes','Real mean-dominant diagnostic support may be insufficient','Close if fixed screen standalone <0.52; no extra mean variants']})
    register_family(family='M',module_name='src.next5.location',version='next5_location_v1',entries=entries,
        test_job='M000_FEATURE_CONTRACTS',test_xml='artifacts/next5/validation/M_FEATURE_CONTRACTS.xml',
        caller='scripts/register_next5_location.py',priority=60)


if __name__=='__main__':main()
