"""The three distinct already-declared bounded-window scale questions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.registration import register_family


def main():
    entries=[]
    for cid,kind,question,change in [
      ('W1_RECENT_SCALE_RATIO','scale_ratio','Do bounded recent energy ratios detect changes without indefinitely accumulated evidence?',
       '24 features: H-relative log energy mean at windows16/32/64/128, separately positive lower/upper branches, raw and conditional channels.'),
      ('W2_TRIMMED_RECENT_ENERGY','trimmed_energy','Can H-capped recent energy and recent log energy reduce outlier sensitivity without losing scale changes?',
       '16 features: window means of energy capped at H-only q95 and mean log(energy+1e-4), relative to H references; same four windows/two channels.'),
      ('W3_MULTIWINDOW_DISAGREEMENT','disagreement','Does disagreement and agreement across fixed short/long windows isolate onset and persistence?',
       '16 features: log-energy ratio difference, absolute difference, lower agreement and upper agreement for 16vs128 and32vs64 in both channels.')]:
        entries.append({'experiment_id':cid,'kind':kind,'parent':['ResearchPool_BL80','H-only NEXT5 innovation channels'],
          'scientific_question':question,'mechanism':'Bounded causal local scale and fixed cross-window contrasts.',
          'exact_feature_change':change,'implementation_constants':{'energy_floor':1e-8,'log_energy_epsilon':1e-4,'H_channel_cap':12.,'state_window_limit':128},
          'expected_risk':['Short windows increase estimator variance','Long windows retain transient shocks','ARCH state can retain influence beyond a rolling window','Trimming can discard real amplitude information']})
    register_family(family='W',module_name='src.next5.windowed',version='next5_windowed_v1',entries=entries,
        test_job='W000_FEATURE_CONTRACTS',test_xml='artifacts/next5/validation/W_FEATURE_CONTRACTS.xml',
        caller='scripts/register_next5_windowed.py',priority=50)


if __name__=='__main__':main()
