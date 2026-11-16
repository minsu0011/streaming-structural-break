"""Two fixed evidence-accumulation questions, with explicit theory limits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.registration import register_family


def main():
    entries=[]
    for cid,kind,question,change in [
      ('E1_SCALE_MIXTURE_EVIDENCE','cumulative','Does an unreset fixed scale likelihood mixture provide useful ranking evidence despite H-estimated nulls?',
       '12 features over raw/conditional channels: lower/upper/total cumulative logmean likelihood, direction contrast, alternative entropy and posterior log variance. Uniform prior over fixed four variance alternatives.'),
      ('E2_SPARSE_START_MIXTURE','sparse_starts','Does a fixed mixture over sparse recent start ages recover changes without absorbing hazard pressure?',
       '18 features: lower/upper/total uniform logmean likelihood across fixed ages1/2/4/8/16/32/64/128/256, contrast, entropy, expected log age/log variance, short and long age mixtures. At early t, each age is truncated to t; no future or score-dependent reset.')]:
        entries.append({'experiment_id':cid,'kind':kind,'parent':['ResearchPool_BL80','S1 fixed Gaussian scale alternatives'],
          'scientific_question':question,'mechanism':'Fixed logsumexp evidence aggregation under a working H-estimated null; ranking features only.',
          'exact_feature_change':change,'implementation_constants':{'uniform_variance_prior':True,'uniform_sparse_age_prior':True,'maximum_feature_channel_absolute_z':12.},
          'expected_risk':['No real-DGP e-process guarantee','H estimation and dependence violate a simple null','Unreset evidence can accumulate null misspecification','Sparse recent windows are not themselves a calibrated anytime test']})
    register_family(family='E',module_name='src.next5.mixture',version='next5_fixed_mixture_v1',entries=entries,
        test_job='E000_FEATURE_CONTRACTS',test_xml='artifacts/next5/validation/E_FEATURE_CONTRACTS.xml',
        caller='scripts/register_next5_mixture.py',priority=55)


if __name__=='__main__':main()
