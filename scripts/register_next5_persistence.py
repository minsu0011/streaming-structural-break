"""Register the three predeclared persistent-versus-transient questions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.registration import register_family


def main():
    entries=[]
    for cid,kind,question,change in [
      ('P1_TWO_TIMESCALE_CONFIRMATION','two_timescale','Can agreement of fast and slow directional energy evidence reduce transient-shock mistakes?',
       '16 features over raw/conditional lower/upper H-relative log energies: variance-normalized EWMA8 and EWMA64, positive minimum confirmation, fast minus slow.'),
      ('P2_SOFT_PERSISTENCE_COUNT','soft_count','Does recent occupancy above an H-only evidence threshold add persistence information beyond amplitude?',
       '16 features: current directional evidence above its H-only q90, exact last-32 fraction, centered indicator EWMA8/64. Four raw/conditional lower/upper branches.'),
      ('P3_SHOCK_RECOVERY','shock_recovery','Can a rapidly decaying shock envelope distinguish sustained scale evidence from isolated shocks?',
       '16 features: log peak envelope decaying at half-life4, H-centered EWMA64 persistence, peak minus persistence, persistence/(1+peak). Fixed states initialize from H branch means.')]:
        entries.append({'experiment_id':cid,'kind':kind,'parent':['ResearchPool_BL80','H-only NEXT5 innovation channels','B2-inspired directional semitail evidence'],
          'scientific_question':question,'mechanism':'Causal distinction between directional evidence amplitude and persistence.',
          'exact_feature_change':change,'implementation_constants':{'branch_cap':12.,'H_branch_scale_floor':.1,'learned_reset':False},
          'expected_risk':['Repeated shocks can mimic persistence','H-only thresholds can be misspecified','Confirmation can delay genuine changes','Synthetic benefit must transfer to primary ResearchPool TS-AUC']})
    register_family(family='P',module_name='src.next5.persistence',version='next5_persistence_v1',entries=entries,
        test_job='P000_FEATURE_CONTRACTS',test_xml='artifacts/next5/validation/P_FEATURE_CONTRACTS.xml',
        caller='scripts/register_next5_persistence.py',priority=40)


if __name__=='__main__':main()
