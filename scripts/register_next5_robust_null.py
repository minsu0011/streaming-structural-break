"""Freeze three null-nuisance questions without context-only target shortcuts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.registration import register_family


def main():
    entries=[]
    for cid,kind,question,change in [
      ('R1_H_ADAPTIVE_TAIL','adaptive_t','Can H-only tail-adaptive likelihoods improve scale ranking under different null tails?',
       '32 directional t likelihood features at windows8/32/128 and current, two channels. nu=clip(4+6/max(H standardized fourth/second-moment-squared minus3,1e-6),4,30). H degrees of freedom are nuisance parameters, never standalone features.'),
      ('R2_CONDITIONAL_SCALE_PIT','energy_pit','Does H-relative innovation-energy PIT surprise improve tail-robust scale evidence?',
       '16 features: lower and upper negative-log smoothed midrank PIT, centered/scaled by H surprise moments, current and EWMA8/32/128. Two channels; sorted H only, no online sorting.'),
      ('R3_QUANTILE_SELF_NORMALIZED','quantile_scale','Can robust H median/IQR energy normalization reduce null-scale misspecification?',
       '16 features: H q25/q50/q75 standardized innovations, log1p squared value relative to H median, lower/upper branches with H-only mean/std calibration; current and EWMA8/32/128, both channels.')]:
        entries.append({'experiment_id':cid,'kind':kind,'parent':['ResearchPool_BL80','H-only NEXT5 innovation channels'],
          'scientific_question':question,'mechanism':'Observation evidence calibrated by H-only nuisance parameters, with no raw H-context target shortcut.',
          'exact_feature_change':change,'implementation_constants':{'H_channel_cap':12.,'branch_scale_floor':.1,'quantile_scale_normalizer':1.349},
          'expected_risk':['Short H gives noisy nuisance estimates','Empirical PIT is not uniform under misspecification/dependence','Tail clipping can remove genuine changes','Synthetic robustness alone cannot satisfy the common real-data promotion gate']})
    register_family(family='R',module_name='src.next5.robust_null',version='next5_robust_null_v1',entries=entries,
        test_job='R000_FEATURE_CONTRACTS',test_xml='artifacts/next5/validation/R_FEATURE_CONTRACTS.xml',
        caller='scripts/register_next5_robust_null.py',priority=55)


if __name__=='__main__':main()
