"""Record formula/source review supporting the nine Korean feature cards."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,digest,activity,utc_now

with activity('Cross-check nine Korean feature definitions against frozen native names and source formulas',
              'SCIENTIFIC_REPORTING','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
    from src.next8.features import NAMES,NORMALIZATION,mechanical_group
    from src.next9.feature_meanings import MEANINGS
    expected={1:'arch8__current_abs_z',51:'serial_variance__raw_energy_current',
        59:'serial_variance__raw_energy_glr_logage',64:'serial_variance__log_variance_forecast_current',
        72:'serial_variance__log_variance_forecast_glr_logage',77:'next4_log_semitail_raw_energy_lower_current',
        81:'next4_log_semitail_raw_energy_upper_current',85:'next4_log_semitail_conditional_energy_lower_current',
        89:'next4_log_semitail_conditional_energy_upper_current'}
    records=[]
    for j,name in expected.items():
        assert NAMES[j]==name and mechanical_group(j)==MEANINGS[j][2]
        records.append(dict(feature=j,native_name=name,Korean_name=MEANINGS[j][0],meaning=MEANINGS[j][1],
            mechanical_group=mechanical_group(j),all_followed_by_frozen_NEXT8_H40_normalization=True))
    files=['src/next8/features.py','src/next5/reference_features.py','src/next4/scale_asymmetry.py',
           'src/next/variance_evidence.py','src/next/serial_variance_evidence.py','src/next/channel_evidence.py',
           'src/next9/feature_meanings.py']
    atomic(BASE/'FEATURE_FORMULA_REVIEW_NEXT9.json',dict(status='PASS',timestamp=utc_now(),records=records,
        source_sha256={p:digest(ROOT/p) for p in files},normalization_contract=NORMALIZATION,
        reviewed_formula_details=[
            'Raw residual energy primitive is centered residual squared / H variance minus1; serial H calibration and final NEXT8 H normalization follow.',
            'Log variance forecast uses conditional variance computed from the previous residual state before inserting the current squared residual.',
            'F59/F72 are log2 of the fixed-grid GLR maximizing window; neither is a true structural-break age.',
            'Semitails use log(energy+1e-4), positive lower/upper deviation from H median, separate H scales and clipping, then NEXT8 normalization.',
            'UP/DOWN in the cards describes pre-current movement of each normalized feature, not positive/negative return or a certified target relationship.'],
        source_formula_review_not_empirical_causal_claim=True,cards_sha256=digest(BASE/'FEATURE_CARDS_NEXT9_KO.md')))
    print(dict(status='PASS',features=len(records)))
