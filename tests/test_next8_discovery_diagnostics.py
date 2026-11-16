"""Wide uncertainty is not positive evidence of a reversed direction."""
import src.next8
from src.next8.discovery_diagnostics import atlas_gate_reasons


def test_uncertain_and_supported_reversal_are_distinct():
    base={'support_status':'SUPPORTED','positive_series':40,'negative_series':100,'series':120}
    rows=[{**base,'domain_type':'all','domain':'all','TS_AUC':.55,'ci_low':.52,'ci_high':.58}]
    for family in ('time','H_length','own_pre_state'):
        for k in range(2):rows.append({**base,'domain_type':family,'domain':str(k),'TS_AUC':.53,'ci_low':.49,'ci_high':.57})
    assert atlas_gate_reasons(rows)['passed']
    rows[1].update(TS_AUC=.50,ci_low=.38,ci_high=.62)
    result=atlas_gate_reasons(rows)
    assert not result['passed'] and result['failed_domains'][0]['failure_type']=='UNCERTAINTY_DOES_NOT_EXCLUDE_MATERIAL_REVERSAL'
    rows[1].update(TS_AUC=.40,ci_low=.35,ci_high=.45)
    assert atlas_gate_reasons(rows)['failed_domains'][0]['failure_type']=='SUPPORTED_MATERIAL_REVERSAL'
