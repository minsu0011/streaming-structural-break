"""Freeze descriptive scope definitions once; register completed-full diagnostics."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next4.diagnostics import source_hash
from src.next4.scheduler import register_job


def main():
    lock=BASE/'diagnostic_policy/FORENSIC_POLICY.json'
    sources={**source_hash(),'scripts/forensic_next4_candidate.py':digest(ROOT/'scripts/forensic_next4_candidate.py')}
    if not lock.exists():
        xml=BASE/'maintenance/FORENSIC_UNIT_TESTS.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
        assert sum(int(s.get('tests',0)) for s in suites)==4 and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
        atomic_json(lock,{'status':'FROZEN_BEFORE_NEXT4_SUBGROUP_SCORES','created_utc':utc_now(),
            'source_sha256':sources,'unit_test_xml_sha256':digest(xml),
            'candidate_rule':'Every complete full-CV candidate with overall delta >= -0.0005, plus every complete Family B or D candidate. No new fitted candidates created by this rule.',
            'scope_count':26,'base_scopes':'Unchanged 24 all-time/time/age/H-regime/mechanism definitions from prior forensic implementation',
            'additional_scopes':'Observed raw-window SD decrease/increase inside existing SCALE_DOMINANT signature, plus all no-break controls; original signature policy windows',
            'bootstrap':'Exactly NEXT4 preregistered 2000 whole-series draws; cache BL80 reference once per fold/scope; reuse prior NEXT4 all-time candidate samples exactly',
            'support':'At least 100 break series; H-regime also requires 100 no-break controls. Directional subsets below 100 are diagnostic only.',
            'diversity_definitions':'Standalone new expert against J102, C205, BL80 in each full held-out fold. Ordinary Spearman ranks pooled within that fold; within-time Spearman correlates per-time percentile ranks. Top 1%/5% tails use those ranks; ties may yield empty tails. Negative-point disagreement includes pre-change points. No fit and no promotion authority from correlation alone.',
            'statistical_scope':'Conditional fixed OOF only. No model-fitting, adaptive-selection or subgroup multiple-comparison correction. H-regime slices are not matched held-out-H-regime model tests.',
            'model_fit_changes':False,'new_fits':0,'seal_rows':0},immutable=True)
    else:assert read_json(lock)['source_sha256']==sources
    for result in sorted((BASE/'experiments').glob('*/full/RESULTS.json')):
        full=read_json(result)
        if full['status']!='COMPLETE' or (full['mean_delta']<-.0005 and full['family'] not in ('B','D')):continue
        candidate=full['candidate'];menu=read_json(BASE/'families'/(full['family']+'_MENU_LOCK.json'))
        index=menu['candidate_ids'].index(candidate);job_id=f'{full["family"]}{index+301:03d}_FORENSIC'
        if (BASE/'queue'/(job_id+'.json')).exists():continue
        bootstrap=f'{full["family"]}{index+201:03d}_BOOTSTRAP'
        if not (BASE/'queue'/(bootstrap+'.json')).exists():continue
        register_job({'id':job_id,'kind':'validation','family':full['family'],'candidate_id':candidate,
            'depends_on_success':[bootstrap],'argv':['scripts/forensic_next4_candidate.py','--candidate',candidate],
            'description':'Frozen 26 diagnostic scopes and standalone diversity against all three references','retry_safe':False,
            'diagnostic_policy_sha256':digest(lock)})
        print('QUEUED',job_id,flush=True)
    progress('Registered fixed forensics','Group support and uncertainty for existing predictions; frozen descriptive policy, no new candidate generation')


if __name__=='__main__':main()
