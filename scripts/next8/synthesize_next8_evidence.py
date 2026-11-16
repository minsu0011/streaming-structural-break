"""Assemble separate A/B/C claims without changing any discovered model."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,digest,atomic,utc_now,activity,note,csv_write
from src.next8.evidence_synthesis import summarize_conditional_information,synthesize_science


def main():
    lock=read_json(BASE/'EVIDENCE_SYNTHESIS_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    directory=BASE/'evidence_synthesis';directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise RuntimeError('Existing synthesis retained; no silent claim revision')
    with activity('Separate A/B/C and conditional information','EVIDENCE_SYNTHESIS','B4_MOVEMENT_VS_DISCRIMINATION'):
        information=summarize_conditional_information();science=synthesize_science(information)
        atomic(directory/'CONDITIONAL_INFORMATION.json',{'timestamp':utc_now(),**information})
        csv_write('evidence_synthesis/CONDITIONAL_INFORMATION.csv',information['rows'])
        atomic(directory/'SCIENTIFIC_SYNTHESIS.json',{'timestamp':utc_now(),**science})
        csv_write('evidence_synthesis/FEATURE_CLASSIFICATION_NEXT8.csv',science['features'])
        atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'files_sha256':{n:digest(directory/n) for n in
            ('CONDITIONAL_INFORMATION.json','CONDITIONAL_INFORMATION.csv','SCIENTIFIC_SYNTHESIS.json','FEATURE_CLASSIFICATION_NEXT8.csv')},
            'counts':science['counts'],'scientific_status':science['scientific_status_subject_to_final_integrity_audit'],'final_audit_pending':True})
    note('Complete separate A/B/C synthesis','No certificate transferred across different direction heads; final audit pending','B4_MOVEMENT_VS_DISCRIMINATION')


if __name__=='__main__':main()
