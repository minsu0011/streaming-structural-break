"""Cross-fold conditional/stable registries before adding separate A/C roles."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,digest,atomic,utc_now,activity,note,csv_write
from src.next8.feature_certification import certify_features


def main():
    lock=read_json(BASE/'FEATURE_CERTIFICATION_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    directory=BASE/'feature_certification';directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise RuntimeError('Feature certificates already exist; no outcome-driven reselection')
    with activity('Cross-fold stable and conditional direction certification','FEATURE_CERTIFICATION','B1_CONDITIONAL_DIRECTION'):
        normal=certify_features(False);early=certify_features(True)
        atomic(directory/'NORMAL.json',{'timestamp':utc_now(),**normal});atomic(directory/'EARLY.json',{'timestamp':utc_now(),**early})
        csv_write('feature_certification/CLASSIFICATION.csv',normal['features']+early['features'])
        csv_write('feature_certification/CONDITIONAL_UNITS.csv',normal['conditional_units']+early['conditional_units'])
        atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'files_sha256':{
            name:digest(directory/name) for name in ('NORMAL.json','EARLY.json','CLASSIFICATION.csv','CONDITIONAL_UNITS.csv')},
            'normal_counts':normal['counts'],'early_counts':early['counts'],'does_not_claim_A_or_C':True})
    note('Complete cross-fold direction certification','Zero discoveries allowed; movement and incremental score remain separate','B1_CONDITIONAL_DIRECTION')


if __name__=='__main__':main()
