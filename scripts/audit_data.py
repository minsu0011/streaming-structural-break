import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.audit import raw_manifest,audit_training
from src.utils.artifacts import write_json

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',default='data/raw'); args=parser.parse_args()
    raw=raw_manifest(args.data,ROOT/'artifacts/audits/RAW_DATA_MANIFEST.json')
    if not raw['files']:
        report={'status':'BLOCKED_AUTH','series_count':None,'break_fraction':None,'seal_evaluated':False,'reason':'No raw competition parquet acquired; no invented statistics'}
        write_json(ROOT/'artifacts/audits/DATA_AUDIT.json',report)
        (ROOT/'artifacts/audits/DATA_AUDIT.md').write_text('# Data audit\n\nNOT RUN: official data unavailable (authentication). Counts, distributions, duplicates, stationarity and break ratios are unknown. Synthetic engineering fixtures are not competition data.\n',encoding='utf-8')
        print(report)
        return
    print(audit_training(args.data,ROOT/'artifacts/audits',ROOT/'artifacts/validation'))

if __name__=='__main__': main()
