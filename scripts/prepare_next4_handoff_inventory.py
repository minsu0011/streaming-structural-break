"""Review raw-free artifact membership before final report publication."""
import argparse
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,utc_now,progress
from src.next4.handoff_payload import collect


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+',args.tag):raise ValueError('Safe inventory tag required')
    paths,files,excluded,missing=collect(require_final=False)
    out=BASE/'handoff_inventory'/args.tag;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'RESULTS.json',{'status':'PAYLOAD_PREFLIGHT_NOT_DELIVERY','created_utc':utc_now(),
        'files':files,'excluded_reports':excluded,'missing_final_reports':missing,
        'files_count':len(files),'uncompressed_bytes':sum(r['bytes'] for r in files.values()),
        'raw_data_included':False,'real_point_OOF_included':False,'archive_created':False,
        'source_sha256':{name:digest(ROOT/name) for name in ['src/next4/handoff_payload.py','scripts/prepare_next4_handoff_inventory.py']}},immutable=True)
    progress('Raw-free handoff inventory prepared','Explicit sources, models and aggregate evidence reviewed; missing final reports remain marked pending',counts_as_research=False)
    print('NEXT4 PAYLOAD PREFLIGHT',len(files),'files;',sum(r['bytes'] for r in files.values()),'bytes;',len(missing),'final reports pending',flush=True)


if __name__=='__main__':main()
