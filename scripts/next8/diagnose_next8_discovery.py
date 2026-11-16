"""A full fixed-gate failure decomposition for the Korean handoff."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity,note,csv_write
from src.next8.discovery_diagnostics import diagnose_discovery


def main():
    lock=read_json(BASE/'DISCOVERY_DIAGNOSTICS_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    directory=BASE/'discovery_diagnostics';directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise RuntimeError('Discovery diagnostics already exist')
    with activity('Explain every fixed TRAIN discovery gate','DISCOVERY_DIAGNOSTICS','B1_CONDITIONAL_DIRECTION'):
        rows,domains,summary=diagnose_discovery()
        csv_write('discovery_diagnostics/GATES.csv',rows);csv_write('discovery_diagnostics/DOMAIN_FAILURES.csv',domains)
        atomic(directory/'SUMMARY.json',{'timestamp':utc_now(),**summary})
        atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'rows':len(rows),'domain_failures':len(domains),
            'files_sha256':{n:digest(directory/n) for n in ('GATES.csv','DOMAIN_FAILURES.csv','SUMMARY.json')},
            'failure_decomposition_does_not_relax_selection':True})
    note('Complete discovery failure decomposition','Wide uncertainty separated from demonstrated direction reversal','B1_CONDITIONAL_DIRECTION')


if __name__=='__main__':main()
