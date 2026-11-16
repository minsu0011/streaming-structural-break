"""Capture the exact cold-test exception after its write-audit scope unwinds."""
from pathlib import Path
import importlib.util
import json
import sys
import traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
out=ROOT/'artifacts/next4/maintenance/COLD_API_FAILURE_001'
out.mkdir(parents=True,exist_ok=False)
path=ROOT/'scripts/verify_next4_cold_api.py'
spec=importlib.util.spec_from_file_location('original_cold_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
try:
    module.worker('B2',out/'B2_TRACE_RESULT.json')
except BaseException as exc:
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'REPRODUCED_EXCEPTION',
        'exception_type':type(exc).__name__,'exception_message':str(exc),'traceback':traceback.format_exc(),
        'original_script_unchanged':True,'new_fits':0,'repeated_original_job':False}
    (out/'EXCEPTION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
else:
    (out/'EXCEPTION.json').write_text(json.dumps({'status':'DID_NOT_REPRODUCE','created_utc':datetime.now(timezone.utc).isoformat(),'new_fits':0},indent=2)+'\n',encoding='utf-8')
    print('Cold B2 trace passed; preserve the original failure for investigation',flush=True)
