"""Complete DEV equivalence of frozen binaries under new runtime pruning."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next.candidate_io import model_class,validate_sources
from src.next.serial_variance_runtime import PreparedSerialVariancePredictor
from src.next.fast_variance_initialization import PreparedFastVariancePredictor
from src.next2.pruned_runtime import PreparedCompactPredictor,source_hash
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    data=prepare_dev(ROOT)
    lock=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text(encoding='utf-8'))
    for role,ref in lock['references'].items():
        eid=ref['experiment_id']
        out=ROOT/'artifacts/next2/runtime_pruning'/eid
        out.mkdir(parents=True,exist_ok=True)
        identity={'model_sha256':ref['model_sha256'],'runtime_sha256':source_hash(),
            'script_sha256':sha256(Path(__file__)),'dev_manifest_sha256':sha256(data.directory/'MANIFEST.json')}
        if (out/'RESULTS.json').exists():
            r=json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
            assert r['identity']==identity and r['status']=='PASS'
            for name,digest in r['files_sha256'].items():
                assert sha256(out/name)==digest
            print('PRUNING REUSED',eid,flush=True)
            continue
        validate_sources(ref['candidate'])
        assert sha256(ROOT/ref['model_path'])==ref['model_sha256']
        model=model_class(ref['candidate']).load(ROOT/ref['model_path'])
        oldp=(PreparedSerialVariancePredictor if 'SERIAL' in eid else PreparedFastVariancePredictor)(model)
        newp=PreparedCompactPredictor(model)
        prediction=np.lib.format.open_memmap(out/'all_dev_predictions.npy',mode='w+',dtype=np.float32,shape=(len(data.online),))
        ho,oo=data.series['historical_offsets'],data.series['online_offsets']
        started=time.perf_counter()
        for j,sid in enumerate(data.ids):
            historical=data.historical[ho[j]:ho[j+1]]
            online=data.online[oo[j]:oo[j+1]]
            left,right=oldp.new_state(historical),newp.new_state(historical)
            expected=np.empty(len(online),dtype=np.float32)
            actual=np.empty(len(online),dtype=np.float32)
            for k,point in enumerate(online):
                expected[k]=left.predict_one(point)
                actual[k]=right.predict_one(point)
                # Selected feature vectors, not only model outputs, must agree.
                np.testing.assert_array_equal(left.output,right.output)
            np.testing.assert_array_equal(expected,actual)
            prediction[oo[j]:oo[j+1]]=actual
            if (j+1)%500==0:
                print('PRUNING EXACT',eid,j+1,round(time.perf_counter()-started,1),'seconds',flush=True)
                checkpoint('Exact runtime DEV equivalence '+eid+' '+str(j+1),
                    ['Finish all three frozen models','Isolated component latency benchmark'],
                    artifacts=['configs/next2/RESEARCH_POLICY_LOCK.json'])
        prediction.flush()
        del prediction
        report={'status':'PASS','created_utc':utc_now(),'identity':identity,'experiment_id':eid,
            'variant': 'B104_RUNTIME_PRUNED' if newp.mode=='local_bayes' else eid+'_RUNTIME_COMPACT',
            'series':len(data.ids),'points':len(data.online),'prediction_max_abs_diff':0.,
            'selected_feature_max_abs_diff':0.,'old_state_array_bytes':left.state_array_bytes,
            'new_state_array_bytes':right.state_array_bytes,
            'state_reduction_fraction':1-right.state_array_bytes/left.state_array_bytes,
            'mode':newp.mode,'unused_channel_state_removed':True,
            'explicit_glr_computation_removed':newp.mode=='local_bayes',
            'glr_memory_removed':newp.mode=='local_bayes','model_binary_unchanged':True,
            'elapsed_seconds':time.perf_counter()-started,'seal_rows':0,
            'quality_authority':'Exact runtime equivalence only; these all-DEV predictions are in-sample, not quality estimates.',
            'files_sha256':{'all_dev_predictions.npy':sha256(out/'all_dev_predictions.npy')}}
        write_json(out/'RESULTS.json',report)
        print('PRUNING PASS',eid,report['old_state_array_bytes'],report['new_state_array_bytes'],flush=True)


if __name__=='__main__':
    main()
