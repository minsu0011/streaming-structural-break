"""Exactly the preregistered whole-series paired draws for frozen G1 finalists."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next7/numba_cache')
import numpy as np
from numba import set_num_threads
from src.next7.common import BASE,save,read_json,digest,csv_write,utc_now,activity,require_guard
from src.next6.data import source
from src.next5.bootstrap import PreparedAUC,paired_summary


def prediction(method,fold,metadata,guard):
    directory=BASE/f'g1/fold_{fold}'
    path=directory/method/'predictions.npz' if method in ('I3_CURRENT_EVENT','M_FULL','I4_EVENT_SCALE_TREE') else directory/'fixed_predictions.npz'
    with np.load(path,allow_pickle=False) as values:
        rows=values['global_row'].copy()
        guard.admit(np.unique(values['dataset_id']),purpose='G1 paired-bootstrap RP admission')
        np.testing.assert_array_equal(metadata['dataset_id'][rows],values['dataset_id'])
        scores=values['prediction' if method in ('I3_CURRENT_EVENT','M_FULL','I4_EVENT_SCALE_TREE') else method].copy()
    return rows,scores


def main():
    guard=require_guard()
    finalist=read_json(BASE/'G1_FINALISTS_FROZEN.json')
    assert finalist['status']=='FROZEN'
    n=finalist['draws']
    assert n in (2000,5000)
    methods=['M0',*finalist['selected']]
    source_sha={name:digest(ROOT/name) for name in ('scripts/next7/bootstrap_g1.py','src/next5/bootstrap.py','src/next/bootstrap.py')}
    lockfile=BASE/'G1_BOOTSTRAP_COUNTS_LOCK.json'
    data=source()
    metadata=data.rows()
    if not lockfile.exists():
        with activity('freeze G1 whole-series bootstrap draws','UNCERTAINTY_SETUP','F5_CAUSAL_NESTED_INTEGRATION'):
            rng=np.random.default_rng(read_json(BASE/'NEXT7_RANDOMNESS_LOCK.json')['bootstrap_seed'])
            records={}
            for fold in range(5):
                ids=np.array(sorted(int(s) for s in data.ids if guard.split.assignment(int(s))[1]==fold),dtype=np.int64)
                assert len(set(guard.split.groups[int(i)] for i in ids))==len(ids)
                counts=np.vstack([np.ones((1,len(ids)),dtype=np.int32),
                    rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=n)]).astype(np.int32)
                path=BASE/f'bootstrap/g1/counts/fold_{fold}.npz'
                path.parent.mkdir(parents=True,exist_ok=True)
                assert not path.exists()
                np.savez(path,dataset_id=ids,multiplicities=counts)
                records[str(fold)]={'path':path.relative_to(BASE).as_posix(),'sha256':digest(path),
                    'groups':len(ids),'identity_row_plus_draws':n+1}
            save('G1_BOOTSTRAP_COUNTS_LOCK.json',{'status':'FROZEN','created_utc':utc_now(),'draws':n,
                'methods':methods,'counts':records,'source_sha256':source_sha,
                'finalist_sha256':digest(BASE/'G1_FINALISTS_FROZEN.json'),
                'all_methods_share_exact_series_multiplicities':True,'resampling':'whole series within original outer fold',
                'no_post_interval_draw_extension':True})
    lock=read_json(lockfile)
    assert lock['source_sha256']==source_sha and lock['draws']==n and lock['methods']==methods
    set_num_threads(4)
    point=read_json(BASE/'G1_OUTER_SCORES.json')['scores']
    boot={}
    for method in methods:
        matrix=np.empty((n+1,5))
        for fold in range(5):
            directory=BASE/f'bootstrap/g1/{method}'
            outfile=directory/f'fold_{fold}.npy'
            done=directory/f'fold_{fold}.json'
            if done.exists():
                assert digest(outfile)==read_json(done)['result_sha256']
                matrix[:,fold]=np.load(outfile,allow_pickle=False)
                continue
            with activity(f'paired whole-series G1 {method} fold{fold} draws{n}','UNCERTAINTY','F5_CAUSAL_NESTED_INTEGRATION'):
                count_path=BASE/lock['counts'][str(fold)]['path']
                assert digest(count_path)==lock['counts'][str(fold)]['sha256']
                with np.load(count_path,allow_pickle=False) as values:
                    guard.admit(values['dataset_id'],purpose='G1 bootstrap count admission')
                    ids=values['dataset_id'].copy()
                    counts=values['multiplicities'].copy()
                rows,p=prediction(method,fold,metadata,guard)
                actual,codes=np.unique(metadata['dataset_id'][rows],return_inverse=True)
                np.testing.assert_array_equal(actual,ids)
                prepared=PreparedAUC.create(metadata['target'][rows],p,metadata['time_online'][rows],codes,len(ids))
                values=prepared.evaluate(counts)
                assert abs(values[0]-point[method][str(fold)])<1e-12
                directory.mkdir(parents=True,exist_ok=True)
                assert not outfile.exists()
                np.save(outfile,values,allow_pickle=False)
                save(str(done.relative_to(BASE)),{'status':'COMPLETE','created_utc':utc_now(),
                    'method':method,'fold':fold,'draws':n,'result_sha256':digest(outfile),
                    'counts_sha256':digest(count_path),'unweighted_scorer_parity':True})
                matrix[:,fold]=values
        boot[method]=matrix
    rows=[]
    for method in finalist['selected']:
        for reference in ('M0','M_CURRENT','M_EVENT'):
            if method==reference or reference not in boot:
                continue
            rows.append({'candidate':method,'reference':reference,
                'fixed_fitted_OOF_conditional':True,'adaptive_selection_not_corrected':True,
                **paired_summary(boot[reference],boot[method])})
    if not (BASE/'PAIRED_BOOTSTRAP_NEXT7_G1.csv').exists():
        csv_write('PAIRED_BOOTSTRAP_NEXT7_G1.csv',rows)
        save('G1_BOOTSTRAP_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'draws':n,
            'finalists':finalist['selected'],'counts_lock_sha256':digest(lockfile),
            'all_unweighted_fold_scores_match':True,'source_sha256':source_sha})
    guard.check_state()


if __name__=='__main__':
    main()
