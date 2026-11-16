"""Bind complete fitted-fold public API verification before first execution."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.synthetic import load_reference,load_candidate
from src.next5.streaming import source_hash
from src.next5.scheduler import register_job


def main():
    state=read_json(BASE/'jobs/RT000_ADAPTER_CONTRACTS/STATE.json');assert state['status']=='COMPLETE' and state['exit_code']==0
    contract=BASE/'runtime_validation/CONTRACTS.xml';tree=ET.parse(contract);assert not tree.findall('.//failure') and not tree.findall('.//error')
    nominees=read_json(BASE/'stability/POLICY_LOCK.json')['diagnostic_nominees'];candidates={}
    for cid in nominees:
        descriptors={}
        for fold in range(5):
            _,references=load_reference('full',fold);_,registration,proof=load_candidate(cid,'full',fold)
            path=BASE/'runtime_validation/descriptors'/cid/('fold_'+str(fold)+'.json')
            atomic_json(path,{'created_utc':utc_now(),'kind':'NEXT5_READONLY_FITTED_RESEARCH_COMPOSITE','candidate':cid,'scope':'full','fold':fold,
                'candidate_proof':proof,'reference_proof':references,'combination':registration['combination'],'streaming_source_sha256':source_hash(),
                'promotion':False,'public_submission':False},immutable=True)
            descriptors[str(fold)]=path.relative_to(ROOT).as_posix()
        candidates[cid]={'family':nominees[cid]['family'],'descriptors':descriptors}
    sources=('src/next5/runtime.py','src/next5/streaming.py','src/next5/stream_contracts.py','scripts/verify_next5_streaming.py',
        'src/next5/synthetic.py','src/next5/model.py','src/next5/data.py','src/next5/guard.py','src/next/readonly_jit.py')
    atomic_json(BASE/'runtime_validation/API_POLICY_LOCK.json',{'created_utc':utc_now(),'candidates':candidates,'prefix_cases':300,'prefix_seed':20260908,
        'real_scope':'All 6400 ResearchPool series under their actual PRIMARY held-out fitted artifact; no Shadow or old full-DEV inference.',
        'scalar_offline_probability_absolute_tolerance':1e-15,'official_score_parity_absolute_tolerance':1e-14,
        'one_shot_current_point_traps':True,'future_suffix_and_length_traps':True,'filesystem_write_trap':True,'reverse_order_per_fold_series':20,
        'contract_proof_sha256':digest(contract),'sources_sha256':{name:digest(ROOT/name) for name in sources}},immutable=True)
    for cid in candidates:
        register_job({'id':'API_'+cid,'kind':'validation','priority':86,'depends_on_success':['RT000_ADAPTER_CONTRACTS',cid+'_full'],
            'argv':['scripts/verify_next5_streaming.py','--candidate',cid],
            'description':'All ResearchPool held-out outputs through read-only scalar API, 300 future-prefix tests and reverse resets: '+cid,'retry_safe':False})
    progress('Full ResearchPool streaming audit preregistered','Seven scientific representatives, all five fitted folds, 300 prefixes each and no persistent writes or future reads')


if __name__=='__main__':main()
