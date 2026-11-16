"""Prospective small mechanism menu, retaining exact previous experiments."""
from pathlib import Path
import sys
import copy
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.io import register,registration_hash
from src.next.extensions import extension_hash
from src.next2.variance_bank import BankConfig,AGE_GRIDS,MEMORIES
from dataclasses import asdict
from src.utils.artifacts import sha256,write_json,utc_now


def main():
    out=ROOT/'configs/next2/VARIANCE_CANDIDATES_V1.json'
    if out.exists():
        raise RuntimeError('Prospective queue already frozen')
    lock=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text(encoding='utf-8'))
    parent=lock['references']['FROZEN_SCIENTIFIC_REFERENCE']['candidate']
    candidates=[]
    for eid,description,predicate in [
        ('D202_NO_EXPLICIT_GLR','Remove explicit GLR columns, retain GLR memory',lambda n:'_glr_' not in n),
        ('D205_NO_CUSUM_MEMORY','Remove positive and negative local CUSUM columns',lambda n:'_cusum_' not in n),
        ('D210_NO_FORECAST','Remove conditional variance forecast channel',lambda n:'raw_energy' in n)]:
        c=copy.deepcopy(parent)
        c.update(experiment_id=eid,parent_id=parent['experiment_id'],generation='NEXT2_V1',
            phase='DISTILLATION',hypothesis=description,change=description,counterexample_target='component attribution',
            keep_names=[n for n in parent['keep_names'] if predicate(n)])
        candidates.append(c)
    base=asdict(BankConfig())
    specs=[('E201_BANK_PARITY',{},'Exact raw/ARCH1 512-age bridge','adapter parity')]
    for energy,cap in [('winsor',3),('winsor',4),('winsor',6),('huber',4),('log1p',4),('abs',4),('clipped_abs',4),('robust_scale',4),('ecdf_tail',4),('log_ratio',4),('log_raw',4)]:
        specs.append((f'V2_{energy.upper()}_{cap}',{'energy':energy,'cap':cap},
            'Change only the raw-energy coordinate; H calibration and two-sided sequential bank are retained',
            'heavy tail / isolated outlier' if energy not in ('log_ratio','log_raw') else 'variance decrease / ARCH forecast error'))
    for ages in AGE_GRIDS:
        if ages!='all':
            specs.append(('A2_'+ages.upper(),{'ages':ages},'Retain a prospective age subgrid with width-normalized Bayes integration','long-age necessity'))
    for memory in MEMORIES:
        if memory!='original':
            specs.append(('M2_'+memory.upper(),{'memory':memory},'Replace only the two GLR memory coordinates with a bounded or forgetting alternative','long-null / late-break'))
    for norm in ('sqrt_age','log_age'):
        specs.append(('N2_'+norm.upper(),{'age_normalization':norm},'Normalize age-specific GLR/Bayes evidence before aggregation','long-null maximum growth'))
    specs.append(('D208_NO_ARCH',{'normalization':'none'},'Remove new-bank ARCH forecasting; preserve old ABCD feature base','ARCH necessity'))
    for eid,changes,hypothesis,target in specs:
        settings={**base,**changes}
        ext={'kind':'next2_variance','settings':settings}
        register(ext)
        candidates.append({'experiment_id':eid,'parent_id':parent['experiment_id'],'generation':'NEXT2_V1',
            'phase':'VARIANCE_MECHANISM','hypothesis':hypothesis,'change':changes or 'adapter only',
            'counterexample_target':target,'engine_config':parent['engine_config'],'groups':['NEXT2_VARIANCE'],
            'base':'arch8','base_family_selection':'ABCD','learner':'M4_lightgbm','weighting':'W3','params':{},
            'extension':ext,'extension_frozen_hash':extension_hash(ext),'next2_registration_sha256':registration_hash(),
            'parity_reference':parent['experiment_id'] if eid=='E201_BANK_PARITY' else None})
    write_json(out,{'version':'NEXT2_VARIANCE_V1','created_utc':utc_now(),
        'policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),'candidates':candidates,
        'bridge_policy':'E201 verifies full feature parity on all DEV against Y001; reuses Y001 fitted quality, never retrains the exact same design.'})
    print('NEXT2 VARIANCE QUEUE FROZEN',len(candidates),flush=True)


if __name__=='__main__':
    main()
