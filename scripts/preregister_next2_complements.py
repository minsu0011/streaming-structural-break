"""Freeze nine bounded standalone complement hypotheses, then fixed blend grid."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.io import register,registration_hash
from src.next.extensions import extension_hash,names_and_groups
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    out=ROOT/'configs/next2/COMPLEMENT_CANDIDATES_V1.json'
    if out.exists():
        raise RuntimeError('Complement queue already frozen')
    recipes=[('C201_MEAN_LOCAL','mean','local',8),('C202_MEAN_BAYES','mean','bayes',8),
        ('C203_RANK_SIGN','rank_sign','local',8),('C204_PAGE_HINKLEY','page_hinkley','all',8),
        ('C205_DEPENDENCE','dependence','compact',8),('C206_SQUARED_DEPENDENCE','squared_dependence','compact',8),
        ('C207_ABSOLUTE_DEPENDENCE','absolute_dependence','compact',8),
        ('C208_CDF8','cdf','all',8),('C209_CDF16','cdf','all',16)]
    candidates=[]
    for eid,family,subset,bins in recipes:
        ext={'kind':'next2_complement','settings':{'family':family,'order':8,'cdf_bins':bins,'cap':4}}
        register(ext)
        names,_=names_and_groups(ext)
        if subset=='local':
            names=[n for n in names if any(n.endswith(s) for s in ('current','ewma8','ewma32','ewma128','cusum_positive','cusum_negative'))]
        elif subset=='bayes':
            names=[n for n in names if any(n.endswith(s) for s in ('bayes025','bayes1','cusum_positive','cusum_negative'))]
        elif subset=='compact':
            names=[n for n in names if any(n.endswith(s) for s in ('ewma8','ewma32','cusum_positive','cusum_negative','bayes025','bayes1'))]
        candidates.append({'experiment_id':eid,'parent_id':'J102_SERIAL_VARIANCE_512','generation':'NEXT2_COMPLEMENT_V1',
            'phase':'STANDALONE_COMPLEMENT','hypothesis':'Conditional-variance-normalized '+family+' evidence may carry information independent of the variance primary.',
            'change':'Standalone small '+family+' bank, feature subset '+subset,'counterexample_target':family,
            'engine_config':{'order':8,'normalization':'sd','cdf_bins':16},'groups':['NEXT2_COMPLEMENT'],
            'base':'none','learner':'M4_lightgbm','weighting':'W3','params':{},'keep_names':names,
            'extension':ext,'extension_frozen_hash':extension_hash(ext),'next2_registration_sha256':registration_hash()})
    write_json(out,{'created_utc':utc_now(),'version':'NEXT2_COMPLEMENT_V1',
        'policy_sha256':sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),'candidates':candidates,
        'promotion':'Standalone screen > .54 and mean matched within-time rank correlation < .95; full standalone > .54 and correlation < .95 before blend evaluation',
        'blend_grid':[.9,.8,.67,.5],'meta_training':False})
    print('NEXT2 COMPLEMENTS FROZEN',len(candidates),flush=True)


if __name__=='__main__':
    main()
