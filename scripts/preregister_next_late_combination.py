"""Freeze eligible fixed blends now; prohibit score evaluation before hour 8."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timedelta
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.engine import EngineConfig
from src.next.fitting import fitting_hash
from src.next.predictor import predictor_hash
from src.next.late_blend import source_hash
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    path = ROOT/'configs/next_late_combination_candidates.json'
    if path.exists():
        print('Late-combination queue already frozen')
        return
    partner = 'K008_RANK_SCORE_HIST128'
    forensic_path = ROOT/'artifacts/next/forensic'/partner/'RESULTS.json'
    forensic = json.loads(forensic_path.read_text(encoding='utf-8'))
    if not forensic['complementarity']['admitted_for_predeclared_fixed_late_combination']:
        raise RuntimeError('No predeclared complementary-partner evidence')
    session = json.loads((ROOT/'artifacts/next/SESSION_NEXT.json').read_text(encoding='utf-8'))
    not_before = datetime.fromisoformat(session['research_start_utc'])+timedelta(hours=8)
    candidates = []
    for j,weight in enumerate((.5,.67,.33),1):
        candidates.append({'experiment_id':f'L00{j}_PRIMARY_K008_{round(weight*100):02d}',
            'parent_id':partner,'generation':7,'phase':'LATE_COMBINATION',
            'hypothesis':'The frozen primary offsets the ranking partner mixed-change/early-time weakness, while the partner adds scale/dependence and late-time strength.',
            'change':f'Fixed {round(weight*100)}/{round((1-weight)*100)} primary/partner float32 score average; no fitted blend weight.',
            'expected_effect':'Combine documented complementary rankings at bounded additional latency.',
            'risks':'Adds model and feature work; score scales differ; complementarity was identified adaptively in DEV and is not an independent discovery.',
            'engine_config':asdict(EngineConfig()),'groups':[],'base':'fixed_components',
            'learner':'fixed_late_blend','weighting':'inherited fixed component recipes','params':{},
            'late_blend':{'primary':'PRIMARY','partner':partner,'primary_weight':weight,'method':'score_average'},
            'late_blend_source_sha256':source_hash(), 'forensic_evidence_sha256':sha256(forensic_path)})
    write_json(path,{'status':'LOCKED','created_utc':utc_now(),'generation':7,
        'fitting_code_sha256':fitting_hash(),'predictor_code_sha256':predictor_hash(),
        'robustness_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'score_evaluation_not_before_utc':not_before.isoformat(),
        'late_stage_rule':'The user requested ensemble work in the latter session stage. Freeze the eligible finite grid after forensic evidence, but do not inspect any combination scores before elapsed research hour 8.',
        'thresholds_splits_seeds_unchanged':True,'complementarity':forensic['complementarity'],'candidates':candidates})
    print('FROZEN LATE COMBINATION',len(candidates),'NOT BEFORE',not_before.isoformat())


if __name__=='__main__':
    main()
