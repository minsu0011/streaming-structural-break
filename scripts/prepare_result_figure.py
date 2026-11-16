"""Prepare fixed DEV figure inputs from frozen reports, without any model run."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_handoff_real import ablation_panel
from src.utils.artifacts import sha256,write_json


def main():
    state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text())
    frozen=json.loads((ROOT/'DEV_CHAMPION_CANDIDATES.json').read_text())
    if frozen['status']!='FROZEN_FOR_FINAL_DIAGNOSTICS':raise RuntimeError('Frozen selection required')
    lookup={row['experiment_id']:row for row in state['results']};primary=lookup[frozen['PRIMARY_DEV_CANDIDATE']]
    spec,panel,_=ablation_panel(state['results'],primary)
    selected={role:{key:lookup[item['experiment_id']][key] for key in ('experiment_id','mean','median','worst_fold','fold_scores','time_profile','age_profile')} for role,item in frozen['candidates'].items()}
    timings={}
    for role,item in frozen['candidates'].items():
        path=ROOT/'artifacts/real/latency/final_deployment'/f"{item['experiment_id']}_fused_fast_init.json"
        row=json.loads(path.read_text());descriptor=json.loads((ROOT/item['model_directory']/'model.json').read_text())
        if row['model_sha256']!=descriptor['model_sha256']:raise RuntimeError('Figure latency artifact mismatch')
        timings[role]={'point_us':row['long_stream']['measured_us_per_point'],
            'including_initialization_estimate_us':row['real_history_initialization']['ten_million_amortized_seconds_at_observed_mean_length']/10,
            'continuous_points_measured':row['long_stream']['points'],'initialization_histories':row['real_history_initialization']['series'],
            'source_path':str(path.relative_to(ROOT)),'source_sha256':sha256(path)}
    baselines=[lookup[f'B{k}_'+next(row['experiment_id'].split('_',1)[1] for row in state['results'] if row['experiment_id'].startswith(f'B{k}_'))] for k in range(8)]
    best_stat=max((row for row in state['results'] if row['phase'] in ('statistical_ladder','complementary_statistical')),key=lambda row:row['mean'])
    baselines.append(best_stat)
    data={'scope':'Frozen DEV-only descriptive results. Adaptive selection among 249 candidates; no independent seal estimate and no reduced result.',
        'selection_sha256':sha256(ROOT/'DEV_CHAMPION_CANDIDATES.json'),'research_state_sha256':sha256(ROOT/'artifacts/real/RESEARCH_STATE.json'),
        'freeze_timestamp':frozen['freeze_timestamp'],'selected':selected,'baseline':lookup['B1_official_ewma'],
        'statistical_ladder':[{'name':row['model'],'mean':row['mean']} for row in baselines],
        'matched_ar4_panel':{'configuration':spec,'families':{name:panel[name]['mean'] for name in 'ABCD'}},'latency':timings,
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    write_json(ROOT/'artifacts/real/figures/FIGURE_INPUTS.json',data)
    print('FROZEN DEV FIGURE INPUTS READY')


if __name__=='__main__':main()
