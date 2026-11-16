"""Record an explicit publication inventory before the final audit is admitted."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,save,digest,utc_now,note
from src.next7.handoff_archive import checked_name,FORBIDDEN_FILES

FINAL_FILES='''00_READ_FIRST_NEXT7_KO.md 01_GPT_HANDOFF_NEXT7.md BOTTLENECK_BOARD_FINAL_NEXT7.json
LATENCY_NEXT7.csv COLD_START_NEXT7.csv H_INITIALIZATION_LATENCY_NEXT7.csv DIRECT_10M_THROUGHPUT_NEXT7.csv
STATE_MEMORY_NEXT7.json LEAKAGE_AUDIT_NEXT7.json PREFIX_INVARIANCE_NEXT7.json DETERMINISM_NEXT7.json
REPRODUCTION_NEXT7.json FINAL_AUDIT_NEXT7.json FINAL_CANDIDATE_FREEZE_NEXT7.json RESEARCH_COMPLETION_READY.json
SEAL_OPEN_RECOMMENDATION_NEXT7.md SHADOW_OPEN_RECOMMENDATION_NEXT7.md LIMITATIONS_NEXT7.md
NEXT_SESSION_PLAN_NEXT7.md SOURCE_MANIFEST_NEXT7.json ENVIRONMENT_NEXT7.json FINAL_CANDIDATE_DECISIONS_NEXT7.csv
SCALE_EXPANSION_CONTRACTION_RESULTS.csv SHOCK_RESULTS_NEXT7.csv FALSE_LATCH_RESULTS_NEXT7.csv
G3_ALL_PAIRS_OBJECTIVE_DIAGNOSTICS_NEXT7.csv G3_ALL_PAIRS_GRADIENT_DIAGNOSTICS_NEXT7.csv
CONDITIONAL_NULL_RESULTS_NEXT7.csv PER_H_CONDITIONAL_NULL_RISK_NEXT7.csv EMPIRICAL_SUPPORT_PARTITION_NEXT7.csv
CROSSED_H_LENGTH_CONTRASTS_NEXT7.csv CROSSED_H_LENGTH_DISCORDANCE_NEXT7.csv H_NORMALIZER_RISK_ASSOCIATIONS_NEXT7.csv
CLASSIFIER_STRESS_RESULTS_NEXT7.csv CLASSIFIER_STRESS_PAIRED_NEXT7.csv CLASSIFIER_AMPLITUDE_CONTROL_NEXT7.csv
CLASSIFIER_LONG_NULL_RESULTS_NEXT7.csv COMPLETE_DIAGNOSTICS_GRADIENTS_NEXT7.json
COMPLETE_DIAGNOSTICS_CLASSIFIERS_NEXT7.json COMPLETE_DIAGNOSTICS_CONDITIONAL_NEXT7.json LATENCY_NEXT7_DONE.json
HANDOFF_CONTENT_PLAN_NEXT7.json'''.split()


def main():
    existing=[p.name for p in BASE.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md','.xml')
              and p.name.lower() not in FORBIDDEN_FILES]
    names=sorted(set(existing+FINAL_FILES))
    for name in names:checked_name('artifacts/next7/'+name)
    save('HANDOFF_CONTENT_PLAN_NEXT7.json',{'status':'PLANNED_NOT_FINAL_ARCHIVE','created_utc':utc_now(),
        'source_sha256':digest(__file__),'explicit_root_artifact_names':names,
        'required_before_export':FINAL_FILES,
        'source_groups':['src/**/*.py','scripts/next7/*.py'],
        'root_source_files':['COUNTEREXAMPLE_SUITE_LOCK.json','requirements.txt','pyproject.toml'],
        'required_config_files':['features.json','arch_input.json','ar_residual_input.json','historical_context.json',
            'historical_quantiles.json','magnitude_dependence.json','rank_gaussian.json'],
        'research_bundle_group':'research_bundles/fold_{0..4}: exactly five original models, scale calibration and manifest per fold',
        'fixed_evidence_groups':[
            'bottlenecks/*.json; research_notes/*.md; final selected visualization snapshots',
            'synthetic_stats aggregate CSVs and original2000x512 multiplicities',
            'bootstrap/g1,g2,g3: original2001 fold score arrays and their receipts; g1/counts original5 multiplicity NPZs',
            'all23 classifier_statistics complete CSV/NPZ/DONE; optionally original synthetic per-seed summary NPZs, never real point OOF',
            'all12 conditional_statistics/support_partition/normalizer completed CSV/NPZ/DONE',
            'all4 crossed_H_length_comparison complete CSV/NPZ/DONE',
            'bootstrap_support per-H exact-support CSV/DONE; normalizer H-parameter CSV/NPZ/DONE',
            'g3_full_gradient fivefold objective/gradient CSV/DONE, no large training feature arrays',
            'reproduction_fixture exact synthetic PREDICTIONS.npz and FIXTURE.json',
            'reviewed audit JSON/XML receipts; full regression summary/XML and guard receipt'],
        'unconditionally_excluded':['raw competition observations','Shadow/Seal values','credentials','large features',
            'individual RP event/tau tables','full real point OOF','old archives','Numba caches','browser profiles',
            'temporary reproduction/extraction directories','worker checkpoints and per-point training caches'],
        'archive_verification':'An exact file allowlist and SHA256/byte manifest are materialized before export. A fresh extraction verifies every member and rejects traversal, Windows aliases, duplicate names, linked entries and manifest mismatches.',
        'fresh_extracted_reproduction':'Use the exact original reproduce_research_bundle.py and fixture; run90 paths in an isolated child with original root reads denied. The original fixture binds the reproducer source.',
        'delivery_receipt_scope':'The final archive SHA and its successful extraction/replay receipt are external to the archive to avoid a self-hash cycle. The embedded FINAL_AUDIT covers completed computation and original isolated replay; FINAL_DELIVERY_AUDIT binds the delivered archive bytes.',
        'desktop_policy':'Fresh NEXT7 desktop directory, exactly20 copied files; preserve all existing NEXT6 files and folders.',
        'final_NEXT7_roles':[],'retained_primary':'FROZEN_BL80',
        'diagnostic_recipe_retention':'Only I4, G2_FULL and G3 research bundles for already frozen fivefold reproduction, plus common base components. No deployment promotion and no retention of all15 candidates as selected roles.'})
    note('Handoff content plan recorded','Explicit result inventory and isolated ZIP reproduction contract prepared','REPRODUCTION_AND_RUNTIME')


if __name__=='__main__':main()
