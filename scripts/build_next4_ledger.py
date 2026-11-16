"""Reconcile preserved prior ledgers and primitive NEXT4 fits without publication."""
import argparse
import csv
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress


def read_table(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);rows=list(reader)
        return reader.fieldnames,rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--fit-inventory-tag',required=True);args=parser.parse_args()
    for value in (args.tag,args.fit_inventory_tag):
        if not re.fullmatch('[A-Za-z0-9_]+',value):raise ValueError('Safe immutable tag required')
    before=BASE/'preflight/retained/EXPERIMENT_LEDGER.csv'
    prior=ROOT/'artifacts/next3/final_tables/prepared_v2/EXPERIMENT_LEDGER.csv'
    fields,original=read_table(before);prior_fields,rows=read_table(prior)
    assert len(original)==213 and len(rows)==242 and fields==prior_fields
    assert rows[:len(original)]==original
    assert digest(before)=='819c2b2385f0bc5b061694f78224853ac651cb7a9ae1857cd8238ccca1a64c20'
    assert digest(prior)=='995c77f06d3e4b9dec979988ca447faf325a61bb7f47287f3184b6eedbdc34fe'
    old_session=ROOT/'artifacts/next3/SESSION.json';assert read_json(old_session)['status']=='STOPPED_BY_USER'
    pointer=read_json(ROOT/'artifacts/next3/runtime/USER_STOP_POINTER.json')
    stop=Path(pointer['receipt']);assert digest(stop)==pointer['sha256']
    inventory_path=BASE/'fit_inventory'/args.fit_inventory_tag/'RESULTS.json'
    inventory=read_json(inventory_path);assert inventory['status']=='PASS'
    fits_path=inventory_path.parent/'PRIMITIVE_FITS.csv'
    assert digest(fits_path)==inventory['files_sha256']['PRIMITIVE_FITS.csv']
    _,fits=read_table(fits_path)
    assert sum(r['completed']=='True' for r in fits)==inventory['real_DEV_learner_fit_completions']
    extras=['next4_candidate_id','next4_evidence_scope','session_finalization_status',
            'prior_session_stop_receipt_sha256','new_meta_fits','frozen_base_new_fits',
            'primitive_fit_inventory_sha256','source_result_path']
    fields=[*fields,*extras]
    # Keep every prior cell unchanged; additional columns describe the stopped context.
    for row in rows[len(original):]:
        row.update(session_finalization_status='NEXT3_STOPPED_BY_USER_NOT_FINAL_AUDITED',
                   prior_session_stop_receipt_sha256=digest(stop))
    evidence={str(p.relative_to(ROOT).as_posix()):digest(p) for p in (before,prior,old_session,inventory_path,fits_path)}
    used_fits=set();candidate_count=0
    def consume(indices):
        assert not used_fits.intersection(indices)
        used_fits.update(indices)
        return sum(fits[i]['completed']=='True' for i in indices)
    for registration in sorted((BASE/'registrations').glob('*.json')):
        reg=read_json(registration);cid=reg['experiment_id'];candidate_count+=1
        evidence[registration.relative_to(ROOT).as_posix()]=digest(registration)
        indices=[i for i,r in enumerate(fits) if r['candidate_or_control']==cid and r['phase'].startswith('selection_')]
        row={'experiment_id':'NEXT4_'+cid,'next4_candidate_id':cid,'generation':'NEXT4',
             'phase':'FIXED_SCREEN_AND_PRIMARY_FULL','hypothesis':reg['hypothesis'],
             'ledger_section':'NEXT4_SCIENTIFIC_RECIPES','new_fits':consume(indices),
             'new_meta_fits':sum(fits[i]['meta_fit']=='True' for i in indices),
             'frozen_base_new_fits':0,'seal_rows':0,'reduced_new_uses':0,
             'primitive_fit_inventory_sha256':digest(inventory_path),
             'next4_evidence_scope':'Existing grouped DEV, conditional fitted OOF; no held-out seal evidence',
             'session_finalization_status':'NEXT4_RESEARCH_SNAPSHOT_NOT_FINAL_SELECTION'}
        screen=BASE/'experiments'/cid/'screen/RESULTS.json';sr=read_json(screen)
        evidence[screen.relative_to(ROOT).as_posix()]=digest(screen)
        row.update(screen_mean=sr['mean'],screen_delta=sr['mean_delta'])
        full=BASE/'experiments'/cid/'full/RESULTS.json'
        if full.exists():
            fr=read_json(full);evidence[full.relative_to(ROOT).as_posix()]=digest(full)
            bp=BASE/'bootstrap'/cid/'RESULTS.json';br=read_json(bp)
            evidence[bp.relative_to(ROOT).as_posix()]=digest(bp);paired=br['paired_all_time']
            row.update(primary_cv_mean=fr['mean'],primary_cv_median=fr['median'],primary_cv_worst=fr['worst_fold'],
                delta_vs_primary=fr['mean_delta'],feature_count=fr['feature_count'],
                bootstrap_replicates=br['draws'],bootstrap_probability=paired['probability_delta_positive'],
                bootstrap_ci_low=paired['ci_2_5'],bootstrap_ci_high=paired['ci_97_5'],
                performance_gate=br['performance_primary_and_bootstrap_admission'],
                status='FULL_COMPLETE_PRIMARY_PERFORMANCE_REJECTED' if not br['performance_primary_and_bootstrap_admission'] else 'FULL_COMPLETE_ADDITIONAL_GATES_REQUIRED',
                reason='Primary performance gate and separate robustness role must not be conflated; see final DEV_CANDIDATES_NEXT4.json when published.',
                source_result_path=full.relative_to(ROOT).as_posix(),experiment_report_sha256=digest(full))
        else:
            assert sr['disposition']=='SCREEN_FAIL_STOP'
            row.update(status='SCREEN_FAIL_STOP',reason='Frozen screen failed; no full fit or full score.',
                source_result_path=screen.relative_to(ROOT).as_posix(),experiment_report_sha256=digest(screen))
        rows.append(row)
    assert candidate_count==16
    # Group only genuinely distinct primitive fits; no parent receipt is a second fit.
    groups={}
    for i,fit in enumerate(fits):
        if i not in used_fits:groups.setdefault((fit['candidate_or_control'],fit['phase']),[]).append(i)
    for (cid,phase),indices in sorted(groups.items()):
        rows.append({'experiment_id':'NEXT4_FIT_ACCOUNTING_'+cid+'_'+phase,'generation':'NEXT4',
            'next4_candidate_id':cid,'phase':phase,'ledger_section':'NEXT4_ADDITIONAL_FIT_ACCOUNTING',
            'new_fits':consume(indices),'new_meta_fits':sum(fits[i]['meta_fit']=='True' for i in indices),
            'frozen_base_new_fits':0,'seal_rows':0,'reduced_new_uses':0,
            'status':'COMPLETE_NOT_NEW_SCIENTIFIC_IDENTITY' if all(fits[i]['completed']=='True' for i in indices) else 'INCOMPLETE_FIT_RECEIPT',
            'reason':'Primitive additional real-DEV learner fits only; no bootstrap, calibration, parent-summary or reused-base double count.',
            'next4_evidence_scope':'Diagnostic all-DEV, negative control, unchanged ALT/H refit or fresh reproduction as explicitly named by phase',
            'primitive_fit_inventory_sha256':digest(inventory_path),
            'session_finalization_status':'NEXT4_RESEARCH_SNAPSHOT_NOT_FINAL_SELECTION'})
    assert used_fits==set(range(len(fits)))
    for name,reason in [('F1','DEDUPLICATE_EXISTING_FIXED_COMPOSITION_NO_ADMITTED_THIRD_EXPERT'),
                        ('F2','CONDITIONAL_SKIP_NO_ADMITTED_COMPONENT_PAIR')]:
        rows.append({'experiment_id':'NEXT4_'+name+'_CONDITIONAL_FAMILY_CLOSURE','generation':'NEXT4',
            'phase':'CONDITIONAL_MENU','ledger_section':'NEXT4_ZERO_FIT_MENU_CLOSURE','new_fits':0,
            'frozen_base_new_fits':0,'seal_rows':0,'reduced_new_uses':0,'status':reason,
            'reason':'No new score, candidate identity or fit. Existing fixed source compositions were already accounted once.',
            'source_result_path':'artifacts/next4/families/F_SELECTION.json',
            'experiment_report_sha256':digest(BASE/'families/F_SELECTION.json')})
    assert len({r['experiment_id'] for r in rows})==len(rows)
    assert sum(int(r.get('new_fits',0)) for r in rows[242:])==inventory['real_DEV_learner_fit_completions']
    out=BASE/'ledger'/args.tag;out.mkdir(parents=True,exist_ok=False)
    output=out/'EXPERIMENT_LEDGER.csv'
    with output.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    _,written=read_table(output)
    assert [{k:r[k] for k in prior_fields} for r in written[:242]]==read_table(prior)[1]
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'rows':len(rows),
        'original_root_rows_preserved_exact_cells':213,'unpublished_stopped_NEXT3_rows_preserved_exact_cells':29,
        'NEXT3_stop_context_columns_added':True,'NEXT4_scientific_recipe_rows':16,
        'NEXT4_conditional_zero_score_menu_rows':2,'NEXT4_real_DEV_completed_fits':inventory['real_DEV_learner_fit_completions'],
        'source_sha256':digest(Path(__file__)),'evidence_sha256':evidence,'new_fits_in_assembly':0,
        'root_ledger_modified':False,'final_role_selection_claimed':False,
        'files_sha256':{'EXPERIMENT_LEDGER.csv':digest(output)}},immutable=True)
    progress('Stopped-session ledger reconciliation complete','Prior 242 rows preserve every existing cell; NEXT4 primitive fits reconciled without parent double counting',counts_as_research=False)
    print('NEXT4 LEDGER SNAPSHOT',len(rows),'rows;',inventory['real_DEV_learner_fit_completions'],'new real-DEV fits',flush=True)


if __name__=='__main__':main()
