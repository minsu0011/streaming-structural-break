"""Recover the completed H-only study after a pandas column/method name clash."""
from pathlib import Path
import sys
import json
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.artifacts import sha256,write_json,utc_now
from src.next.progress import checkpoint


def main():
    out = ROOT/'artifacts/next/historical_holdout'
    policy = json.loads((out/'POLICY.json').read_text(encoding='utf-8'))
    files = ['WHITENING_PER_DEV_SERIES.parquet','NORMALIZATION_PER_DEV_SERIES.parquet']
    wf,nf = [pd.read_parquet(out/name) for name in files]
    assert len(wf)==40000 and len(nf)==64000
    summaries=[]
    for frame,key in [(wf,'order'),(nf,'normalization')]:
        for value,group in frame.groupby(key):
            summaries.append({'kind':key,'variant':str(value),'series':len(group),
                'mean_abs_acf1':float(group.acf_1.abs().mean()),'mean_abs_acf8':float(group.acf_8.abs().mean()),
                'mean_abs_squared_acf1':float(group.squared_acf_1.abs().mean()),
                'mean_abs_squared_acf2':float(group.squared_acf_2.abs().mean()),
                'median_kurtosis':float(group['kurtosis'].median()),
                'mean_tail_abs_over_3':float(group.tail_abs_over_3.mean())})
    pd.DataFrame(summaries).to_csv(out/'SUMMARY.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','timestamp':utc_now(),'policy':policy,'summary':summaries,
        'input_files_sha256':{name:sha256(out/name) for name in files},
        'scientific_source_sha256':{name:sha256(ROOT/name) for name in ['src/next/whitening.py','src/next/normalization.py','src/next/regimes.py']},
        'repair':{'failure':'pandas DataFrame.kurtosis method shadowed the kurtosis column in summary only',
                  'fix':'explicit group column indexing; reuse all completed per-series data; no refits or value changes',
                  'original_script_sha256':policy['script_sha256'],
                  'corrected_script_sha256':sha256(ROOT/'scripts/diagnose_next_historical.py'),
                  'summary_script_sha256':sha256(Path(__file__))},
        'seal_rows':0,'actual_online_values_used':False,'targets_used':False,'reduced_new_usage':0,
        'limitation':'H forward holdout is a method diagnostic and is not competition CV; H quarters need not be stationary.'})
    checkpoint(ROOT,'historical-whitening-diagnostics',['H-only holdout: 40000 AR fits, 64000 normalization profiles'],
        ['finite initial candidate queue','historical-null calibration'],
        failures=['Summary-only pandas kurtosis name collision repaired without recomputing completed data'],
        artifacts=['artifacts/next/historical_holdout/SUMMARY.csv'],best=.611292686371357)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__=='__main__':
    main()
