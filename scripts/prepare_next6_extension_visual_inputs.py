"""Small aggregate input for raw-free figure reproduction; no model recomputation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.next6_extension.common import *

def main():
    source=BASE/'FRESH_RP_PHASE_SERIES.json';records=read_json(source)['rows'];out=[]
    for name in ('no_break','pre','onset','earlyPost','establishedPost','laterPost'):
        rr=[r for r in records if r['phase']==name];values=np.mean([r['state_fraction'] for r in rr],axis=0)
        out.append({'phase':name,'series':len(rr),**{f'state_{j}_mean_series_fraction':float(v) for j,v in enumerate(values)}})
    csv_write('RP_PHASE_OCCUPANCY_AGGREGATE.csv',out)
    import csv
    with (BASE/'RP_PHASE_OCCUPANCY_AGGREGATE.csv').open(encoding='utf-8-sig') as f:reloaded=list(csv.DictReader(f))
    for a,b in zip(out,reloaded):
        np.testing.assert_array_equal([a[f'state_{j}_mean_series_fraction'] for j in range(7)],[float(b[f'state_{j}_mean_series_fraction']) for j in range(7)])
    save('RAW_FREE_VISUAL_AGGREGATE_DONE.json',{'status':'PASS','created_utc':utc_now(),'source_file_sha256':digest(source),'aggregate_sha256':digest(BASE/'RP_PHASE_OCCUPANCY_AGGREGATE.csv'),'six_phase_rows_only':True,'no_series_ID_or_tau_table':'Aggregate contains counts and seven mean state fractions only.','exact_float_round_trip_and_original_plot_value_parity':True,'source_sha256':digest(Path(__file__))})

if __name__=='__main__':main()
