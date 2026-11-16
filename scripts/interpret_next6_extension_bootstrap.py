"""Describe completed preregistered resamples; no fitting or edge selection."""
from pathlib import Path
import sys,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6_extension.common import BASE,save,read_json,digest,utc_now

def rows(name):
    with (BASE/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    assert read_json(BASE/'GRAPH_BOOTSTRAP_FINAL_STATISTICS_DONE.json')['graphs']==1000
    names=['BOOTSTRAP_GRAPH_SIZE_DISTRIBUTION.csv','BOOTSTRAP_EDGE_SELECTION_FREQUENCY.csv','BOOTSTRAP_SELECTED_COEFFICIENT_SIGNS.csv','BOOTSTRAP_FAMILY_SELECTION_FREQUENCY.csv']
    sizes,edges,signs,families=[rows(n) for n in names]
    original=[r for r in edges if r['original_selected']=='True']
    signed=[r for r in signs if r['original_sign']!='']
    assert len(sizes)==1000 and len(original)==len(signed)==45
    fold_summary=[]
    for fold in range(5):
        rr=[r for r in sizes if int(r['fold'])==fold];assert len(rr)==200
        total=sum(int(r['edges']) for r in rr);mechanical=sum(int(r['mechanical_edges']) for r in rr)
        fold_summary.append({'fold':fold,'graphs':200,'edges_min':min(int(r['edges']) for r in rr),'edges_max':max(int(r['edges']) for r in rr),'mean_edges':total/200,'selected_link_occurrences':total,'mechanical_link_occurrences':mechanical,'mechanical_share_of_link_occurrences':mechanical/total})
    pair_summary=[]
    for pair in [('energy','energy'),('energy','conditional_energy'),('energy','tail'),('energy','variance'),('conditional_energy','variance')]:
        rr=[r for r in families if (r['source_family'],r['target_family'])==pair];assert len(rr)==5
        values=[float(r['at_least_one_family_link_frequency']) for r in rr]
        pair_summary.append({'source_family':pair[0],'target_family':pair[1],'minimum_fold_frequency':min(values),'maximum_fold_frequency':max(values),'definition':'At least one selected link across the locked lags, phase choices and source taxonomy; descriptive aggregation only'})
    total=sum(r['selected_link_occurrences'] for r in fold_summary);mechanical=sum(r['mechanical_link_occurrences'] for r in fold_summary)
    save('BOOTSTRAP_INTERPRETATION_SUMMARY.json',{'created_utc':utc_now(),'status':'COMPLETE','graphs':1000,'folds':fold_summary,'family_pairs':pair_summary,'total_selected_link_occurrences':total,'mechanical_selected_link_occurrences':mechanical,'mechanical_share_of_link_occurrences':mechanical/total,'original_edge_fold_pairs':45,'original_selected_frequency_min':min(float(r['selection_frequency']) for r in original),'original_selected_frequency_max':max(float(r['selection_frequency']) for r in original),'original_selected_sign_concordance_min':min(float(r['same_sign_as_original_given_selected']) for r in signed),'original_selected_sign_concordance_max':max(float(r['same_sign_as_original_given_selected']) for r in signed),'selection_and_sign_conditioning':'Sign concordance is conditional on the edge being selected. It is not evidence that its exact lag is stable or that it discriminates a break. Mechanical is the preserved source-lineage audit classification, not a proof for every unmarked link.','inputs_sha256':{n:digest(BASE/n) for n in names},'source_sha256':digest(Path(__file__))})
    print('Bootstrap interpretation complete',total,mechanical,mechanical/total)

if __name__=='__main__':main()
