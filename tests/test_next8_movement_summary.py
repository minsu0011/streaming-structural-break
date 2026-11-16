"""Adversarial repeated-fold, matched-control and empty-random-graph checks."""
import src.next8
import numpy as np
from src.next8.movement_summary import collect_comparison,supported_improvement,transition_evidence,edge_evidence


def panels(definitions):
    result=[]
    for fold in range(5):
        rows=[];arrays={}
        for k,(reference,candidate,kind,metric,gains,informative,domain,family) in enumerate(definitions):
            key=f'C{k:04d}';values=np.full(64,gains[fold]);arrays[key+'_group']=np.arange(64)+fold*64;arrays[key+'_gain']=values
            rows.append({'reference':reference,'candidate':candidate,'target':kind,'metric':metric,'group_array_key':key,
                'domain':domain,'domain_family':family,'informative_random_contrast':informative,'status':'SUPPORTED','gain':gains[fold]})
        result.append((fold,{'comparisons':rows},arrays))
    return result


def test_contrary_fifth_fold_cannot_be_dropped():
    p=panels([('D0_delta','D1_delta','delta','MSE',[.02,.02,.02,.02,-.20],True,'ALL','ALL')])
    r=collect_comparison(p,'D0_delta','D1_delta')
    assert r['included_folds']==list(range(5)) and r['positive_folds']==4 and r['mean_gain']<0
    assert not supported_improvement(r)


def test_graph_requires_tabular_and_informative_random_contrasts():
    refs=('D0_state_persistence','D0_state_majority','T1_state','T2_state','T4_state')
    definitions=[(r,'T3_state','state','LOGLOSS',[.01]*5,r!='T4_state','ALL','ALL') for r in refs]
    result=transition_evidence(panels(definitions))
    assert not result['graph_value_before_Holm'] and not result['random_control_assessable']
    definitions[-1]=('T4_state','T3_state','state','LOGLOSS',[.01]*5,True,'ALL','ALL')
    definitions[-2]=('T2_state','T3_state','state','LOGLOSS',[0.]*5,True,'ALL','ALL')
    result=transition_evidence(panels(definitions))
    assert result['random_control_assessable'] and not result['T2_comparison_passed'] and not result['graph_value_before_Holm']


def test_cross_edge_requires_exact_removal_not_only_self_gain():
    refs=('D1_delta','D5_delta','D5_REMOVE_F021_delta')
    definitions=[(r,'D5_CROSS_delta','delta','MSE',[.01]*5,True,'ALL','ALL') for r in refs]
    for family in ('TIME','H_LENGTH'):
        for domain in range(2):
            definitions.extend((r,'D5_CROSS_delta','delta','MSE',[.01]*5,True,family+str(domain),family) for r in refs[1:])
    result=edge_evidence(panels(definitions),21,{i:.01 for i in range(5)})
    assert result['cross_support_before_Holm']
    definitions[2]=('D5_REMOVE_F021_delta','D5_CROSS_delta','delta','MSE',[0.]*5,True,'ALL','ALL')
    result=edge_evidence(panels(definitions),21,{i:.01 for i in range(5)})
    assert not result['cross_support_before_Holm'] and result['p_value']==1.
