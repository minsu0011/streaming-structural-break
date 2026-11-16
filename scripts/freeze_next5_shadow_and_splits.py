"""Outcome-blind NEXT5 ResearchPool/Shadow and grouped validation locks."""
from collections import defaultdict, Counter
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next.guard import SealGuard

SALT='StructuralBreak_NEXT5_prospective_shadow_v1_20260911'


def hashed(purpose, identifier):
    return hashlib.sha256((SALT+':'+purpose+':'+str(int(identifier))).encode('ascii')).hexdigest()


def allocate(groups, purpose, nfolds=5):
    loads=[0]*nfolds;assigned={}
    for gid,ids in sorted(groups.items(),key=lambda kv:(hashed(purpose,kv[0]),kv[0])):
        fold=min(range(nfolds),key=lambda k:(loads[k],k))
        loads[fold]+=len(ids)
        for sid in ids:assigned[str(sid)]=fold
    return assigned,loads


def main():
    out=BASE/'locks';out.mkdir(parents=True,exist_ok=False)
    # The salt and rules are recorded before looking at membership or any outcomes.
    atomic_json(out/'ALLOCATION_POLICY.json',{'created_utc':utc_now(),'salt':SALT,
        'shadow_fraction':.2,'algorithm':'Hash salt:shadow:minimum_integer_ID per existing duplicate group; choose whole-group hash prefix nearest 20% series count, tie to smaller prefix.',
        'fold_algorithm':'Independent purpose hash order of research groups; place each in the currently smallest of five folds, tie to lowest fold index.',
        'screen_algorithm':'First whole groups in screen hash order within each PRIMARY fold, nearest 512 series per fold.',
        'purpose_salts':['shadow','PRIMARY','ALT_A','ALT_B','screen'],
        'uses_labels_lengths_features_scores':False,'source_sha256':digest(Path(__file__))},immutable=True)
    guard=SealGuard(ROOT);groups=defaultdict(list)
    for sid in sorted(guard.dev_ids):groups[guard.split.groups[sid]].append(sid)
    assert all(min(ids)==gid for gid,ids in groups.items())
    ordered=sorted(groups,key=lambda gid:(hashed('shadow',gid),gid));target=round(.2*len(guard.dev_ids))
    cumulative=[0]
    for gid in ordered:cumulative.append(cumulative[-1]+len(groups[gid]))
    cut=min(range(1,len(ordered)),key=lambda k:(abs(cumulative[k]-target),cumulative[k]))
    shadow_groups=set(ordered[:cut]);shadow=sorted(i for g in shadow_groups for i in groups[g])
    research=sorted(set(guard.dev_ids)-set(shadow));research_groups={g:v for g,v in groups.items() if g not in shadow_groups}
    assert set(research).isdisjoint(shadow) and set(research)|set(shadow)==set(guard.dev_ids)
    for name,ids in [('research_ids',research),('shadow_ids',shadow)]:
        (out/(name+'.txt')).write_bytes(('\n'.join(map(str,ids))+'\n').encode('ascii'))
    locks={'status':'FROZEN_NEXT5_PROSPECTIVE_SHADOW','created_utc':utc_now(),'salt':SALT,
        'research_series':len(research),'shadow_series':len(shadow),'research_groups':len(research_groups),'shadow_groups':len(shadow_groups),
        'original_DEV_series':8000,'original_LOCAL_SEAL_series':2000,'research_shadow_group_overlap':0,
        'original_assignments_sha256':guard.split.digest,'seal_lock_sha256':guard.lock_sha,
        'research_ids_file':(out/'research_ids.txt').relative_to(ROOT).as_posix(),'research_ids_sha256':digest(out/'research_ids.txt'),
        'shadow_ids_file':(out/'shadow_ids.txt').relative_to(ROOT).as_posix(),'shadow_ids_sha256':digest(out/'shadow_ids.txt'),
        'allocation_policy_sha256':digest(out/'ALLOCATION_POLICY.json'),'source_sha256':digest(Path(__file__)),
        'shadow_opened_at_lock':False,'true_independent_seal':False,
        'historical_contamination':'All 8000 DEV series were used in earlier sessions. This partition only limits NEXT5 prospective adaptive selection.',
        'labels_or_observations_loaded':False}
    atomic_json(ROOT/'NEXT5_SHADOW_LOCK.json',locks,immutable=True)
    mappings={};counts={}
    for purpose in ('PRIMARY','ALT_A','ALT_B'):mappings[purpose],counts[purpose]=allocate(research_groups,purpose)
    screen=[]
    for fold in range(5):
        candidates=sorted((g for g,v in research_groups.items() if mappings['PRIMARY'][str(v[0])]==fold),key=lambda g:(hashed('screen',g),g))
        sizes=[0]
        for gid in candidates:sizes.append(sizes[-1]+len(groups[gid]))
        k=min(range(1,len(sizes)),key=lambda j:(abs(sizes[j]-512),sizes[j]))
        screen.extend(i for g in candidates[:k] for i in groups[g])
    split={'status':'FROZEN_NEXT5_RESEARCH_SPLITS','created_utc':utc_now(),
        'shadow_lock_sha256':digest(ROOT/'NEXT5_SHADOW_LOCK.json'),'folds':5,'mappings':mappings,'fold_series_counts':counts,
        'groups':{str(i):guard.split.groups[i] for i in research},'screen_ids':sorted(screen),'screen_folds':[0,4],
        'screen_series':len(screen),'allocation_uses_outcomes':False,'new_postscore_partitions_permitted':False}
    atomic_json(ROOT/'NEXT5_SPLIT_LOCK.json',split,immutable=True)
    atomic_json(ROOT/'NEXT5_RANDOMNESS_LOCK.json',{'status':'FROZEN_BEFORE_NEXT5_SCORES','created_utc':utc_now(),
        'model_seed':20260908,'bootstrap_seed_base':20260908,'bootstrap_seed_rule':'base + frozen fold index',
        'bootstrap_replicates':2000,'prefix_seed':20260908,'label_permutation_seed':20260908,
        'pair_sampling_seed':20260908,'synthetic_seed':2026090902,'shadow_allocation_salt':SALT,
        'new_seed_or_reseed_after_score_permitted':False,'synthetic_protocol':'Keep original NEXT4 fixed DGP seeds and parameters; training scope will be ResearchPool only.'},immutable=True)
    atomic_json(BASE/'shadow/STATE.json',{'status':'CLOSED','shadow_opened':False,'labels_loaded':False,
        'maximum_finalists':3,'evaluations_completed':0,'created_utc':utc_now()},immutable=True)
    progress('NEXT5 outcome-blind locks frozen',str(len(research))+' ResearchPool / '+str(len(shadow))+' Shadow; five grouped folds and two ALT partitions',phase='PREFLIGHT')
    print('NEXT5 SHADOW/SPLITS LOCKED',len(research),len(shadow),counts,'screen',len(screen),flush=True)


if __name__=='__main__':main()
