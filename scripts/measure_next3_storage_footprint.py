"""Read filesystem metadata only; retain logical sizes without cleanup or content reads."""
from pathlib import Path
import argparse
import collections
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.maintenance import resource_snapshot


def metadata_tree(root, *, excluded_names=()):
    buckets = collections.defaultdict(lambda:{'files':0,'logical_bytes':0})
    extensions = collections.defaultdict(lambda:{'files':0,'logical_bytes':0})
    skipped = []; vanished = []; largest = []
    for parent,directories,files in os.walk(root,followlinks=False):
        parent = Path(parent)
        for name in list(directories):
            path = parent/name
            if name in excluded_names or path.is_symlink() or path.is_junction():
                directories.remove(name); skipped.append(path.relative_to(root).as_posix())
        for name in files:
            path = parent/name
            if path.is_symlink() or path.is_junction():
                skipped.append(path.relative_to(root).as_posix());continue
            try:size=path.stat().st_size
            except FileNotFoundError:
                vanished.append(path.relative_to(root).as_posix());continue
            relative=path.relative_to(root); parts=relative.parts
            depth=3 if parts[:2]==('artifacts','next3') else 2 if parts[0]=='artifacts' else 1
            group='/'.join(parts[:min(depth,len(parts)-1)]) if len(parts)>1 else '(root files)'
            for counter in (buckets[group],extensions[path.suffix.lower() or '(none)']):
                counter['files']+=1;counter['logical_bytes']+=size
            largest.append({'path':relative.as_posix(),'logical_bytes':size})
    return {'directory':str(root),'groups':dict(sorted(buckets.items())),
        'extensions':dict(sorted(extensions.items())),
        'files':sum(v['files'] for v in buckets.values()),
        'logical_bytes':sum(v['logical_bytes'] for v in buckets.values()),
        'largest_files_by_metadata':sorted(largest,key=lambda x:x['logical_bytes'],reverse=True)[:12],
        'linked_or_explicitly_excluded_paths':skipped,'transient_files_vanished_during_scan':vanished}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.label):
        raise ValueError('Safe unique storage measurement label required')
    out=BASE/'maintenance/storage_footprint'/args.label;out.mkdir(parents=True,exist_ok=False)
    started=utc_now();phase='FINAL_AUDIT' if read_json(BASE/'SESSION.json').get('audit_started_utc') else 'ACTIVE_RESEARCH'
    progress('NEXT3 storage metadata snapshot','Read logical file sizes and skipped links only; no content reads, cleanup or environment changes',phase=phase)
    project=metadata_tree(ROOT,excluded_names={'.git'})
    siblings=[]
    for path in sorted(ROOT.parent.glob('SB_NEXT3_*')):
        if path.is_dir() and not path.is_symlink() and not path.is_junction():siblings.append(metadata_tree(path))
    resources=resource_snapshot()
    resources['measurement_worker_RSS_bytes']=resources.pop('supervisor_RSS_bytes')
    resources['measurement_worker_child_process_count']=resources.pop('child_process_count')
    resources['measurement_worker_pid']=os.getpid()
    result={'status':'PASS_READ_ONLY_STORAGE_METADATA_SNAPSHOT','started_utc':started,'created_utc':utc_now(),
        'project':project,'ordinary_NEXT3_siblings':siblings,'resources_after_scan':resources,
        'source_sha256':digest(Path(__file__)),'files_deleted_or_moved':0,'file_contents_read_for_measurement':0,
        'seal_observations_labels_or_predictions_read':False,'new_fits':0,'new_quality_scores':0,
        'scope':'Logical file lengths observed across a non-atomic metadata walk. Hard-linked/copied paths are counted per name; physical allocation, global OS temp, other projects and excluded links are not measured. Historical cache sizes are not reconstructed or backfilled.'}
    atomic_json(out/'RESULTS.json',result,immutable=True)
    progress('NEXT3 storage metadata measured',str(project['files'])+' project paths; '+str(project['logical_bytes'])+' logical bytes; no cleanup or content access',phase=phase)
    print('NEXT3 STORAGE METADATA PASS',project['files'],project['logical_bytes'],len(siblings),flush=True)


if __name__ == '__main__': main()
