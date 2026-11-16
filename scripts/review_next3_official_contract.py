"""Read public official source metadata without executing the SDK or its runner."""
from pathlib import Path
import ast
import hashlib
import importlib.metadata
import json
import sys
import urllib.parse
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now

DOC = 'https://docs.crunchdao.com/competitions/competitions/structural-break-real-time'
ALLOWED = {'api.github.com', 'raw.githubusercontent.com', 'docs.crunchdao.com'}


def retrieve(url):
    if urllib.parse.urlsplit(url).hostname not in ALLOWED:
        raise RuntimeError('Only public official source hosts are allowed')
    request = urllib.request.Request(url, headers={'User-Agent': 'StructuralBreak-NEXT3-contract-review',
        'Accept': 'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
    with urllib.request.urlopen(request, timeout=25) as response:
        if urllib.parse.urlsplit(response.url).hostname not in ALLOWED:
            raise RuntimeError('Unexpected public-source redirect')
        content = response.read(2_000_001)
    if len(content) > 2_000_000:
        raise RuntimeError('Public source exceeds review bound')
    return content


def functions(content):
    tree = ast.parse(content.decode('utf-8-sig'))
    return {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def main():
    out = BASE/'maintenance/official_contract_review_v1'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'START.json', {'created_utc': utc_now(), 'kind': 'PUBLIC_SOURCE_AND_LOCAL_RECEIPT_MAINTENANCE',
        'script_sha256': digest(Path(__file__)), 'SDK_executed': False, 'competition_data_requested': False}, immutable=True)
    manifest_path = ROOT/'vendor/official/SOURCE_MANIFEST.json'; manifest = read_json(manifest_path)
    snapshots = {}
    for source in manifest['sources']:
        path = ROOT/source['path']
        assert digest(path) == source['sha256']
        snapshots[path.relative_to(ROOT).as_posix()] = source['sha256']
    current = {}; public_errors = []
    for name in ('runner.py', 'scoring.py'):
        relative = 'competitions/structural-break-real-time/scoring/'+name
        try:
            query = urllib.parse.urlencode({'path': relative, 'per_page': 1})
            commits_url = 'https://api.github.com/repos/crunchdao/competitions/commits?'+query
            commit = json.loads(retrieve(commits_url))[0]['sha']
            if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
                raise RuntimeError('Invalid public commit identity')
            url = 'https://raw.githubusercontent.com/crunchdao/competitions/'+commit+'/'+relative
            content = retrieve(url); local = (ROOT/'vendor/official'/name).read_bytes()
            old_ast, new_ast = functions(local), functions(content)
            current[name] = {'commit': commit, 'url': url, 'retrieved_utc': utc_now(),
                'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                'frozen_sha256': hashlib.sha256(local).hexdigest(), 'byte_identical': content == local,
                'top_level_AST_identical': old_ast == new_ast,
                'changed_top_level_definitions': sorted(k for k in old_ast.keys() | new_ast.keys()
                    if old_ast.get(k) != new_ast.get(k))}
        except Exception as error:
            public_errors.append({'source': relative, 'type': type(error).__name__, 'detail': str(error)})
    documentation = {'url': DOC, 'agent_reviewed_utc': utc_now(), 'archived_full_text': False}
    try:
        content = retrieve(DOC+'.md')
        documentation.update(retrieved_utc=utc_now(), sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    except Exception as error:
        public_errors.append({'source': DOC, 'type': type(error).__name__, 'detail': str(error)})
    sdk = {}
    for name in ('crunch-cli', 'crunch-convert', 'crunch-encrypt'):
        distribution = importlib.metadata.distribution(name)
        sdk[name] = {'version': distribution.version, 'home_page': distribution.metadata.get('Home-page'),
            'inspection': 'Installed distribution metadata only; package not imported or executed'}
    assert sdk['crunch-cli']['version'] == '12.0.2'
    receipts = {}
    for role, relative in (
        ('DISTILLED', 'artifacts/next2/canonical_api_all_dev/DISTILLED/RESULTS.json'),
        ('PERFORMANCE_LEAN', 'artifacts/next3/lean_public_api_all_dev/RESULTS.json')):
        record = read_json(ROOT/relative)
        assert record['status'] == 'PASS' and record['series'] == 8000 and record['points'] == 4014405
        for key in ('first_none_before_any_dataset_access', 'one_current_point_per_output', 'no_length_or_suffix_access'):
            assert record[key] is True
        assert record['persistent_filesystem_write_attempts'] == record['seal_rows'] == 0
        assert record['reverse_order_reset_series'] == 128
        receipts[role] = {'path': relative, 'sha256': digest(ROOT/relative),
            'points': record['points'], 'series': record['series'], 'reverse_order_reset_series': 128,
            'output_fingerprint_sha256': record['output_fingerprint_sha256']}
    entry = ROOT/'next3_lean_performance_main.py'; tree = ast.parse(entry.read_text(encoding='utf-8-sig'))
    parallelism = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'INFER_PARALLELISM' for target in node.targets)
        and isinstance(node.value, ast.Constant)]
    assert parallelism == [1]
    result = {'status': 'PASS_LOCAL_CONTRACT_REVIEW', 'created_utc': utc_now(),
        'current_official_source_comparison': current, 'public_source_errors': public_errors,
        'public_source_comparison_complete': len(current) == 2 and not public_errors,
        'documentation': documentation, 'frozen_source_manifest_sha256': digest(manifest_path),
        'unchanged_frozen_source_snapshots': snapshots, 'installed_SDK_metadata': sdk,
        'local_public_API_evidence': receipts, 'entry_sha256': digest(entry), 'INFER_PARALLELISM': 1,
        'SDK_test_calls': 0, 'reduced_new_usage': 0, 'public_submissions': 0, 'cloud_runs': 0,
        'raw_data_accesses': 0, 'seal_values_accessed': 0, 'new_fits': 0,
        'scope': 'Public documentation/source drift review and already completed local API proofs. No claim of cloud execution, Linux exactness, current quota availability, or live SDK end-to-end validation.',
        'transport': 'Frozen official runner serializes historical values, online observations and returned predictions as float32. Local output fingerprints are float32 bytes; direct timing excludes SDK IPC.',
        'script_sha256': digest(Path(__file__))}
    atomic_json(out/'RESULTS.json', result, immutable=True)
    progress('Official contract maintenance reviewed',
        'Public source drift metadata and frozen local API evidence only; no SDK test, data acquisition, cloud run or seal access')
    print('OFFICIAL CONTRACT REVIEW', len(current), 'current sources;', len(public_errors), 'public retrieval errors;',
        {name: item['byte_identical'] for name, item in current.items()}, flush=True)


if __name__ == '__main__': main()
