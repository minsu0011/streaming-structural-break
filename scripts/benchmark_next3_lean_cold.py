"""Fixed three-pair fresh-process E003 memory/startup cost test."""
from pathlib import Path
import os
import statistics
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from scripts.benchmark_next2_runtime import assert_isolated


def main():
    assert_isolated(); lock = read_json(BASE/'engineering/lean_shared/IMPLEMENTATION_LOCK_CORRECTION_001.json')
    for p, sha in lock['source_sha256'].items(): assert digest(ROOT/p) == sha, p
    out = BASE/'engineering/lean_shared/cold'; out.mkdir(parents=True, exist_ok=False)
    orders = [('PACKED_SHARED', 'LEAN_PACKED_SHARED'), ('LEAN_PACKED_SHARED', 'PACKED_SHARED'), ('PACKED_SHARED', 'LEAN_PACKED_SHARED')]
    atomic_json(out/'POLICY.json', {'created_utc': utc_now(), 'orders': orders,
        'implementation_lock_sha256': digest(BASE/'engineering/lean_shared/IMPLEMENTATION_LOCK_CORRECTION_001.json'),
        'protocol_sha256': digest(ROOT/'docs/NEXT3_LEAN_RUNTIME_PROTOCOL.md')}, immutable=True)
    pairs = []; receipts = {}
    for trial, order in enumerate(orders):
        result = {}
        for variant in order:
            assert_isolated(); path = out/f'trial_{trial}_{variant}.json'
            log_path = out/f'trial_{trial}_{variant}.log'
            with log_path.open('x', encoding='utf-8') as log:
                command = [sys.executable, '-u', str(ROOT/'scripts/probe_next3_lean_cold_process.py'), '--variant', variant, '--output', str(path)]
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                while child.poll() is None:
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        progress('E003 cold process measurement', f'pair {trial} {variant}: child active', experiment='E003_LEAN_SHARED')
                if child.returncode: raise RuntimeError('Cold subprocess failed; preserve '+str(log_path))
            result[variant] = read_json(path); assert result[variant]['status'] == 'PASS'
            receipts[path.name] = digest(path); receipts[log_path.name] = digest(log_path)
            progress('E003 cold process measurement', f'pair {trial} {variant} complete', experiment='E003_LEAN_SHARED')
        a, b = result['PACKED_SHARED'], result['LEAN_PACKED_SHARED']
        pairs.append({'trial': trial, 'RSS_ratio': b['post_warm_memory']['rss']/a['post_warm_memory']['rss'],
            'first_none_ratio': b['first_warmed_none_seconds']/a['first_warmed_none_seconds'],
            'packed_RSS_bytes': a['post_warm_memory']['rss'], 'lean_RSS_bytes': b['post_warm_memory']['rss'],
            'packed_first_none_seconds': a['first_warmed_none_seconds'], 'lean_first_none_seconds': b['first_warmed_none_seconds']})
    # Compare medians of the prospectively declared costs; retain paired ratios too.
    rss_ratio = statistics.median(p['lean_RSS_bytes'] for p in pairs)/statistics.median(p['packed_RSS_bytes'] for p in pairs)
    time_ratio = statistics.median(p['lean_first_none_seconds'] for p in pairs)/statistics.median(p['packed_first_none_seconds'] for p in pairs)
    memory_route = rss_ratio <= .8 and all(p['RSS_ratio'] < 1 for p in pairs) and time_ratio <= 1.10
    startup_route = time_ratio <= .8 and rss_ratio <= 1.
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'pairs': pairs,
        'median_RSS_ratio': rss_ratio, 'median_first_none_ratio': time_ratio,
        'memory_route_pass': memory_route, 'startup_route_pass': startup_route,
        'cold_cost_gate_pass': memory_route or startup_route, 'files_sha256': receipts,
        'new_fits': 0, 'seal_rows': 0, 'fresh_process_trials': 6,
        'limitation': 'Host-conditional OS working set and cold JIT/import measurements; no claim of cross-machine guarantee'}, immutable=True)
    if memory_route or startup_route:
        atomic_json(out/'COST_GATE_PASS.json', {'status': 'PASS', 'result_sha256': digest(out/'RESULTS.json')}, immutable=True)
    progress('E003 cold cost gate complete', f'RSS ratio {rss_ratio:.6f}, first None ratio {time_ratio:.6f}', experiment='E003_LEAN_SHARED')


if __name__ == '__main__': main()
