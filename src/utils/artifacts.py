import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

def commit_id():
    try:
        top=subprocess.check_output(["git","rev-parse","--show-toplevel"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        if Path(top).resolve()==ROOT.resolve():
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):pass
    # A source archive has no .git directory. Never borrow an unrelated parent
    # repository's commit; retain the archived provenance when available.
    manifest=ROOT/'SOURCE_MANIFEST.json'
    return json.loads(manifest.read_text(encoding='utf-8')).get('git_commit') if manifest.exists() else None

LEDGER_FIELDS = ["experiment_id", "timestamp", "git_commit", "feature_version", "feature_families", "model", "params", "sampling_policy", "fold", "ts_auc", "worst_fold", "latency", "memory", "status", "notes"]

def ledger(**record):
    path = ROOT / "experiments/results/EXPERIMENT_LEDGER.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": utc_now(), "git_commit": commit_id(), **record}
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(record)

def checkpoint(number, finished, next_action, **extra):
    write_json(ROOT / f"checkpoints/CHECKPOINT_{number:02d}.json", {
        "timestamp": utc_now(), "git_commit": commit_id(), "finished_tasks": finished,
        "active_candidate": None, "best_dev_score": None, "unresolved_bugs": [],
        "next_action": next_action, **extra})
