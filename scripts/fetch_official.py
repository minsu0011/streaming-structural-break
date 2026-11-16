"""Download only official contract/scorer/quickstarter sources, pinned to a commit."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REV = "ddb0d2240aa885965a0cb5d16df04dae70c66679"
PATHS = ["scoring/runner.py", "scoring/scoring.py", "quickstarters/baseline/baseline.ipynb", "quickstarters/baseline/quickstarter.json"]

def main():
    records = []
    for relative in PATHS:
        url = f"https://raw.githubusercontent.com/crunchdao/competitions/{REV}/competitions/structural-break-real-time/{relative}"
        data = urlopen(url, timeout=60).read()
        destination = ROOT / "vendor/official" / Path(relative).name
        destination.write_bytes(data)
        records.append({"path": str(destination.relative_to(ROOT)), "url": url, "commit": REV, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    (ROOT / "vendor/official/SOURCE_MANIFEST.json").write_text(json.dumps({"retrieved_utc": datetime.now(timezone.utc).isoformat(), "sources": records}, indent=2), encoding="utf-8")
    notebook = json.loads((ROOT / "vendor/official/baseline.ipynb").read_text(encoding="utf-8"))
    (ROOT / "vendor/official/baseline_readable.txt").write_text("\n\n".join("".join(c["source"]) for c in notebook["cells"]), encoding="utf-8")
    print("Official source snapshot saved:", REV)

if __name__ == "__main__":
    main()
