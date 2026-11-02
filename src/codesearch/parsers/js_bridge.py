"""Run the Node extractor (ts_parser.mjs) over a batch of JS/TS sources."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("ts_parser.mjs")


def available() -> bool:
    return shutil.which("node") is not None and SCRIPT.exists()


def parse_batch(jobs: list[dict]) -> dict[str, dict]:
    """jobs: [{id, text, ts, offset}] -> {id: result-or-{"error": str}}.

    One Node process per batch; spawning per file would dominate the index time."""
    if not jobs:
        return {}
    if not available():
        return {j["id"]: {"error": "node or ts_parser.mjs not available"} for j in jobs}
    proc = subprocess.run(["node", str(SCRIPT)], input=json.dumps(jobs).encode("utf-8"),
                          capture_output=True, cwd=str(SCRIPT.parent))
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip() or f"node exited {proc.returncode}"
        return {j["id"]: {"error": msg} for j in jobs}
    out: dict[str, dict] = {}
    for item in json.loads(proc.stdout.decode("utf-8")):
        out[item["id"]] = item["result"] if "result" in item else {"error": item.get("error", "?")}
    return out
