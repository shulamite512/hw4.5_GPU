"""Append one entry per script run to RUN_LOG.txt (human-readable) and results/run_log.jsonl (for GPU-hours).

Import it first thing in a top-level script:  import runlog  # noqa: F401
Records start/end time, duration, the exact command, GPU name + UUID, host, and whether the run finished,
failed or was interrupted. Child probe processes (bench_one.py) are deliberately not logged individually;
their time is inside the parent's duration.
"""
import atexit
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(ROOT, "RUN_LOG.txt")
JSONL = os.path.join(ROOT, "results", "run_log.jsonl")

_start = time.time()
_status = {"value": "completed"}


def gpu_info():
    """[(index, name, uuid), ...] from nvidia-smi; respects CUDA_VISIBLE_DEVICES when set."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        rows = [tuple(x.strip() for x in line.split(",")) for line in out.strip().splitlines()]
    except (OSError, subprocess.SubprocessError):
        return []
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        keep = {v.strip() for v in vis.split(",")}
        rows = [r for r in rows if r[0] in keep or r[2] in keep]
    return rows


_gpus = gpu_info()
_prev_hook = sys.excepthook


def _hook(exc_type, exc, tb):
    _status["value"] = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else f"failed: {exc_type.__name__}"
    _prev_hook(exc_type, exc, tb)


sys.excepthook = _hook


@atexit.register
def _write():
    if "--vllm-child" in sys.argv:  # hw45_partD_kv_cache.py's helper process; the parent is logged
        return
    end = time.time()
    entry = dict(
        start=datetime.fromtimestamp(_start).isoformat(timespec="seconds"),
        end=datetime.fromtimestamp(end).isoformat(timespec="seconds"),
        duration_s=round(end - _start, 1),
        command="python " + " ".join(shlex.quote(a) for a in sys.argv),
        status=_status["value"],
        host=platform.node(), os=platform.platform(terse=True),
        rank=os.environ.get("RANK"), world_size=os.environ.get("WORLD_SIZE"),
        gpus=[dict(index=i, name=n, uuid=u) for i, n, u in _gpus],
    )
    if entry["rank"] not in (None, "0"):  # DDP: rank 0 writes for the whole job
        return
    os.makedirs(os.path.dirname(JSONL), exist_ok=True)
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    gpus = "; ".join(f"{g['name']} {g['uuid']}" for g in entry["gpus"]) or "no GPU found"
    with open(TXT, "a", encoding="utf-8") as f:
        f.write(f"[{entry['start']} -> {entry['end']}] {entry['duration_s']:>8.1f} s  {entry['status']:<12} "
                f"{entry['command']}\n    host={entry['host']} world={entry['world_size'] or 1} gpus: {gpus}\n")
