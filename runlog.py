"""Append one entry per script run to RUN_LOG.txt (human-readable) and results/run_log.jsonl (for GPU-hours),
and save the run's full terminal output to outputs/<timestamp>_<script>.log.

Import it first thing in a top-level script:  import runlog  # noqa: F401
Records start/end time, duration, the exact command, GPU name + UUID, host, and whether the run finished,
failed or was interrupted. Child probe processes (bench_one.py) are deliberately not logged individually;
their time is inside the parent's duration and their summary lines are printed by the parent. vLLM's engine
process writes to the terminal directly, so its INFO lines appear on screen but not in the saved output.
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


class _Tee:
    """Write to the terminal and to the saved output file at the same time."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        if not self.fh.closed:  # shutdown messages after the file is closed still reach the terminal
            self.fh.write(s)
        return self.stream.write(s)

    def flush(self):
        if not self.fh.closed:
            self.fh.flush()
        self.stream.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno ... come from the real terminal stream
        return getattr(self.stream, name)


_out = None
if "--vllm-child" not in sys.argv and os.environ.get("RANK") in (None, "0"):
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    _out = open(os.path.join(ROOT, "outputs", f"{datetime.fromtimestamp(_start):%Y-%m-%d_%H%M%S}_{script}.log"),
                "w", encoding="utf-8")
    _out.write(f"$ python {' '.join(shlex.quote(a) for a in sys.argv)}\n"
               f"# started {datetime.fromtimestamp(_start).isoformat(timespec='seconds')} on {platform.node()}\n\n")
    sys.stdout, sys.stderr = _Tee(sys.stdout, _out), _Tee(sys.stderr, _out)

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
    if _out is not None:
        _out.write(f"\n# finished {entry['end']} ({entry['duration_s']} s), status: {entry['status']}\n")
        _out.close()
    gpus = "; ".join(f"{g['name']} {g['uuid']}" for g in entry["gpus"]) or "no GPU found"
    with open(TXT, "a", encoding="utf-8") as f:
        f.write(f"[{entry['start']} -> {entry['end']}] {entry['duration_s']:>8.1f} s  {entry['status']:<12} "
                f"{entry['command']}\n    host={entry['host']} world={entry['world_size'] or 1} gpus: {gpus}\n")
