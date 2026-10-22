"""GPU UUID capture for the deliverables: run once on each card / machine you use.

    python capture_uuid.py            # writes captures/uuid_<GPU>_<host>_<timestamp>.txt
"""
import os
import platform
import re
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "captures"), exist_ok=True)


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"<failed: {e}>\n"


now = datetime.now()
listing = run(["nvidia-smi", "-L"])
first = re.search(r"GPU 0: (.+?) \(UUID", listing)
name = re.sub(r"[^A-Za-z0-9]+", "", (first.group(1) if first else "unknown").replace("NVIDIA", "").replace("GeForce", ""))
path = os.path.join(ROOT, "captures", f"uuid_{name}_{platform.node()}_{now:%Y%m%d_%H%M%S}.txt")
with open(path, "w", encoding="utf-8") as f:
    f.write(f"Captured: {now.isoformat(timespec='seconds')}\nHost: {platform.node()}\nOS: {platform.platform()}\n")
    f.write(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n\n")
    f.write("$ nvidia-smi -L\n" + listing + "\n")
    f.write("$ nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version,vbios_version "
            "--format=csv\n")
    f.write(run(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version,vbios_version",
                 "--format=csv"]) + "\n")
    f.write("$ nvidia-smi\n" + run(["nvidia-smi"]) + "\n")
    f.write("$ nvidia-smi -q\n" + run(["nvidia-smi", "-q"]))
print(listing.strip())
print(f"[saved] {path}")
