"""One throughput/memory probe at one configuration. Called in a fresh process by common.sweep_batches.

Prints a single line `RESULT {...json...}`; on OOM the JSON is {"oom": true, ...}.
"""
import argparse
import json

import torch

from common import add_data_args, add_train_args, cfg_from_args, is_oom, prepare_data, probe

p = add_train_args(add_data_args(argparse.ArgumentParser()))
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--timed", type=int, default=20)
a = p.parse_args()

cfg = cfg_from_args(a)
data = prepare_data(a)
try:
    r = probe(data, cfg, warmup=a.warmup, timed=a.timed)
except Exception as e:  # noqa: BLE001 - we only swallow OOMs
    if not is_oom(e):
        raise
    r = dict(oom=True, batch_size=cfg.batch_size, accum=cfg.accum, peak_reserved_gib=round(
        torch.cuda.max_memory_reserved() / 1024 ** 3, 3))
print("RESULT " + json.dumps(r), flush=True)
