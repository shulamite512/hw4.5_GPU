"""HW4.5 Part A - batch scaling to the memory wall, HW4 fine-tune otherwise unchanged.

Doubles batch size from --start until CUDA OOM. Each point is a fresh process: 3 warm-up + 20 timed
optimiser steps. Run once per card; results are tagged with the GPU name.

    python hw45_partA_batch_scaling.py --slice <SLICE> --seed <SEED>
"""
import runlog  # noqa: F401  (appends this run to RUN_LOG.txt)
import argparse

from common import TrainCfg, add_data_args, data_argv, env_info, gpu_tag, save_json, sweep_batches

p = add_data_args(argparse.ArgumentParser())
p.add_argument("--start", type=int, default=1)
p.add_argument("--timed", type=int, default=20)
a = p.parse_args()

print(f"Part A on {gpu_tag()}: doubling batch size from {a.start} until OOM")
rows, oom_at = sweep_batches(data_argv(a), TrainCfg(seed=a.seed), start=a.start, timed=a.timed)
best = max(rows, key=lambda r: r["samples_per_s"])
print(f"Largest batch that runs: {rows[-1]['batch_size']} (OOM at {oom_at}); peak throughput "
      f"{best['samples_per_s']} samp/s at bs={best['batch_size']}")
save_json(dict(env=env_info(), rows=rows, max_batch=rows[-1]["batch_size"], oom_at=oom_at),
          f"partA_{gpu_tag()}.json")
