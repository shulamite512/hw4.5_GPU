"""HW4.5 Part B - memory techniques, one at a time, then the best combination.

For each technique:
  * max batch    - doubling sweep until OOM (fresh process per point)
  * throughput + peak memory at the HW4 batch size (32), so every technique is compared at equal work
  * eval accuracy - full HW4 fine-tune (same epochs/lr/seed) with only that technique switched on
Gradient accumulation is measured differently (see the assignment): physical batch --accum-micro, effective
batch --accum-eff, compared with a plain run at physical batch = effective batch over the same optimiser steps.

    python hw45_partB_memory.py --slice <SLICE> --seed <SEED>
"""
import runlog  # noqa: F401  (appends this run to RUN_LOG.txt)
import argparse
import gc
from dataclasses import asdict, replace

import torch

from common import (TrainCfg, add_data_args, data_argv, env_info, evaluate, fit, gpu_tag, prepare_data,
                    probe_subprocess, save_json, sweep_batches)

p = add_data_args(argparse.ArgumentParser())
p.add_argument("--start", type=int, default=32, help="first batch size of each max-batch sweep")
p.add_argument("--accum-micro", type=int, default=8)
p.add_argument("--accum-eff", type=int, default=128)
p.add_argument("--accum-steps", type=int, default=20, help="optimiser steps for the wall-clock comparison")
p.add_argument("--only", nargs="*", help="subset of techniques to run")
a = p.parse_args()

hw4 = TrainCfg(seed=a.seed)
TECHNIQUES = {
    "baseline": hw4,
    "amp": replace(hw4, amp=True),
    "grad_ckpt": replace(hw4, grad_ckpt=True),
    "adamw8bit": replace(hw4, optim="adamw8bit"),
    "combo_amp_8bit": replace(hw4, amp=True, optim="adamw8bit"),
    "combo_amp_8bit_ckpt": replace(hw4, amp=True, optim="adamw8bit", grad_ckpt=True),
}
argv = data_argv(a)
data = prepare_data(a)
results = {}


def accuracy(cfg):
    model, history, stats, _ = fit(data, cfg, log=lambda s: print("   ", s))
    _, acc, _, _ = evaluate(model, data["test_ds"], cfg.amp)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return round(acc, 4), history, stats


for name, cfg in TECHNIQUES.items():
    if a.only and name not in a.only:
        continue
    print(f"\n== {name}: max-batch sweep")
    rows, oom_at = sweep_batches(argv, cfg, start=a.start)
    at32 = next(r for r in rows if r["batch_size"] == hw4.batch_size) if a.start <= 32 else \
        probe_subprocess(argv, cfg)
    print(f"== {name}: accuracy run (HW4 recipe)")
    acc, history, fit_stats = accuracy(cfg)
    results[name] = dict(cfg=asdict(cfg), max_batch=rows[-1]["batch_size"] if rows else None, oom_at=oom_at,
                         sweep=rows, at_bs32=at32, at_max=rows[-1] if rows else None,
                         test_acc=acc, history=history, fit_stats=fit_stats)
    print(f"== {name}: max batch {results[name]['max_batch']}, {at32['samples_per_s']} samp/s @32, "
          f"peak {at32['peak_alloc_gib']} GiB @32, test acc {acc}")

if not a.only or "accum" in a.only:
    k = a.accum_eff // a.accum_micro
    print(f"\n== grad accumulation: physical {a.accum_micro} x {k} = effective {a.accum_micro * k}")
    acc_cfg = replace(hw4, batch_size=a.accum_micro, accum=k)
    accum_run = probe_subprocess(argv, acc_cfg, warmup=2, timed=a.accum_steps)
    direct = probe_subprocess(argv, replace(hw4, batch_size=a.accum_eff), warmup=2, timed=a.accum_steps)
    acc, history, fit_stats = accuracy(acc_cfg)
    results["accum"] = dict(cfg=asdict(acc_cfg), effective_batch=a.accum_micro * k, physical_batch=a.accum_micro,
                            opt_steps=a.accum_steps, accum_probe=accum_run, direct_probe=direct,
                            test_acc=acc, history=history, fit_stats=fit_stats)
    print(f"   {a.accum_steps} optimiser steps: accum {accum_run['timed_s']} s / {accum_run['peak_alloc_gib']} GiB"
          f" vs direct bs={a.accum_eff}: {direct.get('timed_s', 'OOM')} s / {direct.get('peak_alloc_gib')} GiB;"
          f" test acc {acc}")

save_json(dict(env=env_info(), data=data["info"], results=results), f"partB_{gpu_tag()}.json")
