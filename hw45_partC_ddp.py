"""HW4.5 Part C - distributed data parallel on the instructor-assigned two-GPU pairing.

Single-GPU baseline and the pair use the same script:
    torchrun --nproc_per_node=1 hw45_partC_ddp.py --slice <SLICE> --seed <SEED>           # baseline
    torchrun --nproc_per_node=2 hw45_partC_ddp.py --slice <SLICE> --seed <SEED>           # one node, 2 GPUs
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=<0|1> --master_addr=<ip> --master_port=29500 \
             hw45_partC_ddp.py ...                                                         # two nodes

For each per-GPU batch size it measures, over the same timed steps:
  * step time with DDP gradient all-reduce                     (t_sync)
  * step time inside model.no_sync(), i.e. no communication    (t_local, per rank -> straggler gap)
  * exposed communication overhead per step = t_sync - max_rank(t_local)
  * raw all-reduce of a gradient-sized fp32 buffer -> effective bus bandwidth
"""
import runlog  # noqa: F401  (rank 0 appends this run to RUN_LOG.txt)
import argparse
import contextlib
import os
import socket
import subprocess
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler

from common import TrainCfg, add_data_args, build_model, build_optimizer, env_info, prepare_data, save_json

p = add_data_args(argparse.ArgumentParser())
p.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
p.add_argument("--warmup", type=int, default=5)
p.add_argument("--timed", type=int, default=30)
p.add_argument("--tag", default="", help="name of the pairing, e.g. 4090x2 or 5090+4090")
a = p.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local)
dev = torch.device("cuda", local)


def log(*s):
    if rank == 0:
        print(*s, flush=True)


def allmax(x):
    t = torch.tensor([x], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def gather(obj):
    out = [None] * world
    dist.all_gather_object(out, obj)
    return out


# ---- topology
names, hosts = gather(torch.cuda.get_device_name(local)), gather(socket.gethostname())
topo = dict(world_size=world, gpus=names, hosts=hosts, homogeneous=len(set(names)) == 1,
            nodes=len(set(hosts)), single_node=len(set(hosts)) == 1,
            nccl_env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")})
if rank == 0 and topo["single_node"]:
    try:
        topo["nvidia_smi_topo"] = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        pass
    if torch.cuda.device_count() >= 2:
        topo["p2p_0_1"] = torch.cuda.can_device_access_peer(0, 1)
log("Topology:", {k: v for k, v in topo.items() if k != "nvidia_smi_topo"})

data = prepare_data(a)
ds = data["train_ds"]
rows = []

for bs in a.batch_sizes:
    cfg = TrainCfg(batch_size=bs, seed=a.seed)
    base = build_model(len(data["labels"]), cfg)  # same seed on every rank -> identical init
    model = DDP(base, device_ids=[local]) if world > 1 else base
    opt = build_optimizer(model, cfg)
    g = torch.Generator().manual_seed(a.seed + rank)  # each rank draws a different shard of samples
    n = (a.warmup + 2 * a.timed) * bs
    it = iter(DataLoader(ds, batch_size=bs, pin_memory=True,
                         sampler=RandomSampler(ds, replacement=True, num_samples=n, generator=g)))

    def steps(k, sync=True):
        ctx = model.no_sync() if (world > 1 and not sync) else contextlib.nullcontext()
        model.train()
        for _ in range(k):
            ids, mask, y = (t.to(dev, non_blocking=True) for t in next(it))
            with ctx:
                loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    def timed(sync):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        steps(a.timed, sync)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    try:
        torch.cuda.reset_peak_memory_stats()
        steps(a.warmup)
        t_sync = allmax(timed(True))
        t_local_ranks = gather(timed(False))
    except torch.cuda.OutOfMemoryError:
        log(f"bs={bs}: OOM")
        break
    t_local = max(t_local_ranks)
    step_sync, step_local = t_sync / a.timed, t_local / a.timed
    comm = max(step_sync - step_local, 0.0)
    r = dict(per_gpu_batch=bs, global_batch=bs * world, samples_per_s=round(bs * world * a.timed / t_sync, 1),
             step_ms=round(step_sync * 1000, 2), compute_step_ms=round(step_local * 1000, 2),
             compute_step_ms_per_rank=[round(t / a.timed * 1000, 2) for t in t_local_ranks],
             straggler_wait_ms=round((max(t_local_ranks) - min(t_local_ranks)) / a.timed * 1000, 2),
             comm_overhead_ms=round(comm * 1000, 2), comm_fraction=round(comm / step_sync, 3),
             peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3))
    rows.append(r)
    log(f"bs/gpu={bs:>4}: {r['samples_per_s']:>8} samp/s  step {r['step_ms']} ms  compute {r['compute_step_ms']} ms"
        f"  comm {r['comm_overhead_ms']} ms ({r['comm_fraction']:.0%})  straggler {r['straggler_wait_ms']} ms")
    del model, base, opt
    torch.cuda.empty_cache()

# ---- raw all-reduce bandwidth on a gradient-sized buffer (bert-base, fp32 grads)
bw = None
if world > 1:
    n_params = sum(p.numel() for p in build_model(len(data["labels"]), TrainCfg()).parameters())
    buf = torch.ones(n_params, dtype=torch.float32, device=dev)
    for _ in range(3):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(10):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    t = allmax((time.perf_counter() - t0) / 10)
    nbytes = buf.numel() * 4
    bw = dict(bytes=nbytes, allreduce_ms=round(t * 1000, 2), algbw_gbs=round(nbytes / t / 1e9, 2),
              busbw_gbs=round(nbytes / t / 1e9 * 2 * (world - 1) / world, 2))
    log("All-reduce bandwidth:", bw)

if rank == 0:
    suffix = f"_{a.tag}" if a.tag else ""
    save_json(dict(env=env_info(), topology=topo, rows=rows, allreduce=bw), f"partC_w{world}{suffix}.json")
dist.destroy_process_group()
