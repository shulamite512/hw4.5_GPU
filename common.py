"""Shared pieces for HW4.a and HW4.5: data slicing, tokenisation, model, train/eval loops, GPU stats.

Every script imports from here so that "HW4 settings" mean exactly the same thing everywhere.
"""
import argparse
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
MODEL_NAME = "bert-base-uncased"
SUBSET_ROWS = 5000
GIB = 1024 ** 3
os.makedirs(RESULTS, exist_ok=True)


# ---------------------------------------------------------------- environment

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gpu_tag():
    """Short filesystem-safe GPU name, e.g. 'RTX4090'. Used to keep results from different cards apart."""
    name = torch.cuda.get_device_name(0).replace("NVIDIA", "").replace("GeForce", "")
    return re.sub(r"[^A-Za-z0-9]+", "", name)


def env_info():
    import transformers
    p = torch.cuda.get_device_properties(0)
    return dict(
        gpu=torch.cuda.get_device_name(0),
        gpu_tag=gpu_tag(),
        vram_gib=round(p.total_memory / GIB, 2),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
        python=platform.python_version(),
        host=platform.node(),
    )


def save_json(obj, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] {path}")
    return path


def peak_mem():
    return dict(
        peak_alloc_gib=round(torch.cuda.max_memory_allocated() / GIB, 3),
        peak_reserved_gib=round(torch.cuda.max_memory_reserved() / GIB, 3),
    )


def is_oom(err):
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


# ---------------------------------------------------------------- data

def add_data_args(p: argparse.ArgumentParser):
    # personal parameters from SID4 = 1384: SEED = SID4, SLICE = SID4 % 1000
    p.add_argument("--slice", type=int, default=384, help="first row of the working subset (SLICE)")
    p.add_argument("--seed", type=int, default=1384, help="split + init seed (SEED)")
    p.add_argument("--csv", default=None,
                   help="course-provided Twitter CSV; omit to use cardiffnlp/tweet_eval (sentiment)")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--max-len", type=int, default=64, help="BertTokenizer max_length")
    return p


def load_twitter(csv=None, text_col="text", label_col="label"):
    """Full dataset as a DataFrame with columns text, label_raw, in file order."""
    if csv:
        df = pd.read_csv(csv, encoding="utf-8", encoding_errors="replace")
        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label_raw"})
        source = os.path.basename(csv)
    else:
        from datasets import load_dataset
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
        names = ds["train"].features["label"].names
        # train, validation, test concatenated in that order form the "row" numbering
        df = pd.concat([ds[s].to_pandas() for s in ("train", "validation", "test")], ignore_index=True)
        df["label_raw"] = df["label"].map(lambda i: names[i])
        df = df[["text", "label_raw"]]
        source = "cardiffnlp/tweet_eval (sentiment), splits train+validation+test concatenated"
    return df, source


def prepare_data(args):
    """Slice up to 5,000 consecutive rows from SLICE, split 70/15/15 with SEED, tokenise into TensorDatasets."""
    df, source = load_twitter(args.csv, args.text_col, args.label_col)
    if args.slice >= len(df):
        raise SystemExit(f"SLICE={args.slice} is past the end of the dataset ({len(df)} rows)")
    end = min(args.slice + SUBSET_ROWS, len(df))  # fewer than 5,000 left -> take all remaining rows
    sub = df.iloc[args.slice:end].copy()
    sub["row"] = sub.index  # 0-based row number in the full dataset
    n_dropped = int(sub[["text", "label_raw"]].isna().any(axis=1).sum())
    sub = sub.dropna(subset=["text", "label_raw"])
    sub["text"] = sub["text"].astype(str)

    labels = sorted(sub["label_raw"].astype(str).unique())
    l2i = {l: i for i, l in enumerate(labels)}
    sub["label"] = sub["label_raw"].astype(str).map(l2i)

    strat = sub["label"] if sub["label"].value_counts().min() >= 10 else None
    train, rest = train_test_split(sub, test_size=0.30, random_state=args.seed, stratify=strat)
    val, test = train_test_split(rest, test_size=0.50, random_state=args.seed,
                                 stratify=rest["label"] if strat is not None else None)

    tok = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)

    def encode(part):
        # tok(...) is batch_encode_plus; the method name itself was removed in transformers 5
        enc = tok(part["text"].tolist(), add_special_tokens=True, return_attention_mask=True,
                  padding="max_length", truncation=True, max_length=args.max_len, return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(part["label"].values))

    info = dict(source=source, dataset_rows=len(df), row_start=args.slice, row_end=end - 1,
                rows_used=end - args.slice, rows_dropped_nan=n_dropped, seed=args.seed, max_len=args.max_len,
                labels=labels, n_train=len(train), n_val=len(val), n_test=len(test),
                class_counts={labels[k]: int(v) for k, v in sub["label"].value_counts().sort_index().items()},
                stratified=strat is not None)
    return dict(info=info, labels=labels, tokenizer=tok, train_df=train, val_df=val, test_df=test,
                train_ds=encode(train), val_ds=encode(val), test_ds=encode(test))


# ---------------------------------------------------------------- model + training

@dataclass
class TrainCfg:
    """Defaults ARE the HW4 settings; HW4.5 techniques flip one field at a time."""
    batch_size: int = 32
    epochs: int = 4
    lr: float = 2e-5
    eps: float = 1e-8
    amp: bool = False        # bf16 autocast (Ada/Blackwell support bf16 natively, no GradScaler needed)
    grad_ckpt: bool = False
    accum: int = 1
    optim: str = "adamw"     # adamw | adamw8bit
    seed: int = 1384


def add_train_args(p: argparse.ArgumentParser):
    d = TrainCfg()
    p.add_argument("--bs", type=int, default=d.batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--optim", default="adamw", choices=["adamw", "adamw8bit"])
    return p


def cfg_from_args(a):
    return TrainCfg(batch_size=a.bs, epochs=a.epochs, lr=a.lr, amp=a.amp, grad_ckpt=a.grad_ckpt,
                    accum=a.accum, optim=a.optim, seed=a.seed)


def cap_to_physical_vram():
    """On Windows the NVIDIA driver's 'sysmem fallback' silently spills past VRAM into system RAM instead of
    raising OOM (seen at bs=1024: 39.9 GiB 'peak' on a 24 GiB card, 17x slower). Capping the caching allocator
    at physical VRAM restores a real CUDA OOM at the memory wall. Exactly 1.0 means "no cap" to PyTorch
    (verified: 40 GiB allocated at 1.0, OOM at 23 GiB with 0.99), so use 0.99."""
    torch.cuda.set_per_process_memory_fraction(0.99, torch.cuda.current_device())


def build_model(num_labels, cfg):
    cap_to_physical_vram()
    set_seed(cfg.seed)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    if cfg.grad_ckpt:
        model.gradient_checkpointing_enable()
    return model.cuda()


def build_optimizer(model, cfg):
    if cfg.optim == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=cfg.lr, eps=cfg.eps)
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=cfg.eps)


def train_epoch(model, loader, opt, sched, cfg, max_opt_steps=None):
    """One pass over `loader`. Returns (mean train loss, samples seen, optimiser steps taken)."""
    model.train()
    loss_sum = torch.zeros((), device="cuda")
    seen = steps = 0
    n_batches = len(loader)
    opt.zero_grad(set_to_none=True)
    for i, (ids, mask, y) in enumerate(loader):
        ids, mask, y = ids.cuda(non_blocking=True), mask.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
        (loss / cfg.accum).backward()
        loss_sum += loss.detach().float() * y.size(0)  # stays on GPU: no per-step host sync
        seen += y.size(0)
        if (i + 1) % cfg.accum == 0 or i + 1 == n_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            if max_opt_steps and steps >= max_opt_steps:
                break
    return (loss_sum / max(seen, 1)).item(), seen, steps


@torch.no_grad()
def evaluate(model, dataset, amp=False, batch_size=128):
    model.eval()
    loss_sum, preds, labels = 0.0, [], []
    for ids, mask, y in DataLoader(dataset, batch_size=batch_size):
        ids, mask, y = ids.cuda(), mask.cuda(), y.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(input_ids=ids, attention_mask=mask, labels=y)
        loss_sum += out.loss.float().item() * y.size(0)
        preds.append(out.logits.argmax(-1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return loss_sum / len(labels), float((preds == labels).mean()), preds, labels


def fit(data, cfg, keep_best=False, log=print):
    """Full fine-tune with per-epoch validation. Timing starts after the model is on the GPU."""
    model = build_model(len(data["labels"]), cfg)
    opt = build_optimizer(model, cfg)
    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(data["train_ds"], sampler=RandomSampler(data["train_ds"], generator=g),
                          batch_size=cfg.batch_size, pin_memory=True)
    total_steps = math.ceil(len(train_dl) / cfg.accum) * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=total_steps)

    history, best, best_state = [], None, None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(1, cfg.epochs + 1):
        te = time.perf_counter()
        tr_loss, seen, _ = train_epoch(model, train_dl, opt, sched, cfg)
        torch.cuda.synchronize()
        t_train = time.perf_counter() - te
        va_loss, va_acc, _, _ = evaluate(model, data["val_ds"], cfg.amp)
        row = dict(epoch=ep, train_loss=round(tr_loss, 4), val_loss=round(va_loss, 4), val_acc=round(va_acc, 4),
                   epoch_train_s=round(t_train, 2), samples_per_s=round(seen / t_train, 1),
                   elapsed_s=round(time.perf_counter() - t0, 2))
        history.append(row)
        log(f"epoch {ep}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
            f"({t_train:.1f}s, {row['samples_per_s']} samp/s)")
        if keep_best and (best is None or va_loss < best["val_loss"]):
            best = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.cuda.synchronize()
    stats = dict(wall_clock_s=round(time.perf_counter() - t0, 2),
                 train_only_s=round(sum(h["epoch_train_s"] for h in history), 2), **peak_mem())
    return model, history, stats, best_state


def probe(data, cfg, warmup=3, timed=20):
    """Throughput/memory probe: `warmup` + `timed` optimiser steps on batches sampled with replacement
    from the training split (so batch size is not capped by the 3,500-row split). Raises on OOM."""
    model = build_model(len(data["labels"]), cfg)
    opt = build_optimizer(model, cfg)
    g = torch.Generator().manual_seed(cfg.seed)

    def loader(n_steps):
        n = n_steps * cfg.accum * cfg.batch_size
        return DataLoader(data["train_ds"], batch_size=cfg.batch_size, pin_memory=True,
                          sampler=RandomSampler(data["train_ds"], replacement=True, num_samples=n, generator=g))

    torch.cuda.reset_peak_memory_stats()
    train_epoch(model, loader(warmup), opt, None, cfg)
    timed_dl = loader(timed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, seen, steps = train_epoch(model, timed_dl, opt, None, cfg)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = seen / dt
    return dict(batch_size=cfg.batch_size, accum=cfg.accum, effective_batch=cfg.batch_size * cfg.accum,
                amp=cfg.amp, grad_ckpt=cfg.grad_ckpt, optim=cfg.optim,
                timed_opt_steps=steps, timed_s=round(dt, 3), samples_per_s=round(sps, 1),
                s_per_opt_step=round(dt / steps, 4),
                time_per_epoch_s=round(data["info"]["n_train"] / sps, 2),  # one pass over the 70% split
                **peak_mem())


# ---------------------------------------------------------------- probe in a fresh process

def probe_subprocess(data_args, cfg: TrainCfg, warmup=3, timed=20):
    """Run bench_one.py in a new process so an OOM can't fragment or poison this process's allocator."""
    cmd = [sys.executable, os.path.join(ROOT, "bench_one.py"), *data_args,
           "--bs", str(cfg.batch_size), "--accum", str(cfg.accum), "--optim", cfg.optim,
           "--warmup", str(warmup), "--timed", str(timed)]
    if cfg.amp:
        cmd.append("--amp")
    if cfg.grad_ckpt:
        cmd.append("--grad-ckpt")
    out = subprocess.run(cmd, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(f"bench_one failed:\n{out.stdout[-2000:]}\n{out.stderr[-4000:]}")


def sweep_batches(data_args, cfg: TrainCfg, start=1, limit=1 << 16, log=print, **kw):
    """Double the batch size from `start` until OOM. Returns (rows that ran, first batch size that OOMed)."""
    rows, bs = [], start
    while bs <= limit:
        c = TrainCfg(**{**asdict(cfg), "batch_size": bs})
        r = probe_subprocess(data_args, c, **kw)
        if r.get("oom"):
            log(f"  bs={bs:>6}: OOM")
            return rows, bs
        log(f"  bs={bs:>6}: {r['samples_per_s']:>8} samp/s  peak {r['peak_alloc_gib']} GiB  "
            f"epoch {r['time_per_epoch_s']} s")
        rows.append(r)
        bs *= 2
    return rows, None


def data_argv(a):
    """Re-serialise the data args so child processes see the same slice/seed/csv."""
    argv = ["--slice", str(a.slice), "--seed", str(a.seed), "--max-len", str(a.max_len),
            "--text-col", a.text_col, "--label-col", a.label_col]
    if a.csv:
        argv += ["--csv", a.csv]
    return argv
